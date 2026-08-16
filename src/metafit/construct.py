"""Building terms by agglomeration instead of enumerating a fixed library.

`metafit.terms` enumerates every expression the grammar allows and then screens the
result. That is exhaustive within its depth limit but blind: it cannot look at *why* a
feature relates to MCC the way it does, and its depth limit is what stops the library
exploding.

This module takes the opposite approach, and it is structurally hierarchical clustering.
Every feature starts as its own singleton. At each step the pair whose merge -- under one
of the grammar's operations -- becomes **most linear in MCC** is joined, and the pair is
replaced by the merged term. Features that no merge improves are left isolated. The
process stops when no merge helps, so the depth is discovered rather than imposed.

The linkage criterion is linearity because linearity is exactly what the downstream model
can use. An equation is a weighted sum of terms; a term that is monotone in MCC but
strongly curved contributes badly to that sum no matter what weight it gets, while a term
that is linear in MCC contributes perfectly with weight 1. So "how linear is this term in
the target" is not a proxy for term quality here -- it is the definition of it.

Pearson and Spearman are both used, and their disagreement drives the unary step:
Spearman much larger than Pearson means the relation is monotone but curved, which is
precisely the situation a transform repairs. The agglomeration handles pairs; the unary
step handles the curvature of a single feature.

Study chapter: [3. Search and fitting](../../assets/docs/03-search-and-fitting.md) -- the
rationale, in prose, with the figures.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from metafit.stats import pearson, spearman
from metafit.terms import (
    DENOMINATOR_FLOOR,
    MAX_ABS_ZSCORE,
    MAX_DENOMINATOR_RANGE,
    TRANSFORMS,
    Atom,
    Library,
    Term,
    denominator_atom,
    is_admissible,
)

#: A merge has to add at least this much absolute Pearson correlation over the better of
#: its two parents before it is worth making. Without a floor the process merges
#: everything into a single deep expression for gains in the fourth decimal place, which
#: is how symbolic regression produces the unreadable results this project exists to avoid.
MIN_MERGE_GAIN = 0.01


def linearity(values: np.ndarray, target: np.ndarray) -> float:
    """How linear a term is in the target: absolute Pearson correlation."""
    return abs(pearson(values, target))


def curvature(values: np.ndarray, target: np.ndarray) -> float:
    """How much of the term's association with the target a straight line misses.

    Positive means monotone but curved -- a transform should help. Near zero means the
    relation is already as linear as it is monotone, and nothing is left to straighten.
    """
    return abs(spearman(values, target)) - abs(pearson(values, target))


@dataclass(frozen=True)
class Merge:
    """One candidate join, and what it would buy."""

    term: Term
    gain: float
    parents: tuple[int, int]


def straighten(
    feature: str,
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> Atom:
    """The transform of ``feature`` that is most linear in the target.

    This is the unary half of the idea. A feature whose Spearman correlation far exceeds
    its Pearson is monotone but curved; trying the grammar's transforms and keeping the
    most linear one is the direct repair.
    """
    best = Atom(feature, "id")
    best_score = -1.0
    for transform in TRANSFORMS:
        atom = Atom(feature, transform)
        if not atom.is_defined_on(columns):
            continue
        with np.errstate(all="ignore"):
            values = atom.evaluate(columns)
        if not is_admissible(values, max_abs_zscore):
            continue
        score = linearity(values, target)
        if score > best_score:
            best, best_score = atom, score
    return best


def is_safe_divisor(values: np.ndarray) -> bool:
    """Whether a column can be divided by: clear of zero, bounded dynamic range.

    The value-based counterpart of ``terms.denominator_atom``. A nested term has no single
    feature whose eligibility could be looked up, so the same two conditions are checked
    against what it actually computes.
    """
    magnitude = np.abs(values)
    if not np.all(np.isfinite(magnitude)):
        return False
    smallest = float(np.min(magnitude))
    if smallest <= DENOMINATOR_FLOOR:
        return False
    return float(np.max(magnitude)) / smallest <= MAX_DENOMINATOR_RANGE


def _operand(term: Term) -> Atom | Term:
    """Unwrap a singleton so ``atom`` terms nest as their atom rather than as a wrapper."""
    return term.operands[0] if term.operation == "atom" else term


def _candidate_terms(
    left: Term,
    right: Term,
    columns: dict[str, np.ndarray],
    max_depth: int = 1,
) -> list[Term]:
    """Every way the grammar allows two terms to be joined, up to ``max_depth`` nesting.

    With ``max_depth=1`` only atoms compose and the result is flat, matching
    ``terms.build_library``. Above that, an already-merged term may be merged again, so a
    single term can carry a deep expression.

    That is the trade this project has to make consciously. The equation's length is
    counted in terms, so nesting buys expressiveness without lengthening the equation --
    but unbounded nesting is exactly what makes genetic-programming output unreadable. The
    cap is the control, and the depth that actually pays is measured rather than assumed.
    """
    # Bound the depth of the *result*, not of the operands. An "atom" term unwraps to a
    # bare Atom and adds no depth; anything else contributes its own.
    nested = [term.depth for term in (left, right) if term.operation != "atom"]
    if 1 + max(nested, default=0) > max_depth:
        return []
    if set(left.features) & set(right.features):
        return []

    a, b = _operand(left), _operand(right)
    candidates = [Term("product", (a, b))]
    for numerator, divisor_term, divisor in ((a, right, b), (b, left, a)):
        if divisor_term.operation == "atom":
            eligible = denominator_atom(divisor_term.operands[0].feature, columns)  # pyright: ignore[reportAttributeAccessIssue]
            if eligible is not None:
                candidates.append(Term("ratio", (numerator, eligible)))
        else:
            with np.errstate(all="ignore"):
                if is_safe_divisor(divisor_term.evaluate(columns)):
                    candidates.append(Term("ratio", (numerator, divisor)))
    return candidates


def agglomerate(
    features: tuple[str, ...],
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    min_gain: float = MIN_MERGE_GAIN,
    per_round: bool = True,
    max_depth: int = 1,
    rounds: int = 1,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> list[Term]:
    """Merge features pairwise by linearity gain, keeping everything the process builds.

    Two modes, and the difference matters more than it looks.

    Strict clustering (``per_round=False``) joins the single best pair, *consumes* both
    parents, and repeats. That is hierarchical clustering exactly, and on this data it
    yields about eight terms from seventeen features -- structurally too few to build a
    fourteen-term equation from, whatever their quality.

    Round-based (the default) instead lets every active term find its own best partner in
    each round and keeps the parents available. It is the same linkage criterion and the
    same stopping rule, but it produces a pool rather than a tree, which is what the
    selector downstream actually needs.
    """
    active: list[Term] = [Term("atom", (straighten(feature, columns, target, max_abs_zscore),)) for feature in features]
    produced: dict[str, Term] = {term.name: term for term in active}
    scores = [linearity(term.evaluate(columns), target) for term in active]

    if not per_round:
        while len(active) > 1:
            best: Merge | None = None
            for i, j in itertools.combinations(range(len(active)), 2):
                floor = max(scores[i], scores[j])
                for candidate in _candidate_terms(active[i], active[j], columns, max_depth):
                    gain = _gain(candidate, columns, target, floor, max_abs_zscore)
                    if gain is not None and gain > min_gain and (best is None or gain > best.gain):
                        best = Merge(candidate, gain, (i, j))
            if best is None:
                break
            produced[best.term.name] = best.term
            for position in sorted(best.parents, reverse=True):
                del active[position]
                del scores[position]
            active.append(best.term)
            scores.append(linearity(best.term.evaluate(columns), target))
        return list(produced.values())

    for _ in range(rounds):
        merged: list[Term] = []
        for i in range(len(active)):
            best_for_i: Merge | None = None
            for j in range(len(active)):
                if i == j:
                    continue
                floor = max(scores[i], scores[j])
                for candidate in _candidate_terms(active[i], active[j], columns, max_depth):
                    gain = _gain(candidate, columns, target, floor, max_abs_zscore)
                    if gain is not None and gain > min_gain and (best_for_i is None or gain > best_for_i.gain):
                        best_for_i = Merge(candidate, gain, (i, j))
            if best_for_i is not None:
                merged.append(best_for_i.term)
        fresh = [term for term in merged if term.name not in produced]
        if not fresh:
            break
        for term in fresh:
            produced[term.name] = term
        active = active + fresh
        scores = scores + [linearity(term.evaluate(columns), target) for term in fresh]
    return list(produced.values())


def _gain(
    term: Term,
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    floor: float,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> float | None:
    """Linearity gained over the better parent, or ``None`` if the term is unusable.

    ``max_abs_zscore`` must match whatever the resulting library will be built with.
    Constructing under a loose cap and then filtering under a strict one silently discards
    most of what was built, and comparing such a library against an enumerated one is not
    a comparison of methods but of admissibility settings.
    """
    with np.errstate(all="ignore"):
        values = term.evaluate(columns)
    if not is_admissible(values, max_abs_zscore):
        return None
    return linearity(values, target) - floor


def cluster_terms(
    features: tuple[str, ...],
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    n_terms: int,
    *,
    max_depth: int = 3,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> list[Term]:
    """Cut the dendrogram at ``n_terms`` clusters and return them as the equation's terms.

    This is the idea in its purest form. Strict agglomeration builds a dendrogram over the
    features; stopping the merge when ``n_terms`` clusters remain and reading those
    clusters off *is* the equation. There is no subset selection afterwards -- the cut
    level is the equation length, and each surviving cluster is one term.

    It is a different division of labour from the rest of this package, where a library is
    proposed and a beam search picks from it. Here the construction chooses, and the only
    thing left for least squares is the weights. That makes the result far cheaper and
    arguably easier to defend -- there is one mechanism rather than two -- and whether it
    is also *better* is measured rather than assumed.

    Merging continues past the point where linearity stops improving when it has to, since
    the cut level is fixed by ``n_terms`` rather than by a gain threshold. A cluster that
    no longer improves is still merged with its least-bad partner, which is the price of
    letting the caller choose the equation length.
    """
    active: list[Term] = [Term("atom", (straighten(feature, columns, target, max_abs_zscore),)) for feature in features]
    if n_terms >= len(active):
        return active

    scores = [linearity(term.evaluate(columns), target) for term in active]
    while len(active) > n_terms:
        best: Merge | None = None
        for i, j in itertools.combinations(range(len(active)), 2):
            floor = max(scores[i], scores[j])
            for candidate in _candidate_terms(active[i], active[j], columns, max_depth):
                gain = _gain(candidate, columns, target, floor, max_abs_zscore)
                if gain is not None and (best is None or gain > best.gain):
                    best = Merge(candidate, gain, (i, j))
        if best is None:
            break
        for position in sorted(best.parents, reverse=True):
            del active[position]
            del scores[position]
        active.append(best.term)
        scores.append(linearity(best.term.evaluate(columns), target))
    return active


def dendrogram_terms(
    features: tuple[str, ...],
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    max_depth: int = 3,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> list[Term]:
    """Every node of the full dendrogram: the leaves and each merge that was accepted.

    The hybrid. ``cluster_terms`` stops the merge at a chosen height and hands those
    clusters over as the equation; this instead runs the merge all the way to a single
    cluster and keeps **every intermediate node** as a candidate. Construction then
    proposes and selection disposes -- the dendrogram supplies a small, structured pool
    and the beam search picks the ``k`` that fit best together.

    That division is worth having because the two mechanisms fail differently.
    Agglomeration is greedy and pairwise: it can commit early to a merge that is good on
    its own and redundant beside the rest of the equation. Selection over the whole
    dendrogram can decline the merge and keep its parent instead, which a fixed cut
    cannot. The pool stays small -- ``2n - 1`` nodes for ``n`` features, so 33 here
    against the enumerated library's 172 -- which is the point: it is a *shortlist* with a
    reason behind every entry.
    """
    active: list[Term] = [Term("atom", (straighten(feature, columns, target, max_abs_zscore),)) for feature in features]
    produced: dict[str, Term] = {term.name: term for term in active}
    scores = [linearity(term.evaluate(columns), target) for term in active]

    while len(active) > 1:
        best: Merge | None = None
        for i, j in itertools.combinations(range(len(active)), 2):
            floor = max(scores[i], scores[j])
            for candidate in _candidate_terms(active[i], active[j], columns, max_depth):
                gain = _gain(candidate, columns, target, floor, max_abs_zscore)
                if gain is not None and (best is None or gain > best.gain):
                    best = Merge(candidate, gain, (i, j))
        if best is None:
            break
        produced.setdefault(best.term.name, best.term)
        for position in sorted(best.parents, reverse=True):
            del active[position]
            del scores[position]
        active.append(best.term)
        scores.append(linearity(best.term.evaluate(columns), target))
    return list(produced.values())


def merge_until_linear(
    features: tuple[str, ...],
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    min_linearity: float = 0.45,
    slack: float = 0.005,
    max_depth: int = 4,
    max_abs_zscore: float = MAX_ABS_ZSCORE,
) -> list[Term]:
    """Merge until every surviving cluster is linear enough in MCC, or nothing helps.

    The parameter-free variant, in the sense that matters: there is no equation length to
    choose. Merging continues while some cluster still falls short of ``min_linearity``
    **and** a merge would improve the worst offender by more than ``slack``. The number of
    terms is then whatever the data leaves standing.

    ``slack`` is a tolerance rather than a target -- it says how much improvement is worth
    a merge, not how many terms to end with -- which is the substantive difference from
    ``cluster_terms``. Setting ``min_linearity`` above what any term achieves degenerates
    to merging everything into one cluster, so the stopping rule is really the conjunction
    of the two conditions rather than the threshold alone.
    """
    active: list[Term] = [Term("atom", (straighten(feature, columns, target, max_abs_zscore),)) for feature in features]
    scores = [linearity(term.evaluate(columns), target) for term in active]

    while len(active) > 1 and min(scores) < min_linearity:
        weakest = int(np.argmin(scores))
        best: Merge | None = None
        for other in range(len(active)):
            if other == weakest:
                continue
            floor = max(scores[weakest], scores[other])
            for candidate in _candidate_terms(active[weakest], active[other], columns, max_depth):
                gain = _gain(candidate, columns, target, floor, max_abs_zscore)
                if gain is not None and gain > slack and (best is None or gain > best.gain):
                    best = Merge(candidate, gain, (weakest, other))
        if best is None:
            # The weakest cluster cannot be improved; nothing else will improve it either,
            # so leave it isolated and stop rather than merging for its own sake.
            break
        for position in sorted(best.parents, reverse=True):
            del active[position]
            del scores[position]
        active.append(best.term)
        scores.append(linearity(best.term.evaluate(columns), target))
    return active


def structural_terms(
    features: tuple[str, ...],
    columns: dict[str, np.ndarray],
    *,
    min_correlation: float = 0.5,
) -> list[Term]:
    """Terms built from how features relate to *each other*, never to the target.

    Two strongly correlated features share a component -- usually scale, since most of
    these meta-features grow with dataset size. Their ratio cancels it and isolates what
    distinguishes them, and their product amplifies it. Neither operation needs to know
    what MCC is.

    That independence is the point. Target-driven construction has to be redone inside
    every cross-validation fold and is charged for the target it consumed;
    this is a function of the feature columns alone, so it can be built once and costs
    nothing at validation time -- exactly the accounting that applies to
    ``terms.build_library``.
    """
    atoms = {feature: _scale_atom(feature, columns) for feature in features}
    values = {feature: atom.evaluate(columns) for feature, atom in atoms.items()}

    produced: dict[str, Term] = {}
    for first, second in itertools.combinations(features, 2):
        if abs(pearson(values[first], values[second])) < min_correlation:
            continue
        candidates = [Term("product", (atoms[first], atoms[second]))]
        for numerator, denominator in ((first, second), (second, first)):
            divisor = denominator_atom(denominator, columns)
            if divisor is not None:
                candidates.append(Term("ratio", (atoms[numerator], divisor)))
        for term in candidates:
            with np.errstate(all="ignore"):
                if is_admissible(term.evaluate(columns)):
                    produced.setdefault(term.name, term)
    return list(produced.values())


def _scale_atom(feature: str, columns: dict[str, np.ndarray]) -> Atom:
    """Log-compress a strictly positive feature, leave the rest alone."""
    return Atom(feature, "log" if bool(np.all(columns[feature] > 0.0)) else "id")


def constructed_library(
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    min_gain: float = MIN_MERGE_GAIN,
    max_abs_zscore: float = 3.0,
    max_depth: int = 1,
    rounds: int = 1,
) -> Library:
    """A library built by agglomeration rather than enumeration.

    Dataset and model features are agglomerated separately and then together, so the
    process gets a chance to build within-group structure before it starts pairing across
    the groups -- the cross terms are where E2's lift comes from, but they are also the
    ones most able to overfit, so they are not the first thing tried.
    """
    kwargs = {
        "min_gain": min_gain,
        "max_depth": max_depth,
        "rounds": rounds,
        "max_abs_zscore": max_abs_zscore,
    }
    terms = agglomerate(dataset_features, columns, target, **kwargs)
    if model_features:
        terms += agglomerate(model_features, columns, target, **kwargs)
        terms += agglomerate(dataset_features + model_features, columns, target, **kwargs)
    return Library(terms, columns, max_abs_zscore=max_abs_zscore)
