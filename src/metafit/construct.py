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


def straighten(feature: str, columns: dict[str, np.ndarray], target: np.ndarray) -> Atom:
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
        if not is_admissible(values):
            continue
        score = linearity(values, target)
        if score > best_score:
            best, best_score = atom, score
    return best


def _candidate_terms(left: Term, right: Term, columns: dict[str, np.ndarray]) -> list[Term]:
    """Every way the grammar allows two terms to be joined.

    Only atoms can be composed, because the grammar's operands are atoms -- a merged term
    is re-entered into the pool as itself and can still be *selected*, but it cannot be
    nested further. That ceiling is deliberate: unbounded nesting is what makes symbolic
    regression output unreadable, and this project's premise is that the equation must be
    readable.
    """
    if left.operation != "atom" or right.operation != "atom":
        return []
    a, b = left.operands[0], right.operands[0]
    if a.feature == b.feature:
        return []

    candidates = [Term("product", (a, b))]
    divisor_b = denominator_atom(b.feature, columns)
    if divisor_b is not None:
        candidates.append(Term("ratio", (a, divisor_b)))
    divisor_a = denominator_atom(a.feature, columns)
    if divisor_a is not None:
        candidates.append(Term("ratio", (b, divisor_a)))
    return candidates


def agglomerate(
    features: tuple[str, ...],
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    min_gain: float = MIN_MERGE_GAIN,
    per_round: bool = True,
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
    active: list[Term] = [Term("atom", (straighten(feature, columns, target),)) for feature in features]
    produced: dict[str, Term] = {term.name: term for term in active}
    scores = [linearity(term.evaluate(columns), target) for term in active]

    if not per_round:
        while len(active) > 1:
            best: Merge | None = None
            for i, j in itertools.combinations(range(len(active)), 2):
                floor = max(scores[i], scores[j])
                for candidate in _candidate_terms(active[i], active[j], columns):
                    gain = _gain(candidate, columns, target, floor)
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

    merged: list[Term] = []
    for i in range(len(active)):
        best_for_i: Merge | None = None
        for j in range(len(active)):
            if i == j:
                continue
            floor = max(scores[i], scores[j])
            for candidate in _candidate_terms(active[i], active[j], columns):
                gain = _gain(candidate, columns, target, floor)
                if gain is not None and gain > min_gain and (best_for_i is None or gain > best_for_i.gain):
                    best_for_i = Merge(candidate, gain, (i, j))
        if best_for_i is not None:
            merged.append(best_for_i.term)
    for term in merged:
        produced.setdefault(term.name, term)
    return list(produced.values())


def _gain(term: Term, columns: dict[str, np.ndarray], target: np.ndarray, floor: float) -> float | None:
    """Linearity gained over the better parent, or ``None`` if the term is unusable."""
    with np.errstate(all="ignore"):
        values = term.evaluate(columns)
    if not is_admissible(values):
        return None
    return linearity(values, target) - floor


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
) -> Library:
    """A library built by agglomeration rather than enumeration.

    Dataset and model features are agglomerated separately and then together, so the
    process gets a chance to build within-group structure before it starts pairing across
    the groups -- the cross terms are where E2's lift comes from, but they are also the
    ones most able to overfit, so they are not the first thing tried.
    """
    terms = agglomerate(dataset_features, columns, target, min_gain=min_gain)
    if model_features:
        terms += agglomerate(model_features, columns, target, min_gain=min_gain)
        terms += agglomerate(dataset_features + model_features, columns, target, min_gain=min_gain)
    return Library(terms, columns, max_abs_zscore=max_abs_zscore)
