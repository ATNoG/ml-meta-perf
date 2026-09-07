"""Selecting which terms enter the equation, and with what weights.

Four ideas do the work here.

*Correlation-guided term design.* Pearson and Spearman disagree in a informative way.
A feature whose Pearson and Spearman correlations against MCC are similar is related to
the target linearly, and ``f`` on its own is the right term -- adding ``log(f)`` and
``1/f`` for it only inflates the search space. A feature whose Spearman correlation is
much the larger is monotonically but *non*-linearly related, and that is precisely the
case where a compressing or inverting transform earns its place. Screening on that gap
shrinks the library before search begins, which is what keeps the selected equations
short without a parsimony penalty bolted on afterwards.

*Beam search over subsets, not greedy descent.* Greedy forward selection commits to its
first term forever, and on collinear libraries that first commitment is often wrong.
A beam keeps several partial equations alive, and because the beam records its best
subset at every size, one run yields the whole accuracy-versus-number-of-terms curve.

*Gram-matrix arithmetic.* Every candidate refit is a k-by-k solve against a precomputed
Gram matrix rather than a least-squares call against the full design, which is what
makes an exhaustive sweep inside 20 cross-validation folds tractable.

*One term per feature combination.* An equation that describes ``a`` against ``b`` twice --
once as ``a/b`` and once as ``b/a`` -- cannot be read a term at a time, because neither
term means on its own what the table says it means. The selector therefore refuses a
candidate whose feature combination is already represented, which is a constraint on
*form* rather than on fit and is not negotiable against R2: an equation nobody can reason
about has failed at the only thing this study asks of it. It is measured, not assumed, to
cost nothing -- see `terms.Library.feature_groups`. Note that this is a different test
from ``COLLINEARITY_LIMIT``, which asks a numerical question; the pairs this removes were
well inside that limit.

Study chapter: [3. Term generation and selection](../../assets/docs/03-term-selection.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_meta_perf import beam
from ml_meta_perf.beam import VANILLA, BeamPolicy
from ml_meta_perf.model import Equation
from ml_meta_perf.stats import pearson, pearson_columns, rank_columns, rankdata, spearman
from ml_meta_perf.terms import Library, Term, is_trivial, simplify

RIDGE_DEFAULT = 10.0
BEAM_WIDTH_DEFAULT = 6
CANDIDATE_POOL_DEFAULT = 24
COLLINEARITY_LIMIT = 0.95
REFINE_ROUNDS_DEFAULT = 2


@dataclass(frozen=True)
class Standardizer:
    """Column means and scales, learned on training rows only."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> Standardizer:
        scale = matrix.std(axis=0)
        return cls(matrix.mean(axis=0), np.where(scale < 1e-12, 1.0, scale))

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        return (matrix - self.mean) / self.scale


@dataclass(frozen=True)
class Subset:
    """One candidate equation: which terms, their standardised weights, and its fit."""

    indices: tuple[int, ...]
    weights: np.ndarray
    #: The **penalised** objective this subset reached, ``||y - Xw||^2 + lambda*||w||^2``,
    #: not a residual sum of squares. It is only ever compared -- the beam sorts on it and
    #: the refinement pass tests it against an incumbent -- so its scale does not matter,
    #: but two things follow from the name. Ridge shrinkage can drive it slightly negative
    #: on a near-perfect fit, which is harmless for ordering and would not be for anything
    #: taking a square root of it; nothing does, and nothing should without clamping at the
    #: point of use rather than here, since clamping would flatten the ordering the beam
    #: depends on among near-perfect subsets.
    rss: float


def ridge_solve(gram: np.ndarray, rhs: np.ndarray, penalty: float) -> np.ndarray:
    """Solve ``(G + penalty*I) w = rhs`` for a centred, standardised design.

    Centring both sides removes the intercept from the system, so no penalty is ever
    applied to it -- shrinking an intercept toward zero would bias every prediction
    toward MCC = 0, which is not a value this data goes anywhere near.

    This is the **reference form**, written to be read. `Selector._evaluate` does not call
    it: it adds the penalty to the full Gram diagonal once at construction and indexes the
    submatrix straight out of that, which is the same arithmetic without a copy and an
    index array per solve. ``test_fit`` asserts the two agree, so this function is what
    the fast path is checked against rather than documentation of something else.
    """
    system = gram.copy()
    if penalty:
        system.flat[:: system.shape[0] + 1] += penalty
    try:
        return np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, rhs, rcond=None)[0]


def transform_gap(values: np.ndarray, target: np.ndarray) -> float:
    """How much more monotone than linear a feature's relation to the target is.

    Near zero means a plain linear term is adequate. Clearly positive means a transform
    (log, inverse, power) or a ratio is likely to pay, because the association is there
    but a straight line cannot capture it.
    """
    return abs(spearman(values, target)) - abs(pearson(values, target))


def guided_screen(
    library: Library,
    target: np.ndarray,
    *,
    keep: int,
    linear_gap: float = 0.05,
) -> list[int]:
    """Reduce the library to the terms worth searching over.

    Terms are ranked by the stronger of their two correlations, so a term that is
    strongly monotone but weakly linear survives -- it is exactly the term a linear
    equation needs a transform for. Terms that are near-duplicates of a stronger term
    already kept are dropped, since the beam would otherwise spend its width on
    variations of one idea.
    """
    matrix = library.matrix
    # Ranking is the expensive half of a Spearman correlation -- it sorts, then averages
    # ties -- and screening a 600-term pool inside 20 folds asks for it tens of thousands
    # of times. Nothing here is per-candidate: the target's ranks do not change from one
    # candidate to the next, the candidates' own ranks are one vectorised pass over the
    # matrix (`stats.rank_columns`), and both correlations are one matrix-vector product
    # over the whole pool (`stats.pearson_columns`) rather than a `pearson` call per column.
    ranked_target = rankdata(target)
    ranked = rank_columns(matrix) if matrix.shape[0] > 1 else matrix
    linear = np.abs(pearson_columns(matrix, target))
    monotone = (
        np.abs(pearson_columns(ranked, ranked_target)) if matrix.shape[0] > 1 else np.zeros(matrix.shape[1])
    )
    # A term that is only monotone is still useful, but it is preferred a little
    # less than an equally strong linear one: linear terms read more simply.
    strength = np.maximum(linear, monotone) - np.where(monotone - linear > linear_gap, 0.02, 0.0)
    # Descending by strength, ties broken by ascending library index, for the reason
    # `_descending` sets out: the survivors of the duplicate pass below must be a property
    # of the library rather than of the order a caller assembled it in.
    scored = _descending(strength)

    standardized = Standardizer.fit(matrix).apply(matrix)
    rows = float(standardized.shape[0])

    # The duplicate check is one matrix-vector product against the columns kept so far,
    # not a Python loop of dot products over them. At 1599 candidates keeping 1200 the
    # loop form ran 17 million times and accounted for over half the total runtime of a
    # cross-validated sweep; this is the same arithmetic in one BLAS call per candidate.
    accepted = np.empty((standardized.shape[0], min(keep, standardized.shape[1])))
    kept: list[int] = []
    for position in scored:
        if len(kept) >= keep:
            break
        index = int(position)
        column = standardized[:, index]
        if kept:
            correlations = np.abs(column @ accepted[:, : len(kept)]) / rows
            if float(correlations.max()) > 0.995:
                continue
        accepted[:, len(kept)] = column
        kept.append(index)
    return kept


def _descending(scores: np.ndarray) -> np.ndarray:
    """Indices of ``scores`` from largest to smallest, ties broken by ascending index.

    ``np.argsort(scores)[::-1]`` looks equivalent and is not, twice over: the default sort is
    quicksort, which is unstable, and reversing even a stable ascending sort *inverts* the tie
    order rather than preserving it. Either way ties were resolved by where a term happened to
    land in the library, and `terms.build_library` emits terms in feature declaration order --
    so handing `run_equation` the same columns in a different order produced a different
    equation, and leave-one-dataset-out moved by as much as 0.286 across seven orderings of a
    single ten-feature set.

    Sorting the negated scores with an explicitly stable kind breaks ties by ascending library
    index instead, which is a property of the library rather than of the caller's argument
    order.
    """
    return np.argsort(-scores, kind="stable")


class Selector:
    """Beam search over term subsets against a fixed, standardised design."""

    def __init__(
        self,
        design: np.ndarray,
        target: np.ndarray,
        penalty: float,
        groups: np.ndarray | None = None,
        term_features: list[frozenset[str]] | None = None,
    ) -> None:
        #: The raw features each library term names, for `beam.determinantal_select`. Only a
        #: diversity-weighted policy reads it, so it stays optional: every other caller would
        #: be paying to build a list it never looks at.
        self.term_features = term_features
        self.design = design
        self.centered = target - target.mean()
        self.offset = float(target.mean())
        self.penalty = penalty
        self.gram = design.T @ design
        # The ridge penalty added to the *full* Gram diagonal, once. A submatrix's
        # diagonal is drawn from the full diagonal, so ``penalized[ix, ix]`` already
        # equals ``gram[ix, ix] + penalty*I`` -- which removes a copy, an arange and a
        # fancy-index assignment from every one of the million solves a study runs.
        # ``gram`` itself stays unpenalised: the residual scoring and the collinearity
        # guard both need it that way.
        self.penalized = self.gram.copy()
        if penalty:
            self.penalized.flat[:: self.penalized.shape[0] + 1] += penalty
        self.projection = design.T @ self.centered
        self.total = float(self.centered @ self.centered)
        self.normalizer = float(design.shape[0])
        # The collinearity test as a boolean matrix, taken once. ``_blocked`` runs it for
        # every parent of every beam step -- tens of thousands of times per fit -- and the
        # arithmetic does not depend on the parent, only the columns selected out of it.
        # Deciding it here turns each call into a bool gather over an eighth of the bytes.
        self._collinear = np.abs(self.gram) / self.normalizer > COLLINEARITY_LIMIT
        self._cache: dict[tuple[int, ...], Subset] = {}
        # `terms.Library.feature_groups`, expanded once into a membership matrix so that
        # `_blocked` is a row gather and an `any`, not an `np.isin` over the pool on each
        # of several hundred thousand calls. Row ``g`` marks every term in group ``g``.
        self.groups = groups
        self._members: np.ndarray | None = None
        if groups is not None and groups.size:
            count = int(groups.max()) + 1
            if count > 0:
                members = np.zeros((count, groups.shape[0]), dtype=bool)
                present = groups >= 0
                members[groups[present], np.flatnonzero(present)] = True
                self._members = members

    def _evaluate(self, indices: tuple[int, ...]) -> Subset:
        """Fit one subset. Memoised, because the refinement pass revisits subsets.

        The residual sum of squares uses an identity rather than a second quadratic form.
        Since ``(G + lambda I) w = b``, we have ``w'Gw = w'b - lambda w'w``, so

            rss = total - 2 w'b + w'Gw = total - w'b - lambda w'w

        which removes a matrix-vector product from the inner loop. The two are equal to
        floating-point tolerance whenever the solve succeeded, which ``test_fit`` checks
        directly against the literal definition.
        """
        cached = self._cache.get(indices)
        if cached is not None:
            return cached

        order = np.array(indices, dtype=np.intp)
        # ``penalized[order[:, None], order]`` rather than ``np.ix_(order, order)``.
        # Identical arithmetic: ix_ exists to build exactly this pair of broadcast index
        # arrays, and it spends most of its time on dtype checks already known to hold
        # here. At a million calls that check cost 24 of the study's 137 seconds, and
        # doing the reshape directly is 1.8x faster.
        block = self.penalized[order[:, None], order]
        rhs = self.projection[order]
        try:
            weights = np.linalg.solve(block, rhs)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(block, rhs, rcond=None)[0]
        rss = self.total - float(weights @ rhs) - self.penalty * float(weights @ weights)
        subset = Subset(indices, weights, rss)
        self._cache[indices] = subset
        return subset

    def _evaluate_many(self, batch: list[tuple[int, ...]]) -> list[Subset]:
        """Fit several subsets **of the same size** in one stacked solve.

        A beam step and a refinement position both produce a few dozen candidate subsets
        that differ by one index, and every one of them is a ``k``-by-``k`` solve. At
        that size numpy's per-call wrapper -- dtype promotion, array coercion, the
        errstate context manager -- costs several times the LAPACK call it guards, and
        the study makes over a million of them. Stacking the blocks into one ``(n, k, k)``
        array moves that loop into C: same routine per slice, bitwise identical weights,
        two to ten times faster depending on ``k``.

        Order is preserved and the cache is shared with `_evaluate`, so callers see
        exactly what the one-at-a-time path gave them.
        """
        pending: list[tuple[int, ...]] = []
        queued: set[tuple[int, ...]] = set()
        for indices in batch:
            if indices not in self._cache and indices not in queued:
                queued.add(indices)
                pending.append(indices)

        if pending:
            order = np.array(pending, dtype=np.intp)
            blocks = self.penalized[order[:, :, None], order[:, None, :]]
            rhs = self.projection[order]
            try:
                weights = np.linalg.solve(blocks, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                # One singular block fails the whole stack, so fall back to the
                # per-subset path, which has its own least-squares rescue.
                for indices in pending:
                    self._evaluate(indices)
            else:
                scores = self.total - np.einsum("ij,ij->i", weights, rhs)
                if self.penalty:
                    scores -= self.penalty * np.einsum("ij,ij->i", weights, weights)
                for indices, row, rss in zip(pending, weights, scores, strict=True):
                    self._cache[indices] = Subset(indices, row, float(rss))

        return [self._cache[indices] for indices in batch]

    def _residual_scores(self, subset: Subset) -> np.ndarray:
        order = np.array(subset.indices, dtype=np.intp)
        return np.abs(self.projection - self.gram[:, order] @ subset.weights)

    def _blocked(self, indices: tuple[int, ...]) -> np.ndarray | None:
        """Which candidates this parent may not take, as one boolean mask.

        Computed per parent rather than per candidate. The per-candidate form rebuilt an
        index array and took a max on every one of several hundred thousand calls; this
        does the same work as a single vectorised reduction over the Gram matrix.

        Two independent refusals, and they answer different questions. The correlation
        test asks whether a candidate is numerically redundant against what is already
        chosen. The group test asks whether it would state an *already-stated relationship
        a second time* -- see `terms.Library.feature_groups`. Neither implies the other:
        the mirrored pairs the group test removes sat at 0.891 and 0.786, comfortably
        inside ``COLLINEARITY_LIMIT``, and two terms may correlate above the limit while
        describing different feature pairs entirely.
        """
        if not indices:
            return None
        order = np.array(indices, dtype=np.intp)
        blocked = self._collinear[:, order].any(axis=1)
        if self._members is None or self.groups is None:
            return blocked
        chosen = self.groups[order]
        chosen = chosen[chosen >= 0]
        if chosen.size == 0:
            return blocked
        # No `np.unique` on the way in: a repeated group id gathers the same membership row
        # twice and an `any` over it reduces to the same mask, so the sort it costs bought
        # nothing across several hundred thousand calls.
        return blocked | self._members[chosen].any(axis=0)

    def search(
        self,
        pool: list[int],
        max_terms: int,
        *,
        beam_width: int = BEAM_WIDTH_DEFAULT,
        candidates: int = CANDIDATE_POOL_DEFAULT,
        refine_rounds: int = REFINE_ROUNDS_DEFAULT,
        policy: BeamPolicy = VANILLA,
        seeds: list[int] | None = None,
    ) -> dict[int, Subset]:
        """Return the best subset found at every size from 1 to ``max_terms``.

        ``policy`` selects how the beam is pruned and kept diverse; the default is the
        textbook beam and the one every published number comes from. `ml_meta_perf.beam`
        documents the variants and why each might pay on a library as redundant as this one.

        ``seeds`` are library indices to guarantee a place in the *first* step's beam, which
        is where a redundant library does its damage: the six strongest single terms are often
        six spellings of one idea. `ml_meta_perf.seeding` chooses them.
        """
        available = np.array(pool)
        beam: list[Subset] = [Subset((), np.zeros(0), self.total)]
        best: dict[int, Subset] = {}
        vanilla = policy.is_vanilla()

        for size in range(1, max_terms + 1):
            seen: set[tuple[int, ...]] = set()
            children: list[tuple[int, ...]] = []
            origins: list[int] = []
            for position, parent in enumerate(beam):
                scores = self._residual_scores(parent)
                ranked = available[_descending(scores[available])]
                blocked = self._blocked(parent.indices)
                taken = 0
                for candidate in ranked:
                    if taken >= candidates:
                        break
                    index = int(candidate)
                    if index in parent.indices or (blocked is not None and blocked[index]):
                        continue
                    child = tuple(sorted((*parent.indices, index)))
                    if child in seen:
                        continue
                    seen.add(child)
                    children.append(child)
                    origins.append(position)
                    taken += 1
            # Every child of this step has the same size, which is what lets them go
            # through the solver as one stack rather than one at a time.
            generated = self._evaluate_many(children)
            if not generated:
                break
            if vanilla:
                generated.sort(key=lambda item: item.rss)
                beam = generated[:beam_width]
            else:
                beam = self._select_beam(generated, origins, beam_width, policy)
            if size == 1 and seeds:
                beam = self._with_seeds(beam, seeds, beam_width)
            best[size] = self._refine(beam[0], available, refine_rounds, candidates)
            if best[size].rss < beam[0].rss:
                beam[0] = best[size]
        return best

    def _select_beam(
        self,
        generated: list[Subset],
        origins: list[int],
        beam_width: int,
        policy: BeamPolicy,
    ) -> list[Subset]:
        """Apply one non-vanilla policy's pruning, cap and diversity to a step's children.

        Score order first, always: every rule here decides who to *skip*, never how to rank,
        so a policy can narrow or diversify the beam but cannot promote a worse subset above a
        better one except through the explicitly weighted diversity term.
        """
        order = _descending(-np.array([subset.rss for subset in generated]))
        ranked = [generated[int(position)] for position in order]
        ranked_origins = [origins[int(position)] for position in order]

        scores = np.array([subset.rss for subset in ranked])
        kept = beam.prune(scores, policy, self.total)
        ranked = [subset for subset, keep in zip(ranked, kept, strict=True) if keep]
        ranked_origins = [origin for origin, keep in zip(ranked_origins, kept, strict=True) if keep]

        if policy.diversity_weight > 0.0:
            features = [self._feature_names(subset) for subset in ranked]
            chosen = beam.determinantal_select(
                np.array([subset.rss for subset in ranked]), features, beam_width, policy.diversity_weight
            )
            ranked = [ranked[position] for position in chosen]
            ranked_origins = [ranked_origins[position] for position in chosen]

        positions = beam.cap_per_parent(ranked_origins, beam_width, policy.per_parent_cap)
        return [ranked[position] for position in positions] or ranked[:beam_width]

    def _feature_names(self, subset: Subset) -> frozenset[str]:
        """Which raw features a subset talks about, for the diversity kernel.

        Feature sets rather than term indices: two subsets built from different spellings of
        the same relationship are *not* diverse in the sense this study cares about, and an
        index-based kernel would call them so.
        """
        if self.term_features is None:
            return frozenset()
        names: set[str] = set()
        for index in subset.indices:
            names |= self.term_features[index]
        return frozenset(names)

    def _with_seeds(self, beam: list[Subset], seeds: list[int], beam_width: int) -> list[Subset]:
        """Guarantee the seeded single terms a place in the first step's beam.

        The best-scoring child keeps position zero -- it is what `_refine` improves and what
        `best[1]` reports, so displacing it would change the published length-1 equation for
        reasons that have nothing to do with diversity. Seeds fill the slots behind it.
        """
        head = beam[:1]
        present = {subset.indices for subset in head}
        seeded: list[Subset] = []
        for index in seeds:
            key = (index,)
            if key in present:
                continue
            present.add(key)
            seeded.append(self._evaluate(key))
        remainder = [subset for subset in beam[1:] if subset.indices not in present]
        return (head + seeded + remainder)[:beam_width]

    def _refine(
        self,
        subset: Subset,
        available: np.ndarray,
        rounds: int = REFINE_ROUNDS_DEFAULT,
        candidates: int = CANDIDATE_POOL_DEFAULT,
    ) -> Subset:
        """Local search: try replacing each chosen term with a better one.

        Beam search still fixes early choices under a narrow width. Swapping one term at
        a time recovers the cases where a term that looked good at step two is dominated
        once the rest of the equation is in place.
        """
        current = subset
        for _ in range(rounds):
            improved = False
            warmed: tuple[int, ...] | None = None
            for position in range(len(current.indices)):
                # The probe for every position of one subset is a leave-one-out of it, so all
                # of them are the same size and price in a single stacked solve. Priced up
                # front and read out of the cache position by position; `_evaluate_many` shares
                # the cache with `_evaluate`, so this only changes when the solves happen.
                if warmed != current.indices:
                    warmed = current.indices
                    if len(warmed) > 1:
                        self._evaluate_many(
                            [tuple(x for i, x in enumerate(warmed) if i != p) for p in range(len(warmed))]
                        )
                remaining = tuple(x for i, x in enumerate(current.indices) if i != position)
                probe = self._evaluate(remaining) if remaining else Subset((), np.zeros(0), self.total)
                scores = self._residual_scores(probe) if remaining else np.abs(self.projection)
                ranked = available[_descending(scores[available])][:candidates]
                blocked = self._blocked(remaining)
                trials = [
                    tuple(sorted((*remaining, int(candidate))))
                    for candidate in ranked
                    if int(candidate) not in remaining and (blocked is None or not blocked[int(candidate)])
                ]
                # Scanned in ranked order for the *first* improvement, exactly as the
                # one-at-a-time loop did. Evaluating the rest of the batch is not waste:
                # a refinement round that improves nothing has to price every candidate
                # anyway, and anything priced here is cached for the rounds after it.
                for trial in self._evaluate_many(trials):
                    if trial.rss < current.rss - 1e-12:
                        current = trial
                        improved = True
                        break
            if not improved:
                break
        return current


def to_equation(
    library: Library,
    subset: Subset,
    standardizer: Standardizer,
    offset: float,
    name: str,
) -> Equation:
    """Fold the standardisation back into the weights so the equation reads in raw units."""
    order = list(subset.indices)
    raw = subset.weights / standardizer.scale[order]
    intercept = offset - float(raw @ standardizer.mean[order])
    return Equation(
        intercept=intercept,
        terms=tuple(library.terms[index] for index in order),
        weights=tuple(float(value) for value in raw),
        standardized_weights=tuple(float(value) for value in subset.weights),
        name=name,
    )


@dataclass(frozen=True)
class FitResult:
    """Everything one fit produces: the equation per size, and the path that found it."""

    equations: dict[int, Equation]
    pool_size: int

    def best(self, max_terms: int | None = None) -> Equation:
        sizes = [size for size in self.equations if max_terms is None or size <= max_terms]
        return self.equations[max(sizes)]


def fit(
    library: Library,
    target: np.ndarray,
    *,
    max_terms: int = 12,
    penalty: float = RIDGE_DEFAULT,
    pool_size: int = 250,
    beam_width: int = BEAM_WIDTH_DEFAULT,
    candidates: int = CANDIDATE_POOL_DEFAULT,
    refine_rounds: int = REFINE_ROUNDS_DEFAULT,
    name: str = "equation",
    policy: BeamPolicy = VANILLA,
) -> FitResult:
    """Fit equations of every size up to ``max_terms`` over the given library.

    ``policy`` is the beam's pruning and diversity configuration; the default is the textbook
    beam every published number comes from. See `ml_meta_perf.beam`.
    """
    standardizer = Standardizer.fit(library.matrix)
    design = standardizer.apply(library.matrix)
    pool = guided_screen(library, target, keep=pool_size)
    features = [frozenset(term.features) for term in library.terms] if policy.diversity_weight > 0.0 else None
    selector = Selector(design, target, penalty, library.feature_groups, features)
    seeds = _seed_terms(design, library, target, pool, policy) if policy.seed_terms else None
    subsets = selector.search(
        pool,
        max_terms,
        beam_width=beam_width,
        candidates=candidates,
        refine_rounds=refine_rounds,
        policy=policy,
        seeds=seeds,
    )
    equations = {
        size: to_equation(library, subset, standardizer, selector.offset, f"{name}_k{size}")
        for size, subset in subsets.items()
    }
    return FitResult(equations=equations, pool_size=len(pool))


def _seed_terms(
    design: np.ndarray,
    library: Library,
    target: np.ndarray,
    pool: list[int],
    policy: BeamPolicy,
) -> list[int]:
    """Space-filling starting terms for the beam, from `ml_meta_perf.seeding`.

    The screening strength is recomputed rather than threaded out of `guided_screen`, which
    keeps that function's signature the one every other caller wants. It is one vectorised
    pass over the pool's columns and is not on any hot path -- seeding happens once per fit,
    against a beam step that happens tens of thousands of times.
    """
    from ml_meta_perf.seeding import seed_positions

    columns = library.matrix[:, pool]
    strength = np.abs(pearson_columns(columns, target))
    return seed_positions(design, pool, strength, policy.seed_terms)


#: A term whose contribution never moves predicted MCC by this much across the data is
#: not doing work worth printing. The default is deliberately well below the resolution
#: anyone reads MCC at.
MIN_CONTRIBUTION = 0.002


def prune(
    equation: Equation,
    columns: dict[str, np.ndarray],
    target: np.ndarray,
    *,
    penalty: float = RIDGE_DEFAULT,
    min_contribution: float = MIN_CONTRIBUTION,
) -> Equation:
    """Drop terms that do no work, simplify what remains, and refit the weights.

    Two things make a published equation longer than it needs to be.

    A term can survive selection and then contribute nothing: beam search adds terms while
    residual sum of squares falls, and the last few can fall by amounts invisible in the
    output. One dendrogram-cut equation carried a term weighted -3.7e-15 -- a full
    expression over six features, contributing zero.

    A term can also be algebraically redundant, computing ``a`` while printing
    ``([a] / [b]) * [b]``. ``terms.simplify`` handles that, and terms that reduce to a
    constant are removed outright since they duplicate the intercept.

    Weights are refitted after pruning rather than carried over. Dropping a term changes
    what the others should be, and keeping stale weights would leave an equation that is
    shorter but no longer the best fit of its own remaining terms.
    """
    if not equation.terms:
        return equation

    contributions = np.column_stack(
        [weight * term.evaluate(columns) for term, weight in zip(equation.terms, equation.weights, strict=True)]
    )
    keep: list[Term] = []
    seen: set[str] = set()
    for index, term in enumerate(equation.terms):
        column = contributions[:, index]
        low, high = np.percentile(column, [1.0, 99.0])
        if float(high - low) < min_contribution:
            continue
        reduced = simplify(term)
        if is_trivial(reduced) or reduced.name in seen:
            continue
        seen.add(reduced.name)
        keep.append(reduced)

    if not keep:
        return Equation(
            intercept=float(target.mean()), terms=(), weights=(), standardized_weights=(), name=equation.name
        )

    library = Library(keep, columns)
    standardizer = Standardizer.fit(library.matrix)
    selector = Selector(standardizer.apply(library.matrix), target, penalty, library.feature_groups)
    subset = selector._evaluate(tuple(range(len(library))))
    return to_equation(library, subset, standardizer, selector.offset, equation.name)


def selected_terms(equation: Equation) -> list[Term]:
    """The terms of an equation, strongest standardised contribution first."""
    return [term for term, _, _ in equation.ranked_terms()]
