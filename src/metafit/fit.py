"""Selecting which terms enter the equation, and with what weights.

Three ideas do the work here.

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

Study chapter: [3. Search and fitting](../../assets/docs/03-search-and-fitting.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metafit.model import Equation
from metafit.stats import pearson, spearman
from metafit.terms import Library, Term, is_trivial, simplify

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
    rss: float


def ridge_solve(gram: np.ndarray, rhs: np.ndarray, penalty: float) -> np.ndarray:
    """Solve ``(G + penalty*I) w = rhs`` for a centred, standardised design.

    Centring both sides removes the intercept from the system, so no penalty is ever
    applied to it -- shrinking an intercept toward zero would bias every prediction
    toward MCC = 0, which is not a value this data goes anywhere near.

    The penalty is added along the diagonal of a copy rather than by building
    ``penalty * np.eye(k)``. This runs several hundred thousand times in a full sweep and
    the identity matrix was pure allocation.
    """
    system = gram.copy()
    if penalty:
        diagonal = np.arange(system.shape[0])
        system[diagonal, diagonal] += penalty
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
    scored: list[tuple[float, int]] = []
    for index in range(matrix.shape[1]):
        column = matrix[:, index]
        linear = abs(pearson(column, target))
        monotone = abs(spearman(column, target))
        gap = monotone - linear
        # A term that is only monotone is still useful, but it is preferred a little
        # less than an equally strong linear one: linear terms read more simply.
        strength = max(linear, monotone) - (0.02 if gap > linear_gap else 0.0)
        scored.append((strength, index))
    scored.sort(key=lambda item: item[0], reverse=True)

    standardized = Standardizer.fit(matrix).apply(matrix)
    kept: list[int] = []
    for _, index in scored:
        if len(kept) >= keep:
            break
        column = standardized[:, index]
        if any(abs(float(column @ standardized[:, other]) / column.shape[0]) > 0.995 for other in kept):
            continue
        kept.append(index)
    return kept


class Selector:
    """Beam search over term subsets against a fixed, standardised design."""

    def __init__(self, design: np.ndarray, target: np.ndarray, penalty: float) -> None:
        self.design = design
        self.centered = target - target.mean()
        self.offset = float(target.mean())
        self.penalty = penalty
        self.gram = design.T @ design
        self.projection = design.T @ self.centered
        self.total = float(self.centered @ self.centered)
        self.normalizer = float(design.shape[0])
        self._cache: dict[tuple[int, ...], Subset] = {}

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

        order = np.array(indices, dtype=int)
        block = self.gram[np.ix_(order, order)]
        rhs = self.projection[order]
        weights = ridge_solve(block, rhs, self.penalty)
        rss = self.total - float(weights @ rhs) - self.penalty * float(weights @ weights)
        subset = Subset(indices, weights, rss)
        self._cache[indices] = subset
        return subset

    def _residual_scores(self, subset: Subset) -> np.ndarray:
        order = np.array(subset.indices, dtype=int)
        return np.abs(self.projection - self.gram[:, order] @ subset.weights)

    def _blocked(self, indices: tuple[int, ...]) -> np.ndarray | None:
        """Which candidates are too collinear with ``indices``, as one boolean mask.

        Computed per parent rather than per candidate. The per-candidate form rebuilt an
        index array and took a max on every one of several hundred thousand calls; this
        does the same work as a single vectorised reduction over the Gram matrix.
        """
        if not indices:
            return None
        order = np.array(indices, dtype=int)
        return (np.abs(self.gram[:, order]) / self.normalizer > COLLINEARITY_LIMIT).any(axis=1)

    def search(
        self,
        pool: list[int],
        max_terms: int,
        *,
        beam_width: int = BEAM_WIDTH_DEFAULT,
        candidates: int = CANDIDATE_POOL_DEFAULT,
        refine_rounds: int = REFINE_ROUNDS_DEFAULT,
    ) -> dict[int, Subset]:
        """Return the best subset found at every size from 1 to ``max_terms``."""
        available = np.array(pool)
        beam: list[Subset] = [Subset((), np.zeros(0), self.total)]
        best: dict[int, Subset] = {}

        for size in range(1, max_terms + 1):
            seen: set[tuple[int, ...]] = set()
            generated: list[Subset] = []
            for parent in beam:
                scores = self._residual_scores(parent)
                ranked = available[np.argsort(scores[available])[::-1]]
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
                    generated.append(self._evaluate(child))
                    taken += 1
            if not generated:
                break
            generated.sort(key=lambda item: item.rss)
            beam = generated[:beam_width]
            best[size] = self._refine(beam[0], available, refine_rounds, candidates)
            if best[size].rss < beam[0].rss:
                beam[0] = best[size]
        return best

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
            for position in range(len(current.indices)):
                remaining = tuple(x for i, x in enumerate(current.indices) if i != position)
                probe = self._evaluate(remaining) if remaining else Subset((), np.zeros(0), self.total)
                scores = self._residual_scores(probe) if remaining else np.abs(self.projection)
                ranked = available[np.argsort(scores[available])[::-1]][:candidates]
                blocked = self._blocked(remaining)
                for candidate in ranked:
                    index = int(candidate)
                    if index in remaining or (blocked is not None and blocked[index]):
                        continue
                    trial = self._evaluate(tuple(sorted((*remaining, index))))
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
) -> FitResult:
    """Fit equations of every size up to ``max_terms`` over the given library."""
    standardizer = Standardizer.fit(library.matrix)
    design = standardizer.apply(library.matrix)
    pool = guided_screen(library, target, keep=pool_size)
    selector = Selector(design, target, penalty)
    subsets = selector.search(
        pool, max_terms, beam_width=beam_width, candidates=candidates, refine_rounds=refine_rounds
    )
    equations = {
        size: to_equation(library, subset, standardizer, selector.offset, f"{name}_k{size}")
        for size, subset in subsets.items()
    }
    return FitResult(equations=equations, pool_size=len(pool))


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
    selector = Selector(standardizer.apply(library.matrix), target, penalty)
    subset = selector._evaluate(tuple(range(len(library))))
    return to_equation(library, subset, standardizer, selector.offset, equation.name)


def selected_terms(equation: Equation) -> list[Term]:
    """The terms of an equation, strongest standardised contribution first."""
    return [term for term, _, _ in equation.ranked_terms()]
