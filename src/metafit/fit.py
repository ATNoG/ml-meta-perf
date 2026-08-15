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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metafit.model import Equation
from metafit.stats import pearson, spearman
from metafit.terms import Library, Term

RIDGE_DEFAULT = 10.0
BEAM_WIDTH_DEFAULT = 6
CANDIDATE_POOL_DEFAULT = 24
COLLINEARITY_LIMIT = 0.95


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
    """
    size = gram.shape[0]
    try:
        return np.linalg.solve(gram + penalty * np.eye(size), rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(gram + penalty * np.eye(size), rhs, rcond=None)[0]


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

    def _evaluate(self, indices: tuple[int, ...]) -> Subset:
        order = np.array(indices, dtype=int)
        weights = ridge_solve(self.gram[np.ix_(order, order)], self.projection[order], self.penalty)
        rss = self.total - 2.0 * float(weights @ self.projection[order]) + float(
            weights @ self.gram[np.ix_(order, order)] @ weights
        )
        return Subset(indices, weights, rss)

    def _residual_scores(self, subset: Subset) -> np.ndarray:
        order = np.array(subset.indices, dtype=int)
        return np.abs(self.projection - self.gram[:, order] @ subset.weights)

    def _too_collinear(self, candidate: int, indices: tuple[int, ...]) -> bool:
        if not indices:
            return False
        order = np.array(indices, dtype=int)
        correlations = np.abs(self.gram[candidate, order]) / self.normalizer
        return bool(np.max(correlations) > COLLINEARITY_LIMIT)

    def search(
        self,
        pool: list[int],
        max_terms: int,
        *,
        beam_width: int = BEAM_WIDTH_DEFAULT,
        candidates: int = CANDIDATE_POOL_DEFAULT,
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
                taken = 0
                for candidate in ranked:
                    if taken >= candidates:
                        break
                    index = int(candidate)
                    if index in parent.indices or self._too_collinear(index, parent.indices):
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
            best[size] = self._refine(beam[0], available)
            if best[size].rss < beam[0].rss:
                beam[0] = best[size]
        return best

    def _refine(self, subset: Subset, available: np.ndarray, rounds: int = 2) -> Subset:
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
                ranked = available[np.argsort(scores[available])[::-1]][:CANDIDATE_POOL_DEFAULT]
                for candidate in ranked:
                    index = int(candidate)
                    if index in remaining or self._too_collinear(index, remaining):
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
    name: str = "equation",
) -> FitResult:
    """Fit equations of every size up to ``max_terms`` over the given library."""
    standardizer = Standardizer.fit(library.matrix)
    design = standardizer.apply(library.matrix)
    pool = guided_screen(library, target, keep=pool_size)
    selector = Selector(design, target, penalty)
    subsets = selector.search(pool, max_terms, beam_width=beam_width)
    equations = {
        size: to_equation(library, subset, standardizer, selector.offset, f"{name}_k{size}")
        for size, subset in subsets.items()
    }
    return FitResult(equations=equations, pool_size=len(pool))


def selected_terms(equation: Equation) -> list[Term]:
    """The terms of an equation, strongest standardised contribution first."""
    return [term for term, _, _ in equation.ranked_terms()]
