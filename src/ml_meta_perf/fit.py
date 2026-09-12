"""Fitting weights to a form that has already been chosen.

The study's central distinction, in a module boundary. An equation is two things -- which
terms it contains and what they are multiplied by -- and they are decided by different means
and validated by different protocols. `ml_meta_perf.search` chooses the terms;
this module does the arithmetic that turns a chosen set of them into an equation, and nothing
here has an opinion about which set that should be.

That line is not a tidiness preference. `validate.cross_validate_fixed_form` is the study's
reported protocol precisely because it holds the form still and refits only the weights per
fold, and until 2026-09-09 the function that performed a search was called `fit`, which
blurred the one distinction the results rest on. Chapter 3 calls the other half "term
selection" and always has.

*Gram-matrix arithmetic.* Every refit is a k-by-k solve against a precomputed Gram matrix
rather than a least-squares call against the full design, which is what makes an exhaustive
sweep inside 20 cross-validation folds tractable. `ridge_solve` states that arithmetic in one
place; `search.Selector` indexes submatrices out of a Gram it holds, for speed, and
`tests/test_search.py` checks the two agree.

Study chapter: [3. Term generation and selection][study-chapter] -- the rationale, in
prose, with the figures.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/03-term-selection.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_meta_perf.model import Equation
from ml_meta_perf.terms import Library

#: The ridge penalty a caller gets if it does not choose one.
RIDGE_DEFAULT = 10.0


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
