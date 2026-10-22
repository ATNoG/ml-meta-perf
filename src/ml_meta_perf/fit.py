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

*Gram-matrix arithmetic.* Every refit is a $k \\times k$ solve against a precomputed Gram matrix
$G = Z^\\top Z$ rather than a least-squares call against the full design, which is what makes an
exhaustive sweep inside 20 cross-validation folds tractable:

$$(G + \\lambda I)\\,w = b, \\qquad G = Z^\\top Z, \\quad b = Z^\\top (y - \\bar{y}).$$

`ridge_solve` states that arithmetic in one place; `search.Selector` indexes submatrices out of a
Gram it holds, for speed, and `tests/test_search.py` checks the two agree.

Study chapter: [3. Term generation and selection][study-chapter] -- the rationale, in prose.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/03-term-selection.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_meta_perf.config import Bounds
from ml_meta_perf.model import MCC_LOWER, MCC_UPPER, Equation
from ml_meta_perf.terms import Library

#: The ridge penalty a caller gets if it does not choose one.
RIDGE_DEFAULT = 10.0


@dataclass(frozen=True)
class Standardizer:
    """Column means $\\mu_j$ and scales $\\sigma_j$, learned on training rows only."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> Standardizer:
        """Estimate column means $\\mu_j$ and population scales $\\sigma_j$ (clamped to $1$ below $10^{-12}$)."""
        scale = matrix.std(axis=0)
        return cls(matrix.mean(axis=0), np.where(scale < 1e-12, 1.0, scale))

    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """Standardise ``matrix`` column-wise: $Z_{\\cdot,j} = (X_{\\cdot,j} - \\mu_j) / \\sigma_j$."""
        return (matrix - self.mean) / self.scale


@dataclass(frozen=True)
class Subset:
    """One candidate equation: which terms, their standardised weights, and its fit."""

    indices: tuple[int, ...]
    weights: np.ndarray
    #: The **penalised** objective this subset reached,
    #: $J(w) = \\|(y - \\bar{y}) - Z w\\|_2^2 + \\lambda \\|w\\|_2^2$
    #: $= \\|y - \\bar{y}\\|_2^2 - w^\\top b - \\lambda w^\\top w$,
    #: not an unpenalised residual sum of squares. It is only ever compared -- the beam sorts on it and
    #: the refinement pass tests it against an incumbent -- so its scale does not matter,
    #: but two things follow from the name. Ridge shrinkage can drive it slightly negative
    #: on a near-perfect fit, which is harmless for ordering and would not be for anything
    #: taking a square root of it; nothing does, and nothing should without clamping at the
    #: point of use rather than here, since clamping would flatten the ordering the beam
    #: depends on among near-perfect subsets.
    rss: float


def ridge_solve(gram: np.ndarray, rhs: np.ndarray, penalty: float) -> np.ndarray:
    """Solve $(G + \\lambda I)\\,w = b$ for a standardised design ($G = Z^\\top Z$, $b = Z^\\top (y - \\bar{y})$).

    Centring both sides removes the intercept from the system, so no penalty is ever
    applied to it -- shrinking an intercept toward zero would bias every prediction
    toward $0$ rather than the training mean $\\bar{y}$.

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
    *,
    bounds: Bounds | None = (MCC_LOWER, MCC_UPPER),
    target: str = "MCC",
) -> Equation:
    """Fold the standardisation $(\\mu_i, \\sigma_i)$ back into the weights so the equation reads in raw units:

    $$w_i^{\\text{raw}} = \\frac{w_i}{\\sigma_i}, \\qquad
    w_0^{\\text{raw}} = \\bar{y} - \\sum_{i=1}^{k} w_i^{\\text{raw}} \\mu_i.$$
    """
    order = list(subset.indices)
    raw = subset.weights / standardizer.scale[order]
    intercept = offset - float(raw @ standardizer.mean[order])
    return Equation(
        intercept=intercept,
        terms=tuple(library.terms[index] for index in order),
        weights=tuple(float(value) for value in raw),
        standardized_weights=tuple(float(value) for value in subset.weights),
        name=name,
        bounds=bounds,
        target=target,
    )
