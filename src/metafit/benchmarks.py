"""Reference models, to say what the equation is being traded against.

The additive oracle bounds the *form* of the equation, but it says nothing about what a
conventional learner would extract from the same features. That is a different and
equally fair question, so it gets its own answer here: fit standard regressors on the raw
meta-features and score them under the identical protocols.

scikit-learn is imported lazily and is **not** a runtime dependency of ``metafit``. The
equation pipeline needs polars and numpy only, and making every install pull scikit-learn
for a comparison table would be a poor trade. It lives in ``requirements-dev.txt``;
``available()`` reports whether it is installed and every entry point degrades to an empty
table rather than raising when it is not.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl

from metafit.data import (
    ALL_FEATURES,
    DATASET_COLUMN,
    MODEL_COLUMN,
    columns_as_arrays,
    groups,
    target,
)
from metafit.model import MCC_LOWER, MCC_UPPER
from metafit.validate import leave_one_group_out, score


def available() -> bool:
    """Whether scikit-learn can be imported."""
    try:
        import sklearn  # noqa: F401
    except ImportError:
        return False
    return True


DEFAULT_TREES = 300


def _estimators(trees: int = DEFAULT_TREES) -> dict[str, Callable[[], Any]]:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV

    return {
        "RidgeCV (linear)": lambda: RidgeCV(alphas=np.logspace(-3, 4, 40)),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=trees, random_state=0, n_jobs=-1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=0),
    }


def _design(frame: pl.DataFrame) -> np.ndarray:
    columns = columns_as_arrays(frame, ALL_FEATURES)
    return np.column_stack([columns[name] for name in ALL_FEATURES])


def _cross_validated(make: Callable[[], Any], design: np.ndarray, truth: np.ndarray, labels: np.ndarray):
    predictions = np.zeros_like(truth)
    for _, train, test in leave_one_group_out(labels):
        estimator = make()
        estimator.fit(design[train], truth[train])
        predictions[test] = np.clip(estimator.predict(design[test]), MCC_LOWER, MCC_UPPER)
    return predictions


def reference_models(
    frame: pl.DataFrame,
    *,
    only: tuple[str, ...] | None = None,
    trees: int = DEFAULT_TREES,
) -> pl.DataFrame:
    """Standard regressors on the raw meta-features, under the same three protocols.

    ``only`` restricts the estimator set by name and ``trees`` shrinks the forest. Each
    estimator is refitted once per fold across two protocols -- 46 fits -- so the full
    benchmark is by far the slowest thing in this package, and the test suite and the
    ``--quick`` run both cut it down rather than paying for it on every commit. The
    qualitative result is insensitive to tree count: memorising 476 rows does not require
    300 trees.

    In-sample numbers for the tree ensembles are near-perfect and mean nothing on their
    own -- a forest can memorise 476 rows. They are reported anyway, precisely so the
    collapse between their in-sample and cross-validated columns is visible next to the
    equation's much smaller gap. That contrast is the argument for the equation.
    """
    if not available():
        return pl.DataFrame(
            schema={"model": pl.String, "r2_in_sample": pl.Float64, "r2_loo_dataset": pl.Float64,
                    "r2_loo_model": pl.Float64, "mae_loo_dataset": pl.Float64}
        )

    design = _design(frame)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    rows: list[dict[str, object]] = []
    for name, make in _estimators(trees).items():
        if only is not None and name not in only:
            continue
        fitted = make()
        fitted.fit(design, truth)
        in_sample = score(truth, np.clip(fitted.predict(design), MCC_LOWER, MCC_UPPER))
        by_dataset = score(truth, _cross_validated(make, design, truth, datasets))
        by_model = score(truth, _cross_validated(make, design, truth, models))
        rows.append(
            {
                "model": name,
                "r2_in_sample": in_sample.r2,
                "r2_loo_dataset": by_dataset.r2,
                "r2_loo_model": by_model.r2,
                "mae_loo_dataset": by_dataset.mae,
            }
        )
    return pl.DataFrame(rows)
