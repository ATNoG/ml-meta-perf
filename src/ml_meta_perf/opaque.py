"""Opaque regressors on the same rows, scored under the same protocols.

The study's central claim is a trade: accuracy given up for an equation that can be read.
A trade needs both sides priced, and the expensive side is what a flexible model reaches on
this meta-data. That number is quoted throughout the chapters and was, for a long time,
measured once by hand and reproducible by nobody -- which for a study whose whole argument is
that its analysis is generated rather than narrated was the wrong way round. This module
computes it on every run.

Three regressors, chosen because they are what a reader would reach for rather than because
they are the strongest available:

* **Ridge** with a cross-validated penalty -- the linear-but-opaque control. It sees the same
  eighteen raw columns the grammar is built over, unstandardised structure and all.
* **Random forest** -- the model that produces the R2 ~ 0.9 figures reported for opaque
  meta-learners, and the one whose transfer collapse is the point.
* **Gradient boosting** -- the same family with a different bias, so the forest result cannot
  be read as an artefact of one estimator.

**Every one is scored under the study's own protocols**, by the same
`ml_meta_perf.validate.leave_one_group_out` splits, with the same clip to the training fold's
observed range that every reported number uses. A comparison is only a comparison if both
sides are scored the same way, and this went wrong twice in this project's history in ways
that flattered the equation.

Fixed seeds, and no tuning. Tuning the opaque side would strengthen it and change nothing
about the conclusion -- the failure is that 20 dataset groups let a flexible model memorise
group identity, which more trees cannot fix -- but an untuned opponent should be described as
one, so `n_estimators` and depth are stated in the table rather than buried.

Study chapter: [5. Evaluation](../../assets/docs/05-evaluation.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV

from ml_meta_perf.data import (
    ALL_FEATURES,
    DATASET_COLUMN,
    MODEL_COLUMN,
    columns_as_arrays,
    groups,
    target,
)
from ml_meta_perf.model import MCC_LOWER, MCC_UPPER
from ml_meta_perf.stats import mae, r2_score
from ml_meta_perf.validate import leave_one_group_out

#: Seed for every estimator that takes one. Fixed so the comparison is reproducible; the
#: conclusion does not depend on it: the gap being measured is two orders of magnitude
#: larger than the spread a different seed produces.
SEED = 0

#: The regressors, as (label, factory). Factories rather than instances so each fold gets a
#: fresh unfitted estimator -- refitting a fitted scikit-learn estimator is fine, but sharing
#: one across folds is the kind of thing that silently leaks if an estimator ever caches.
ESTIMATORS: tuple[tuple[str, str], ...] = (
    ("RidgeCV (linear)", "ridge"),
    ("RandomForest (300 trees)", "forest"),
    ("GradientBoosting (100 stages)", "boosting"),
)


def _build(kind: str) -> RidgeCV | RandomForestRegressor | GradientBoostingRegressor:
    if kind == "ridge":
        # A wide log-spaced grid rather than a chosen penalty: the point of the control is
        # that a *well-regularised* linear model on raw columns still transfers badly, so
        # picking the penalty by hand would leave the result open to that objection.
        return RidgeCV(alphas=np.logspace(-3, 3, 25))
    if kind == "forest":
        return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    return GradientBoostingRegressor(n_estimators=100, random_state=SEED)


def _design(frame: pl.DataFrame) -> np.ndarray:
    """The eighteen raw meta-feature columns, in `ALL_FEATURES` order.

    Raw and untransformed on purpose. The grammar's transforms are part of what this study
    contributes, so handing them to the opaque side would be comparing the equation against
    itself; handing the opaque side nothing but the columns is the comparison a reader means
    by "would a forest have done better".
    """
    columns = columns_as_arrays(frame, ALL_FEATURES)
    return np.column_stack([columns[name] for name in ALL_FEATURES])


def _cross_validate(design: np.ndarray, truth: np.ndarray, labels: np.ndarray, kind: str) -> np.ndarray:
    """Out-of-fold predictions under one leave-one-group-out split.

    Clipped to the training fold's own observed range, exactly as
    `validate._clip_to_training` does for every reported number. Without it the ridge control
    is scored on an unbounded extrapolation the equation is never scored on, which would make
    the comparison unfair in the equation's favour.
    """
    predictions = np.zeros_like(truth)
    for _, train, test in leave_one_group_out(labels):
        estimator = _build(kind)
        estimator.fit(design[train], truth[train])
        held = np.asarray(estimator.predict(design[test]), dtype=np.float64)
        predictions[test] = np.clip(held, float(truth[train].min()), float(truth[train].max()))
    return predictions


@dataclass(frozen=True)
class OpaqueRun:
    """One pass over the opaque regressors: the score table, and the predictions behind it.

    Both together because the folds are the expensive part -- a random forest refits 45 times
    across the two protocols -- and the ranking and threshold comparisons need the same
    predictions the regression table was scored from. Computing them twice would double the
    cost and, worse, leave open the possibility of the two disagreeing.
    """

    table: pl.DataFrame
    #: ``predictions[label][protocol]``, for every estimator in `ESTIMATORS` and every
    #: protocol in the table. Clipped exactly as the reported numbers are.
    predictions: dict[str, dict[str, np.ndarray]]


def evaluate(frame: pl.DataFrame) -> OpaqueRun:
    """Fit and score every opaque regressor once, keeping the predictions.

    See `opaque_baselines` for what the table says; this is the same work with the
    intermediate predictions retained so the decision tasks can reuse them.
    """
    design = _design(frame)
    truth = target(frame)
    splits = {"loo_dataset": groups(frame, DATASET_COLUMN), "loo_model": groups(frame, MODEL_COLUMN)}

    rows: list[dict[str, object]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for label, kind in ESTIMATORS:
        fitted = _build(kind)
        fitted.fit(design, truth)
        held: dict[str, np.ndarray] = {
            "in_sample": np.clip(np.asarray(fitted.predict(design), dtype=np.float64), MCC_LOWER, MCC_UPPER)
        }
        for name, labels in splits.items():
            held[name] = _cross_validate(design, truth, labels, kind)
        predictions[label] = held
        rows.append(
            {
                "model": label,
                "features": len(ALL_FEATURES),
                **{
                    key: value
                    for name, values in held.items()
                    for key, value in ((f"r2_{name}", r2_score(truth, values)), (f"mae_{name}", mae(truth, values)))
                },
            }
        )
    return OpaqueRun(table=pl.DataFrame(rows), predictions=predictions)


def opaque_baselines(frame: pl.DataFrame) -> pl.DataFrame:
    """Each opaque regressor in-sample and under both leave-one-group-out protocols.

    **Read the in-sample and leave-one-dataset-out columns of the forest row together.** That
    pair is the study's argument in two numbers: a flexible model fits this meta-data almost
    perfectly and generalises to an unseen dataset worse than a fifteen-term additive
    equation. With twenty dataset groups and features constant within a group, a forest can
    identify the dataset and look its answer up, and identification is worth nothing on a
    dataset nobody has run.

    It is also the likely provenance of the R2 ~ 0.9 figures reported for opaque meta-models
    elsewhere: an in-sample or randomly-split forest reproduces them exactly.
    """
    return evaluate(frame).table
