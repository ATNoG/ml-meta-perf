"""Run a study: fit the equations its file lists, under the protocols it names, and score them.

A study file (``config/study.json`` or ``config/flexfl/*.json``) says which corpus, which target,
which features, which protocols and which equations; nothing about the problem is in the code. The
equations are E1 (the dataset features alone), E2 (the model features alone), E3-Valid (both: the
smaller, explainable equation) and E3-MAX (both, at the capability arity, at the raw maximum of the
floor). Each is fitted by `experiment.run_equation` or `experiment.run_capability` under the same
configuration, so the gaps between them are evidence about the features and not about the tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf.attribution import group_shares, term_effects
from ml_meta_perf.config import E3_MAX, Configuration, SelectionConfig, StudyConfig, load_config
from ml_meta_perf.data import Schema, columns_as_arrays, drop_constant_features, from_fit_scale, load, target
from ml_meta_perf.experiment import PROTOCOL_NAMES, EquationReport, run_capability, run_equation
from ml_meta_perf.validate import score

#: The name a published equation goes by in a table: E3 in the study file is E3-Valid in the outputs.
VARIANT_NAMES = {"E1": "E1", "E2": "E2", "E3": "E3-Valid", E3_MAX: E3_MAX}


@dataclass(frozen=True)
class StudyReport:
    """What a study produced: its equations, what the terms of E3-Valid are worth, and every score."""

    schema: Schema
    #: The fitted equations by the name the study file gave them (``E1``, ``E2``, ``E3``, ``E3-MAX``).
    equations: dict[str, EquationReport]
    #: Per-term effects and the share of each feature group for E3-Valid; empty when E3 is not fitted.
    effects: pl.DataFrame = field(default_factory=pl.DataFrame)
    shares: pl.DataFrame = field(default_factory=pl.DataFrame)
    #: Every equation's scores under every protocol, on the fitted scale and on the target's own.
    scores: pl.DataFrame = field(default_factory=pl.DataFrame)
    #: Counts and ranges read off the loaded corpus: rows, datasets, models, the target's minimum and maximum.
    facts: dict[str, float | int] = field(default_factory=dict)

    @property
    def valid(self) -> EquationReport:
        """E3-Valid, the equation a study publishes: every study fits it."""
        return self.equations["E3"]

    @property
    def maximum(self) -> EquationReport | None:
        """E3-MAX, the best score the configuration reaches, when the study asks for it."""
        return self.equations.get(E3_MAX)


def original_scale_scores(report: EquationReport, frame: pl.DataFrame, schema: Schema, variant: str) -> pl.DataFrame:
    """One row per protocol and scale: the equation's scores on the fitted scale and on the target's own.

    An equation fitted on ``log(y)`` predicts ``log(y)``; its R2 there is a statement about ratios, and
    the question a user asks is usually about ``y`` itself. Each protocol's held-out predictions are
    mapped back with `data.from_fit_scale` and scored against the raw target. For an untransformed
    target the two scales coincide. ``exp`` of a mean on the log scale is the geometric mean, so the
    back-transformed predictions of a skewed cost sit below its arithmetic mean (chapter 7, 7.5).
    """
    truth_fit = target(frame, schema)
    truth = frame[schema.target_column].to_numpy().astype(np.float64)
    columns = columns_as_arrays(frame, schema.features)
    predictions = {"IS": report.equation.predict(columns)} | {
        PROTOCOL_NAMES.get(label, label): path[report.n_terms].predictions for label, path in report.paths.items()
    }
    rows: list[dict[str, object]] = []
    for protocol, fitted in predictions.items():
        for scale, observed, predicted in (
            ("fit", truth_fit, fitted),
            ("original", truth, from_fit_scale(fitted, schema)),
        ):
            scores = score(observed, predicted)
            rows.append(
                {
                    "variant": variant,
                    "protocol": protocol,
                    "scale": scale,
                    "r2": scores.r2,
                    "mae": scores.mae,
                    "smape": scores.smape,
                }
            )
    return pl.DataFrame(rows)


def run_schema(
    path: str | Path | None,
    schema: Schema,
    config: Configuration | None = None,
    selection: SelectionConfig | None = None,
    equations: tuple[str, ...] = ("E3",),
) -> StudyReport:
    """Load the corpus ``schema`` describes and fit ``equations`` on it.

    ``config`` and ``selection`` default to ``config/study.json``. **One configuration serves every
    equation**, so an override changes the study consistently.
    """
    if config is None or selection is None:
        shipped = load_config()
        config, selection = config or shipped.search, selection or shipped.selection
    frame = load(path, schema)
    if schema.drop_constant_features:
        schema = drop_constant_features(frame, schema)
    fitted: dict[str, EquationReport] = {}
    for name in equations:
        if name == E3_MAX:
            fitted[name] = run_capability(frame, config, selection, schema=schema)
            continue
        dataset = schema.dataset_features if name in ("E1", "E3") else ()
        model = schema.model_features if name in ("E2", "E3") else ()
        fitted[name] = run_equation(frame, dataset, model, config, selection, schema.tag(name), schema=schema)
    effects = shares = pl.DataFrame()
    if "E3" in fitted:
        columns = columns_as_arrays(frame, schema.features)
        effects = term_effects(fitted["E3"].equation, columns, schema.dataset_features, schema.model_features)
        shares = group_shares(fitted["E3"].equation, columns, schema.dataset_features, schema.model_features)
    scores = pl.concat(
        [original_scale_scores(report, frame, schema, VARIANT_NAMES[name]) for name, report in fitted.items()]
    )
    facts: dict[str, float | int] = {
        "rows": frame.height,
        "datasets": int(frame[schema.dataset_column].n_unique()),
        "target_min": float(frame[schema.target_column].min()),  # pyright: ignore[reportArgumentType]
        "target_max": float(frame[schema.target_column].max()),  # pyright: ignore[reportArgumentType]
    }
    if schema.model_column is not None:
        facts["models"] = int(frame[schema.model_column].n_unique())
    return StudyReport(schema, fitted, effects, shares, scores, facts)


def run_study(study: StudyConfig, path: str | Path | None = None) -> StudyReport:
    """Fit the equations a study file lists on the corpus it names (or on ``path``)."""
    return run_schema(path, study.schema(), study.search, study.selection, study.equations)
