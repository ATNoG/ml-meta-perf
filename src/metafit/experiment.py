"""The end-to-end experiment: fit E1 and E2, validate both, and report.

The defaults below are not arbitrary. They are the configuration that survived a sweep
over the term stability cap, the ridge penalty and the equation length, scored on
leave-one-dataset-out rather than on fit. The sweep is reproducible through
``sweep_configurations``; ``DEFAULT_E1`` and ``DEFAULT_E2`` are simply where it landed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from metafit.analysis import screen
from metafit.attribution import group_shares, term_effects, variance_decomposition
from metafit.benchmarks import reference_models
from metafit.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FEATURES,
    aggregate_by_dataset,
    columns_as_arrays,
    groups,
    load,
    target,
)
from metafit.fit import fit
from metafit.model import Equation
from metafit.practices import best_practices
from metafit.stats import r2_score
from metafit.terms import build_library
from metafit.validate import (
    CrossValidation,
    additive_oracle,
    baseline_group_mean,
    cross_validate_path,
    random_kfold_groups,
    ranking_report,
    score,
)


@dataclass(frozen=True)
class Configuration:
    """The knobs that were actually tuned, and the values that won."""

    max_abs_zscore: float
    penalty: float
    pool_size: int
    max_terms: int
    headline_terms: int
    beam_width: int = 6


DEFAULT_E1 = Configuration(max_abs_zscore=3.0, penalty=1.0, pool_size=200, max_terms=6, headline_terms=5)
DEFAULT_E2 = Configuration(max_abs_zscore=3.0, penalty=20.0, pool_size=400, max_terms=16, headline_terms=12)

SWEEP_SIZES: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16)

# The accuracy-leaning alternative: a wider library and almost no shrinkage. It reaches
# in-sample R2 = 0.619 at 16 terms and 0.647 at 20 -- comfortably past 0.6 -- and gives up
# leave-one-dataset-out R2 (0.32 against 0.37) while actually improving leave-one-model-out
# (0.49 against 0.46). Both configurations are reported; the defaults above take the
# generalising side of the trade, this one takes the fit.
ACCURATE_E2 = Configuration(max_abs_zscore=4.0, penalty=1.0, pool_size=400, max_terms=20, headline_terms=16)

# Model features only. There are five of them, so the library is tiny and the equation
# is short by necessity rather than by choice.
DEFAULT_MODEL_ONLY = Configuration(
    max_abs_zscore=3.0, penalty=20.0, pool_size=100, max_terms=9, headline_terms=9
)


@dataclass
class EquationReport:
    """One equation together with everything said about it."""

    equation: Equation
    in_sample: dict[str, float | int]
    curve: pl.DataFrame
    cross_validated: dict[str, dict[str, float | int]] = field(default_factory=dict)
    stability: pl.DataFrame | None = None


def _curve(
    sizes: tuple[int, ...],
    in_sample: dict[int, float],
    paths: dict[str, dict[int, CrossValidation]],
    truth: np.ndarray,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for size in sizes:
        if size not in in_sample:
            continue
        row: dict[str, object] = {"n_terms": size, "r2_in_sample": in_sample[size]}
        for label, path in paths.items():
            row[f"r2_{label}"] = path[size].scores(truth).r2
            row[f"mae_{label}"] = path[size].scores(truth).mae
        rows.append(row)
    return pl.DataFrame(rows)


def run_e1(frame: pl.DataFrame, config: Configuration = DEFAULT_E1) -> EquationReport:
    """Dataset features only, fitted on the per-dataset mean MCC.

    Every model on a given dataset shares one feature vector, so this equation can only
    ever predict a per-dataset constant. That is not a flaw to be corrected -- it is the
    point of the comparison. E1 measures how much of MCC is explained by the data alone.
    """
    aggregated = aggregate_by_dataset(frame)
    columns = columns_as_arrays(aggregated, DATASET_FEATURES)
    truth = target(aggregated)
    labels = groups(aggregated, DATASET_COLUMN)

    library = build_library(DATASET_FEATURES, (), columns, max_abs_zscore=config.max_abs_zscore)
    result = fit(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
        name="E1",
    )
    path = cross_validate_path(
        library,
        columns,
        truth,
        labels,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
    )
    sizes = tuple(size for size in range(1, config.max_terms + 1))
    in_sample = {size: r2_score(truth, eq.predict(columns)) for size, eq in result.equations.items()}
    equation = result.equations[config.headline_terms]

    return EquationReport(
        equation=equation,
        in_sample=score(truth, equation.predict(columns)).as_dict(),
        curve=_curve(sizes, in_sample, {"loo_dataset": path}, truth),
        cross_validated={"loo_dataset": path[config.headline_terms].scores(truth).as_dict()},
        stability=path[config.headline_terms].stability(),
    )


def run_e2(frame: pl.DataFrame, config: Configuration = DEFAULT_E2) -> EquationReport:
    """Dataset and model features, fitted on all rows -- one input, one output."""
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    library = build_library(
        DATASET_FEATURES, MODEL_FEATURES, columns, max_abs_zscore=config.max_abs_zscore
    )
    result = fit(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
        name="E2",
    )
    paths = {
        label: cross_validate_path(
            library,
            columns,
            truth,
            labels,
            max_terms=config.max_terms,
            penalty=config.penalty,
            pool_size=config.pool_size,
            beam_width=config.beam_width,
        )
        for label, labels in (("loo_dataset", datasets), ("loo_model", models))
    }
    in_sample = {size: r2_score(truth, eq.predict(columns)) for size, eq in result.equations.items()}
    equation = result.equations[config.headline_terms]

    return EquationReport(
        equation=equation,
        in_sample=score(truth, equation.predict(columns)).as_dict(),
        curve=_curve(SWEEP_SIZES, in_sample, paths, truth),
        cross_validated={
            label: path[config.headline_terms].scores(truth).as_dict() for label, path in paths.items()
        },
        stability=paths["loo_dataset"][config.headline_terms].stability(),
    )


def run_model_only(frame: pl.DataFrame, config: Configuration = DEFAULT_MODEL_ONLY) -> EquationReport:
    """Model features only -- the mirror image of E1, and the control for the claim that
    model choice dominates dataset difficulty.

    There are only five model features and one of them is constant per model, so the
    library this can draw on is tiny. That is itself the finding: the meta-data describes
    datasets far better than it describes models.
    """
    columns = columns_as_arrays(frame, MODEL_FEATURES)
    truth = target(frame)
    labels = groups(frame, DATASET_COLUMN)

    library = build_library((), MODEL_FEATURES, columns, max_abs_zscore=config.max_abs_zscore)
    result = fit(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
        name="EM",
    )
    available = max(result.equations)
    size = min(config.headline_terms, available)
    path = cross_validate_path(
        library,
        columns,
        truth,
        labels,
        max_terms=available,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
    )
    in_sample = {k: r2_score(truth, eq.predict(columns)) for k, eq in result.equations.items()}
    equation = result.equations[size]
    return EquationReport(
        equation=equation,
        in_sample=score(truth, equation.predict(columns)).as_dict(),
        curve=_curve(tuple(sorted(result.equations)), in_sample, {"loo_dataset": path}, truth),
        cross_validated={"loo_dataset": path[size].scores(truth).as_dict()},
        stability=path[size].stability(),
    )


def correlation_analysis(frame: pl.DataFrame, config: Configuration = DEFAULT_E2, top: int = 15) -> pl.DataFrame:
    """Rank candidate terms by how they relate to MCC, linearly and monotonically.

    Both correlations are reported because they say different things and the difference
    is what guides term design. A term whose Pearson and Spearman correlations agree is
    linear in MCC and needs no transform. A term whose Spearman correlation is clearly
    the larger is monotone but curved, which is the signal that a log, an inverse or a
    ratio will pay for itself. The ``within_pearson`` column is the strict one: both the
    term and the target are centred inside each dataset first, so a term is credited
    only for variation that dataset identity does not already account for.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    library = build_library(
        DATASET_FEATURES, MODEL_FEATURES, columns, max_abs_zscore=config.max_abs_zscore
    )
    table = screen(library, target(frame), groups(frame, DATASET_COLUMN))
    return table.with_columns(
        (pl.col("spearman").abs() - pl.col("pearson").abs()).alias("monotone_gap")
    ).head(top)


def baselines(frame: pl.DataFrame) -> pl.DataFrame:
    """What the equations have to beat, and the ceiling neither of them can pass."""
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    rows = [
        {
            "baseline": "global mean (loo-dataset)",
            **score(truth, baseline_group_mean(truth, datasets)).as_dict(),
        },
        {
            "baseline": "per-model mean (loo-dataset)",
            **score(truth, baseline_group_mean(truth, datasets, models)).as_dict(),
        },
        {
            "baseline": "global mean (loo-model)",
            **score(truth, baseline_group_mean(truth, models)).as_dict(),
        },
        {
            "baseline": "per-dataset mean (loo-model)",
            **score(truth, baseline_group_mean(truth, models, datasets)).as_dict(),
        },
        {
            "baseline": "additive oracle (ceiling, in-sample)",
            **score(truth, additive_oracle(truth, datasets, models)).as_dict(),
        },
    ]
    return pl.DataFrame(rows)


def leakage_demonstration(frame: pl.DataFrame, config: Configuration = DEFAULT_E2) -> pl.DataFrame:
    """The same equation scored under a random split and under a grouped split.

    Dataset features are constant across a dataset's rows, so a random k-fold split puts
    the same dataset on both sides of the fold. The equation then recognises the dataset
    rather than generalising to it. This table exists so the difference is visible in
    numbers rather than asserted in prose.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    library = build_library(
        DATASET_FEATURES, MODEL_FEATURES, columns, max_abs_zscore=config.max_abs_zscore
    )
    protocols = {
        "random 10-fold (leaky)": random_kfold_groups(truth.shape[0]),
        "leave-one-dataset-out": groups(frame, DATASET_COLUMN),
        "leave-one-model-out": groups(frame, MODEL_COLUMN),
    }
    rows: list[dict[str, object]] = []
    for label, labels in protocols.items():
        path = cross_validate_path(
            library,
            columns,
            truth,
            labels,
            max_terms=config.headline_terms,
            penalty=config.penalty,
            pool_size=config.pool_size,
            beam_width=config.beam_width,
        )
        rows.append({"protocol": label, **path[config.headline_terms].scores(truth).as_dict()})
    return pl.DataFrame(rows)


def comparison(
    frame: pl.DataFrame,
    e1: EquationReport,
    e2: EquationReport,
    model_only: EquationReport | None = None,
) -> pl.DataFrame:
    """E1 against E2 on the one scale where they are comparable: all rows.

    E1's own R2 is computed over 20 dataset means and E2's over 476 rows, so the two
    headline numbers share no denominator. Evaluating E1's equation on every row puts
    both on the same variance and makes the gap between them mean something.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    dataset_ceiling = np.zeros_like(truth)
    for label in np.unique(datasets):
        mask = datasets == label
        dataset_ceiling[mask] = truth[mask].mean()
    model_ceiling = np.zeros_like(truth)
    for label in np.unique(models):
        mask = models == label
        model_ceiling[mask] = truth[mask].mean()
    if model_only is None:
        model_only = run_model_only(frame)

    return pl.DataFrame(
        [
            {"equation": "E1 (dataset only)", **score(truth, e1.equation.predict(columns)).as_dict()},
            {
                "equation": "E1 ceiling (true dataset means)",
                **score(truth, dataset_ceiling).as_dict(),
            },
            {
                "equation": "EM (model only)",
                **score(truth, model_only.equation.predict(columns)).as_dict(),
            },
            {
                "equation": "EM ceiling (true model means)",
                **score(truth, model_ceiling).as_dict(),
            },
            {"equation": "E2 (dataset + model)", **score(truth, e2.equation.predict(columns)).as_dict()},
            {
                "equation": "additive oracle (ceiling)",
                **score(truth, additive_oracle(truth, datasets, models)).as_dict(),
            },
        ]
    )


def model_selection(frame: pl.DataFrame, e2: EquationReport) -> pl.DataFrame:
    """Can the equation pick a good model for a dataset it has never seen?"""
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    return ranking_report(truth, e2.equation.predict(columns), datasets)


@dataclass
class Report:
    """Everything the experiment produces."""

    e1: EquationReport
    e2: EquationReport
    e2_accurate: EquationReport
    model_only: EquationReport
    practices: pl.DataFrame
    effects: pl.DataFrame
    shares: pl.DataFrame
    decomposition: pl.DataFrame
    correlations: pl.DataFrame
    baselines: pl.DataFrame
    comparison: pl.DataFrame
    leakage: pl.DataFrame
    selection: pl.DataFrame
    reference: pl.DataFrame


# Small enough to run in a couple of seconds. Intended for smoke-testing the wiring,
# not for reporting: the equations it produces are far shorter than the studied ones.
QUICK_E1 = Configuration(max_abs_zscore=3.0, penalty=1.0, pool_size=40, max_terms=3, headline_terms=3)
QUICK_E2 = Configuration(max_abs_zscore=3.0, penalty=20.0, pool_size=40, max_terms=3, headline_terms=3)
QUICK_ACCURATE = Configuration(
    max_abs_zscore=4.0, penalty=1.0, pool_size=40, max_terms=4, headline_terms=4
)


def run(path: str | None = None, *, quick: bool = False) -> Report:
    """Run the whole study."""
    frame = load(path)
    config_e1 = QUICK_E1 if quick else DEFAULT_E1
    config_e2 = QUICK_E2 if quick else DEFAULT_E2
    config_accurate = QUICK_ACCURATE if quick else ACCURATE_E2
    e1 = run_e1(frame, config_e1)
    e2 = run_e2(frame, config_e2)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    return Report(
        e1=e1,
        e2=e2,
        e2_accurate=run_e2(frame, config_accurate),
        model_only=run_model_only(frame),
        practices=best_practices(e2.equation, columns, e2.stability),
        effects=term_effects(e2.equation, columns, DATASET_FEATURES, MODEL_FEATURES),
        shares=group_shares(e2.equation, columns, DATASET_FEATURES, MODEL_FEATURES),
        decomposition=variance_decomposition(
            target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)
        ),
        correlations=correlation_analysis(frame, config_e2),
        baselines=baselines(frame),
        comparison=comparison(frame, e1, e2),
        leakage=leakage_demonstration(frame, config_e2),
        selection=model_selection(frame, e2),
        reference=reference_models(frame, only=("RidgeCV (linear)",), trees=25)
        if quick
        else reference_models(frame),
    )
