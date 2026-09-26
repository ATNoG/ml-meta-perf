"""The end-to-end experiment: fit E1, E2 and E3, validate all three, and report.

The three equations differ only in which features they may draw on -- E1 sees dataset
meta-features, E2 sees model meta-features, E3 sees both -- so the gaps between them
measure what each half of the meta-data is worth. Everything else is shared: **one
configuration** (`config.Configuration`, read from ``config/study.json``), one search, one
set of protocols, and **one length rule**, `selection.plateau_knee`. Each equation searches
every feature it is allowed; the search, not the configuration, decides which it keeps.

E3-Valid is the E3 the study publishes. E3-MAX is the same configuration under the wider
grammar (``selection.capability_arity``) at the raw maximum of its worst-protocol curve: a
capability bound on the additive form, not an equation put forward to be read.

Study chapter: [4. The equation][study-chapter] -- the rationale, in
prose, with the figures.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/04-equation.md
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ml_meta_perf.analysis import feature_reach, grammar_ceiling, saturated_fit, screen
from ml_meta_perf.attribution import group_shares, term_effects, variance_decomposition
from ml_meta_perf.config import Configuration, OpaqueConfig, SelectionConfig, load_config
from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FEATURES,
    columns_as_arrays,
    groups,
    load,
    target,
)
from ml_meta_perf.identity import correct_out_of_fold
from ml_meta_perf.model import Equation
from ml_meta_perf.opaque import Builder as OpaqueBuilder
from ml_meta_perf.opaque import OpaqueRun
from ml_meta_perf.opaque import estimators as opaque_estimators
from ml_meta_perf.opaque import evaluate as opaque_evaluate
from ml_meta_perf.practices import best_practices, feature_practices
from ml_meta_perf.search import prune, search
from ml_meta_perf.selection import complexity, floor_argmax, floor_curve, plateau_knee, protocol_spread, smoothed
from ml_meta_perf.stats import mae, r2_score
from ml_meta_perf.terms import Library, build_library
from ml_meta_perf.validate import (
    CrossValidation,
    Scores,
    additive_mean_reference,
    baseline_group_centre,
    cross_validate_doubly_held_out,
    cross_validate_fixed_form,
    decision_report,
    fold_selections,
    interaction_capture,
    leave_one_group_out,
    library_rows,
    oracle_ladder,
    paired_comparison,
    random_kfold_groups,
    ranking_report,
    score,
    term_stability,
)

#: E1 is fitted on all 476 rows like the other two. An earlier version fitted it on the 20
#: aggregated per-dataset means, on the grounds that a predictor constant inside a group can
#: only ever predict that group's mean anyway. That was true and still the wrong choice: it put
#: E1's R2 on a 20-point denominator, so its headline could not be compared with E3's without a
#: paragraph of explanation. See [chapter 4][study-chapter].


@dataclass
class EquationReport:
    """One equation together with everything said about it."""

    equation: Equation
    #: The selected length: `selection.plateau_knee` for E1, E2 and E3-Valid, the raw floor
    #: maximum for E3-MAX. Every output reads the result from here.
    n_terms: int
    #: The grammar this equation was searched under -- the configuration's for E1, E2 and
    #: E3-Valid, ``selection.capability_arity`` for E3-MAX.
    arity: int
    in_sample: dict[str, float | int]
    curve: pl.DataFrame
    cross_validated: dict[str, dict[str, float | int]] = field(default_factory=dict)
    stability: pl.DataFrame | None = None
    #: The full cross-validated path per protocol, kept so later tables can reuse it.
    #: ``leakage_demonstration`` and ``decision_quality`` ask for the same folds at the
    #: same settings, and recomputing them was a third of the study's runtime.
    paths: dict[str, dict[int, CrossValidation]] = field(default_factory=dict)


def _curve(
    sizes: tuple[int, ...],
    in_sample: dict[int, Scores],
    paths: dict[str, dict[int, CrossValidation]],
    truth: np.ndarray,
) -> pl.DataFrame:
    """One row per equation length, carrying every metric under every protocol.

    IS gets the same three metrics as the cross-validated columns, not just R²:
    an error curve that omits the IS line cannot show how far apart fit and
    transfer run, which is the whole point of plotting them together.
    """
    rows: list[dict[str, object]] = []
    for size in sizes:
        if size not in in_sample:
            continue
        row: dict[str, object] = {
            "n_terms": size,
            "r2_in_sample": in_sample[size].r2,
            "mae_in_sample": in_sample[size].mae,
            "smape_in_sample": in_sample[size].smape,
        }
        for label, path in paths.items():
            scores = path[size].scores(truth)
            row[f"r2_{label}"] = scores.r2
            row[f"mae_{label}"] = scores.mae
            row[f"smape_{label}"] = scores.smape
        rows.append(row)
    return pl.DataFrame(rows)


def _fixed_form_path(
    columns: dict[str, np.ndarray],
    truth: np.ndarray,
    labels: np.ndarray,
    config: Configuration,
    library: Library | None = None,
    model_features: tuple[str, ...] = MODEL_FEATURES,
) -> tuple[dict[int, CrossValidation], Library]:
    """Fit once, then cross-validate the resulting forms with only their weights refit.

    The fallback used when a caller has no precomputed path to reuse. It exists so that every
    cross-validated number in this module comes from the same protocol: one equation, its
    terms fixed, its constants recalibrated per fold. Re-running selection inside the folds
    would answer a different question and is never done for a reported figure.
    """
    if library is None:
        library = build_library(
            DATASET_FEATURES,
            model_features,
            columns,
            max_arity=config.max_arity,
            max_abs_zscore=config.max_abs_zscore,
        )
    result = search(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
    )
    path = cross_validate_fixed_form(library, columns, truth, labels, result.equations, penalty=config.penalty)
    return path, library


@dataclass(frozen=True)
class FittedPath:
    """Everything `run_equation` computes before a length is chosen.

    Searching and cross-validating every length is the expensive half of an equation; choosing
    one length and finalising it is cheap. Keeping the two apart means a length rule only ever
    reads a curve that was computed once, and lets `ml_meta_perf.sweep` score hundreds of
    configurations without finalising any of them.
    """

    library: Library
    columns: dict[str, np.ndarray]
    truth: np.ndarray
    datasets: np.ndarray
    equations: dict[int, Equation]
    paths: dict[str, dict[int, CrossValidation]]
    curve: pl.DataFrame
    config: Configuration


def fit_path(
    frame: pl.DataFrame,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    config: Configuration,
    name: str,
) -> FittedPath:
    """Search one grammar and cross-validate every length on it, choosing none of them."""
    columns = columns_as_arrays(frame, dataset_features + model_features)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    library = build_library(
        dataset_features,
        model_features,
        columns,
        max_arity=config.max_arity,
        max_abs_zscore=config.max_abs_zscore,
    )
    result = search(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
        name=name,
    )
    # Fixed form: the terms are chosen once, here, and only the weights are refit in each
    # fold. See `validate.cross_validate_fixed_form` for why that is the reported protocol.
    paths = {
        label: cross_validate_fixed_form(library, columns, truth, labels, result.equations, penalty=config.penalty)
        for label, labels in (("loo_dataset", datasets), ("loo_model", models))
    }
    # DHO at **every** length, not just the selected one: every selection rule reads the
    # complete curve, and E3-MAX selects on this protocol through `selection.floor_curve`.
    paths["loo_cell"] = cross_validate_doubly_held_out(
        library, columns, truth, datasets, models, result.equations, penalty=config.penalty
    )
    in_sample = {k: score(truth, eq.predict(columns)) for k, eq in result.equations.items()}
    curve = _curve(tuple(sorted(result.equations)), in_sample, paths, truth)
    return FittedPath(library, columns, truth, datasets, result.equations, paths, curve, config)


def finalise(fitted: FittedPath, size: int) -> EquationReport:
    """Publish one length of a fitted path: prune it, and measure how stable its form is."""
    if size not in fitted.equations:
        raise ValueError(f"selected equation length {size} is unavailable")
    config, truth = fitted.config, fitted.truth
    # The same folds with selection re-run inside them, kept only for `stability`: the share
    # of the equation's terms that survive when a fifth of the data is removed. That is what
    # licenses fixing the form, and it is not part of any reported score.
    reselected = fold_selections(
        fitted.library,
        truth,
        fitted.datasets,
        n_terms=size,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
    )
    equation = prune(fitted.equations[size], fitted.columns, truth, penalty=config.penalty)

    return EquationReport(
        equation=equation,
        n_terms=size,
        arity=config.max_arity,
        in_sample=score(truth, equation.predict(fitted.columns)).as_dict(),
        curve=fitted.curve,
        cross_validated={
            label: path[size].scores(truth).as_dict() | path[size].dispersion() for label, path in fitted.paths.items()
        },
        stability=term_stability(reselected, tuple(term.name for term in equation.terms)),
        paths=fitted.paths,
    )


def _resolve(config: Configuration | None, selection: SelectionConfig | None) -> tuple[Configuration, SelectionConfig]:
    """Fill whichever of the two was not given from ``config/study.json``."""
    if config is None or selection is None:
        study = load_config()
        return config or study.search, selection or study.selection
    return config, selection


def select_length(curve: pl.DataFrame, selection: SelectionConfig) -> int:
    """The length rule every published equation shares: `selection.plateau_knee`."""
    return plateau_knee(curve, delta=selection.delta, window=selection.window, smoothing=selection.smoothing)


def run_equation(
    frame: pl.DataFrame,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    config: Configuration,
    selection: SelectionConfig,
    name: str,
) -> EquationReport:
    """Fit one equation and validate it. The three equations differ **only** in the
    features they may draw on, and this is the single code path that says so.

    Every one of them is fitted on all 476 rows and scored on all 476 rows, under all four
    protocols, and has its length chosen by the same rule. That uniformity is the point: the
    gaps between E1, E2 and E3 are only evidence about what each half of the meta-data is
    worth if nothing else differs between them -- not the fitting scale, not the protocol,
    not the denominator of the R2, not the way the length was picked.

    **The length is derived, not asserted**: `select_length` reads it off the complete
    cross-validated curve.
    """
    fitted = fit_path(frame, dataset_features, model_features, config, name)
    return finalise(fitted, select_length(fitted.curve, selection))


def run_e1(
    frame: pl.DataFrame, config: Configuration | None = None, selection: SelectionConfig | None = None
) -> EquationReport:
    """Dataset features only -- how much of MCC the data alone explains.

    Every model on a given dataset shares one feature vector, so this equation can only
    ever predict a per-dataset constant. That is not a flaw to be corrected, it is the
    control: whatever E3 reaches beyond this is what knowing the model buys.

    Least squares finds that per-dataset constant on its own, so aggregating the rows to
    twenty dataset means first would be unnecessary as well as harmful to the comparison.
    """
    config, selection = _resolve(config, selection)
    return run_equation(frame, DATASET_FEATURES, (), config, selection, "E1")


def run_e2(
    frame: pl.DataFrame, config: Configuration | None = None, selection: SelectionConfig | None = None
) -> EquationReport:
    """Model features only -- the mirror image of E1.

    There are six model features and five of them are constant per model, so the library
    is tiny and the equation is short by necessity rather than by choice. That is itself
    the finding: the meta-data describes datasets far better than it describes models.

    Aggregating this one to 25 per-model means -- the mirror of aggregating E1 -- was
    measured and is worse. `Processing Units Number` is intentionally allowed to vary with
    the dataset because it is a capacity calculation over that dataset's shape. Averaging
    it away discards real variation, and the optimum collapses to a single term.
    """
    config, selection = _resolve(config, selection)
    return run_equation(frame, (), MODEL_FEATURES, config, selection, "E2")


def run_e3(
    frame: pl.DataFrame, config: Configuration | None = None, selection: SelectionConfig | None = None
) -> EquationReport:
    """Dataset and model features: E3-Valid, the equation the study publishes.

    Every dataset feature and every model feature is in the term pool; the configuration
    restricts the search only through its hyperparameters.
    """
    config, selection = _resolve(config, selection)
    return run_equation(frame, DATASET_FEATURES, MODEL_FEATURES, config, selection, "E3")


def run_capability(
    frame: pl.DataFrame, config: Configuration | None = None, selection: SelectionConfig | None = None
) -> EquationReport:
    """E3-MAX: E3's configuration under the wider grammar, at the raw maximum of its floor.

    **A capability measurement, not a recommendation.** It answers the question E3-Valid cannot
    answer about itself -- whether the additive form is out of reach or merely out of the
    readable range -- so it takes `selection.floor_argmax` rather than the length rule, with
    no smoothing and no complexity penalty. Only the arity differs from E3-Valid.
    """
    config, selection = _resolve(config, selection)
    wider = dataclasses.replace(config, max_arity=selection.capability_arity)
    fitted = fit_path(frame, DATASET_FEATURES, MODEL_FEATURES, wider, "E3-MAX")
    return finalise(fitted, floor_argmax(fitted.curve))


def selection_summary(
    e3: EquationReport, capability: EquationReport, config: Configuration, selection: SelectionConfig
) -> pl.DataFrame:
    """E3-Valid and E3-MAX side by side: the rule each used and where it landed.

    ``smoothed_floor`` is the quantity the length rule reads; ``floor`` is the raw worst
    protocol at that length; ``at_horizon`` flags a bound that stopped at ``max_terms`` rather
    than at a maximum of the data.
    """
    rows: list[dict[str, object]] = []
    for role, report, rule in (
        ("E3-Valid", e3, "first plateau knee"),
        ("E3-MAX", capability, "raw floor maximum"),
    ):
        curve = report.curve
        position = list(curve["n_terms"]).index(report.n_terms)
        rows.append(
            {
                "role": role,
                "rule": rule,
                "arity": report.arity,
                "n_terms": report.n_terms,
                "complexity": complexity(report.arity, report.n_terms),
                "r2_in_sample": float(curve["r2_in_sample"][position]),
                "r2_loo_dataset": float(curve["r2_loo_dataset"][position]),
                "r2_loo_model": float(curve["r2_loo_model"][position]),
                "r2_loo_cell": float(curve["r2_loo_cell"][position]),
                "floor": float(floor_curve(curve)[position]),
                "smoothed_floor": float(smoothed(floor_curve(curve), selection.smoothing)[position]),
                "spread": float(protocol_spread(curve)[position]),
                "at_horizon": report.n_terms == config.max_terms,
            }
        )
    return pl.DataFrame(rows)


def reach_analysis(frame: pl.DataFrame, config: Configuration) -> tuple[pl.DataFrame, pl.DataFrame]:
    """What each raw feature is worth, and how far the grammar reaches before any search.

    Returns the per-feature table and the three-level ceiling ladder as a one-row frame.
    Both are computed from the library alone, so they are available *before* an equation
    exists and can be read as an expectation the fitted equation is then held against.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    library = build_library(
        DATASET_FEATURES,
        MODEL_FEATURES,
        columns,
        max_arity=config.max_arity,
        max_abs_zscore=config.max_abs_zscore,
    )
    truth = target(frame)
    features = DATASET_FEATURES + MODEL_FEATURES
    ladder = grammar_ceiling(library, truth, features)
    return feature_reach(library, truth, features), pl.DataFrame([ladder])


def saturated_analysis(frame: pl.DataFrame, config: Configuration) -> dict[str, float]:
    """`analysis.saturated_fit` over the library E3 actually searches.

    The library rather than the screened pool, and the E3 grammar rather than the capability
    one, because the claim it supports is about the procedure the study publishes: handing
    *these* candidates to least squares in one go is what selection is being compared with.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    library = build_library(
        DATASET_FEATURES,
        MODEL_FEATURES,
        columns,
        max_arity=config.max_arity,
        max_abs_zscore=config.max_abs_zscore,
    )
    return saturated_fit(library, target(frame), groups(frame, DATASET_COLUMN))


def correlation_analysis(frame: pl.DataFrame, config: Configuration, top: int = 15) -> pl.DataFrame:
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
        DATASET_FEATURES,
        MODEL_FEATURES,
        columns,
        max_arity=config.max_arity,
        max_abs_zscore=config.max_abs_zscore,
    )
    table = screen(library, target(frame), groups(frame, DATASET_COLUMN))
    return table.with_columns((pl.col("spearman").abs() - pl.col("pearson").abs()).alias("monotone_gap")).head(top)


def baselines(frame: pl.DataFrame) -> pl.DataFrame:
    """What the equations have to beat, and the ceiling neither of them can pass.

    Every trivial predictor is reported at both its mean and its median, because the metrics
    disagree about which is the honest opponent. R2 and RMSE are squared-error metrics and the
    mean minimises squared error; MAE is minimised by the median, and SMAPE is an
    absolute-error ratio that behaves the same way. Quoting the equation's MAE against a
    *mean* baseline compares it with a predictor that is not minimising the metric it is being
    judged on, which flatters the equation. Both rows are here so each metric can be read
    against whichever centre is hardest to beat on it.
    """
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    rows: list[dict[str, object]] = []
    for centre in ("mean", "median"):
        for label, outer, inner in (
            (f"global {centre} (loo-dataset)", datasets, None),
            (f"per-model {centre} (loo-dataset)", datasets, models),
            (f"global {centre} (loo-model)", models, None),
            (f"per-dataset {centre} (loo-model)", models, datasets),
        ):
            prediction = baseline_group_centre(truth, outer, inner, centre=centre)
            rows.append({"baseline": label, **score(truth, prediction).as_dict()})
    rows.append(
        {
            "baseline": "additive mean-based reference (in-sample)",
            **score(truth, additive_mean_reference(truth, datasets, models)).as_dict(),
        }
    )
    return pl.DataFrame(rows)


def leakage_demonstration(
    frame: pl.DataFrame,
    n_terms: int,
    config: Configuration,
    known: dict[str, dict[int, CrossValidation]] | None = None,
) -> pl.DataFrame:
    """The same equation scored under a random split and under a grouped split.

    Dataset features are constant across a dataset's rows, so a random k-fold split puts
    the same dataset on both sides of the fold. The equation then recognises the dataset
    rather than generalising to it. This table exists so the difference is visible in
    numbers rather than asserted in prose.

    ``known`` supplies grouped paths that `run_e3` has already computed. They are the same
    folds over the same library at the same settings, and beam search records its best
    subset at every size as it goes, so a path built to ``max_terms`` contains the shorter
    entry this table needs and does not have to be recomputed.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    protocols = {
        "random 10-fold (leaky)": ("random", random_kfold_groups(truth.shape[0])),
        "leave-one-dataset-out": ("loo_dataset", groups(frame, DATASET_COLUMN)),
        "leave-one-model-out": ("loo_model", groups(frame, MODEL_COLUMN)),
    }
    library: Library | None = None
    rows: list[dict[str, object]] = []
    for label, (key, labels) in protocols.items():
        path = (known or {}).get(key)
        if path is None or n_terms not in path:
            path, library = _fixed_form_path(columns, truth, labels, config, library)
        size = n_terms if n_terms in path else max(path)
        rows.append({"protocol": label, **path[size].scores(truth).as_dict()})
    return pl.DataFrame(rows)


def selection_optimism(frame: pl.DataFrame, e3: EquationReport, config: Configuration) -> pl.DataFrame:
    """LODO with the terms **re-chosen inside every fold**, beside the reported fixed-form LODO.

    The reported protocol fixes the form -- chosen once, on all rows -- and refits only the
    weights per fold (`validate.cross_validate_fixed_form`). That is a position about what the
    equation is, and it has a price: the form was chosen with the held-out dataset in view. This
    measures the price, by repeating the selection itself without the held-out dataset, at
    E3-Valid's length and configuration, and scoring those predictions.

    **A diagnostic, not a score.** Twenty folds fit twenty different equations, so the nested
    row describes the discovery procedure rather than the published equation; it is reported
    so the fixed-form transfer numbers are read as the upper estimate they are.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth, datasets = target(frame), groups(frame, DATASET_COLUMN)
    library = build_library(
        DATASET_FEATURES, MODEL_FEATURES, columns, max_arity=config.max_arity, max_abs_zscore=config.max_abs_zscore
    )
    position = {name: index for index, name in enumerate(library.names)}
    nested = np.zeros_like(truth)
    for _, train, test in leave_one_group_out(datasets):
        result = search(
            library_rows(library, train),
            truth[train],
            max_terms=e3.n_terms,
            penalty=config.penalty,
            pool_size=config.pool_size,
            beam_width=config.beam_width,
        )
        equation = result.equations[max(size for size in result.equations if size <= e3.n_terms)]
        held = library.matrix[test][:, [position[term.name] for term in equation.terms]]
        nested[test] = np.clip(
            equation.intercept + held @ np.asarray(equation.weights), truth[train].min(), truth[train].max()
        )
    fixed = e3.paths["loo_dataset"][e3.n_terms].predictions
    return pl.DataFrame(
        [
            {"LODO": "form chosen once, on all rows (reported)", **score(truth, fixed).as_dict()},
            {"LODO": "form re-chosen inside every fold", **score(truth, nested).as_dict()},
        ]
    )


def identity_ceiling(frame: pl.DataFrame, e3: EquationReport) -> pl.DataFrame:
    """The upper bound on what any model descriptor could add, measured rather than argued.

    Under LODO **every one of the 25 models appears in every training fold**,
    so the equation's residual can be averaged per model on the training rows and applied to
    the held-out dataset with no leak. That replaces the model descriptors with the best
    possible substitute -- the model's *identity*, fitted freely -- and what it adds is
    therefore a ceiling on what any descriptor set could reach by telling these classifiers
    apart. It is the number the study's central negative result is measured against: every
    rejected model-side encoding is rejected for failing to recover it.

    Two rungs, because the ceiling depends on what the identity is allowed to carry. A level
    per model shifts every one of its rows equally; a level and a slope on a dataset feature
    lets a model's advantage depend on the data, which is what a *capability* descriptor would
    have to do. `identity.correct_out_of_fold` reuses the per-fold equations the reported path
    already recorded, so the protocol is unchanged and no search is repeated.

    Generated because the chapter's hand-written copy of this table had all three rows wrong
    and listed its two rungs as identical, which no run of this function can produce.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets, models = groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)
    path = e3.paths.get("loo_dataset")
    size = len(e3.equation.terms)
    if path is None or size not in path:
        return pl.DataFrame(schema={"correction": pl.String, "r2_loo_dataset": pl.Float64, "mae": pl.Float64})
    fold = path[size]
    rungs = {
        f"none (E3-Valid, {size} terms)": fold.predictions,
        "per-model level": correct_out_of_fold(fold, columns, truth, datasets, models),
        "per-model level and slope": correct_out_of_fold(
            fold, columns, truth, datasets, models, features=DATASET_FEATURES
        ),
    }

    def per_dataset_mae(prediction: np.ndarray) -> np.ndarray:
        return np.array([mae(truth[test], prediction[test]) for _, _, test in leave_one_group_out(datasets)])

    # **Whether the gap is real is a paired question, not a difference of two pooled numbers.**
    # That distinction has produced three wrong conclusions on this project, and this gap is
    # the one the study's central negative result is measured against -- so each rung is paired
    # against the uncorrected equation over the twenty held-out datasets, on per-dataset MAE.
    baseline = per_dataset_mae(fold.predictions)
    rows: list[dict[str, object]] = []
    for label, values in rungs.items():
        row: dict[str, object] = {
            "correction": label,
            "r2_loo_dataset": r2_score(truth, values),
            "mae": mae(truth, values),
        }
        if label.startswith("none"):
            row |= {
                "gain": 0.0,
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "sign_p": float("nan"),
                "wins": 0,
                "verdict": "baseline",
            }
        else:
            paired = paired_comparison(per_dataset_mae(values), baseline, lower_is_better=True)
            row |= {
                "gain": paired.mean,
                "ci_low": paired.low,
                "ci_high": paired.high,
                "sign_p": paired.p_value,
                "wins": paired.wins,
                "verdict": "real" if paired.significant else "tie",
            }
        rows.append(row)
    return pl.DataFrame(rows)


def interaction_reached(frame: pl.DataFrame, e3: EquationReport) -> pl.DataFrame:
    """How much of the leading interaction pattern the fitted equation actually reaches.

    The oracle ladder prices interaction components; this says whether E3 gets any of
    them. Reported under both the fit and the held-out protocol, because an equation can
    align with a pattern under IS and lose it out of fold -- and that difference is the
    whole question for a component nobody can predict from meta-features.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets, models = groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)
    size = len(e3.equation.terms)
    predictions = {"in-sample": e3.equation.predict(columns)}
    path = e3.paths.get("loo_dataset")
    if path is not None and size in path:
        predictions["leave-one-dataset-out"] = path[size].predictions
    rows = [
        {"protocol": label, "rank": rank, **interaction_capture(truth, prediction, datasets, models, rank=rank)}
        for label, prediction in predictions.items()
        for rank in (1, 2)
    ]
    return pl.DataFrame(rows)


def comparison(
    frame: pl.DataFrame,
    e1: EquationReport,
    e3: EquationReport,
    e2: EquationReport | None = None,
    e3_capability: EquationReport | None = None,
) -> pl.DataFrame:
    """E1 against E3 on the one scale where they are comparable: all rows.

    All three equations are already fitted and scored on the same 476 rows, so this table
    is not correcting a denominator -- an earlier version of E1 was fitted on 20 aggregated
    dataset means and it was. What it adds is the two *ceilings* beside the equations: the
    true per-dataset and per-model means, which are the most any equation restricted to
    that half of the meta-data could reach. Without them a reader can compare the raw E1 and
    E3 scores and conclude the dataset side is weak, even when E1 is already close to its
    own structural ceiling.
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
    if e2 is None:
        e2 = run_e2(frame)

    return pl.DataFrame(
        [
            {
                "equation": f"E1, dataset only ({e1.equation.n_terms} terms)",
                "n_terms": e1.equation.n_terms,
                **score(truth, e1.equation.predict(columns)).as_dict(),
            },
            {
                "equation": "E1 reference: true dataset means",
                "n_terms": None,
                **score(truth, dataset_ceiling).as_dict(),
            },
            {
                "equation": f"E2, model only ({e2.equation.n_terms} terms)",
                "n_terms": e2.equation.n_terms,
                **score(truth, e2.equation.predict(columns)).as_dict(),
            },
            {
                "equation": "E2 reference: true model means",
                "n_terms": None,
                **score(truth, model_ceiling).as_dict(),
            },
            {
                "equation": f"E3-Valid, dataset + model ({e3.equation.n_terms} terms)",
                "n_terms": e3.equation.n_terms,
                **score(truth, e3.equation.predict(columns)).as_dict(),
            },
            *(
                [
                    {
                        "equation": (
                            f"E3-MAX, arity "
                            f"{max((len(term.features) for term in e3_capability.equation.terms), default=0)}"
                            f" ({len(e3_capability.equation.terms)} terms)"
                        ),
                        "n_terms": len(e3_capability.equation.terms),
                        **score(truth, e3_capability.equation.predict(columns)).as_dict(),
                    }
                ]
                if e3_capability is not None and e3_capability is not e3
                else []
            ),
            {
                "equation": "reference: additive mean-based reference",
                "n_terms": None,
                **score(truth, additive_mean_reference(truth, datasets, models)).as_dict(),
            },
        ]
    )


def decision_quality(
    frame: pl.DataFrame,
    n_terms: int,
    config: Configuration,
    known: dict[int, CrossValidation] | None = None,
) -> pl.DataFrame:
    """Go/no-go decision quality, with both the dataset and the model held out.

    Same requirement as `model_selection` and for the same reason: the question is asked about
    a pair nobody has run, so neither half of it may be in the training set. ``known`` is
    `run_e3`'s LODO path and is the fallback when the DHO fit
    cannot be built.
    """
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    doubly = doubly_held_out_predictions(frame, n_terms, config)
    if doubly is not None:
        return decision_report(truth, doubly, datasets)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    path = known
    if path is None or n_terms not in path:
        path, _ = _fixed_form_path(columns, truth, datasets, config)
    size = n_terms if n_terms in path else max(path)
    return decision_report(truth, path[size].predictions, datasets)


def length_comparison(frame: pl.DataFrame, e3: EquationReport, config: Configuration) -> pl.DataFrame:
    """Every equation length paired against the E3-Valid selection, per held-out dataset.

    E3-Valid is selected before this diagnostic is computed. The table measures how each
    length differs from that retained equation; it does not define another selection rule.

    Per-dataset **MAE** rather than per-dataset R2, because three datasets here have almost no
    within-dataset variance -- `5G_Slicing`'s 25 models all score about 0.986 -- and an R2
    over a near-constant target is dominated by its denominator. See
    `validate.CrossValidation.dispersion`.

    A row marked ``worse`` differs significantly from E3-Valid under this paired comparison.
    """
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    labels = np.unique(datasets)
    path = e3.paths.get("loo_dataset", {})
    if not path:
        return pl.DataFrame()
    # Referenced to what the rule chose, not to a length passed in: the table has to be able
    # to say that the rule's own pick is the right one, which it cannot do if the pick is the
    # thing being assumed.
    chosen = e3.n_terms
    if chosen not in path:
        chosen = max(path)
    if chosen not in path:
        return pl.DataFrame()
    published = chosen

    def per_fold(size: int) -> np.ndarray:
        prediction = path[size].predictions
        return np.array([mae(truth[datasets == label], prediction[datasets == label]) for label in labels])

    reference = per_fold(published)
    rows: list[dict[str, object]] = []
    for size in sorted(path):
        result = paired_comparison(reference, per_fold(size), lower_is_better=True)
        # `mean` is positive when the first argument -- the published equation -- is better.
        verdict = (
            "selected"
            if size == published
            else ("worse" if result.significant and result.mean > 0 else ("better" if result.significant else "tie"))
        )
        rows.append(
            {
                "n_terms": size,
                "r2_loo_dataset": path[size].scores(truth).r2,
                "mae_loo_dataset": float(per_fold(size).mean()),
                "mean_difference": result.mean,
                "p_value": result.p_value,
                "ci_low": result.low,
                "ci_high": result.high,
                "verdict": verdict,
            }
        )
    return pl.DataFrame(rows)


def doubly_held_out_predictions(frame: pl.DataFrame, n_terms: int, config: Configuration) -> np.ndarray | None:
    """E3's predictions with **both** the dataset and the model of each cell held out.

    Used for the ranking and the threshold decision and for nothing else. Those two are the
    questions a practitioner actually asks -- *which model should I run on this data*, and
    *will it clear my bar* -- and both are asked about a pair that has not been run. Neither
    single-group protocol answers that: LODO has seen the learner on nineteen
    other problems, LOMO has seen the dataset.

    The R2 curve and the length rule stay on the single-group protocols, which is the right
    scope for them: they are about how the equation degrades as one axis becomes unfamiliar,
    and a per-cell refit would answer a different question at 32 times the cost.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)
    library = build_library(
        DATASET_FEATURES,
        MODEL_FEATURES,
        columns,
        max_arity=config.max_arity,
        max_abs_zscore=config.max_abs_zscore,
    )
    result = search(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
    )
    if n_terms not in result.equations:
        return None
    size = n_terms
    path = cross_validate_doubly_held_out(
        library,
        columns,
        truth,
        datasets,
        models,
        {size: result.equations[size]},
        penalty=config.penalty,
    )
    return path[size].predictions if size in path else None


def _e3_predictions(frame: pl.DataFrame, e3: EquationReport, protocol: str, config: Configuration) -> np.ndarray | None:
    """E3's predictions under one protocol.

    ``in_sample``, ``loo_dataset``, ``loo_model``, or ``loo_cell`` -- the last holding out both
    the dataset and the model of every cell, which is the protocol the ranking and the
    threshold decision are reported under.
    """
    if protocol == "loo_cell":
        return doubly_held_out_predictions(frame, e3.n_terms, config)
    if protocol == "in_sample":
        columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
        return e3.equation.predict(columns)
    path = e3.paths.get(protocol)
    if not path:
        return None
    size = e3.n_terms if e3.n_terms in path else max(path)
    return path[size].predictions if size in path else None


def model_selection(
    frame: pl.DataFrame,
    e3: EquationReport,
    config: Configuration,
    protocol: str = "loo_cell",
) -> pl.DataFrame:
    """Can the equation pick a good model for a dataset it has never run it on?

    **Reported under ``loo_cell``: both the dataset and the model of every cell are out of the
    training set.** Anything weaker answers a different question. LODO has
    seen the learner on nineteen other problems; LOMO has seen the dataset. A
    recommendation is asked about a pair that has not been run, so both have to go.

    The cost of that is small, and structurally so rather than by luck: a ranking depends only
    on the *within-dataset* ordering of models, and the dataset-side terms shift every model
    on a dataset by the same amount, so holding the dataset out moves the level and not the
    order. `ranking_baselines` reports every protocol side by side so that stays visible.
    """
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    prediction = _e3_predictions(frame, e3, protocol, config)
    if prediction is None:
        return pl.DataFrame()
    return ranking_report(truth, prediction, datasets)


#: How an opaque estimator's protocol keys are labelled in the comparison tables. Every row
#: names its protocol, because a comparison is only a comparison if both sides were scored
#: under the same one -- which this project has shipped wrong twice.
OPAQUE_PROTOCOLS = {
    "in_sample": "in-sample",
    "loo_dataset": "loo-dataset",
    "loo_model": "loo-model",
    "loo_cell": "loo-cell: both held out",
}


def _opaque_candidates(opaque: OpaqueRun) -> dict[str, np.ndarray]:
    """Every opaque estimator under every protocol, keyed the way the equation's rows are."""
    return {
        f"{label} ({OPAQUE_PROTOCOLS[protocol]})": values
        for label, held in opaque.predictions.items()
        for protocol, values in held.items()
        if protocol in OPAQUE_PROTOCOLS
    }


def ranking_baselines(
    frame: pl.DataFrame,
    e3: EquationReport,
    config: Configuration,
    opaque: OpaqueRun | None = None,
) -> pl.DataFrame:
    """The equation's ranking against the trivial rankings, averaged over datasets.

    **Every predictor is scored under the same protocol**, which is what makes the comparison
    a comparison. The equation appears under IS, LODO, LOMO, and DHO, so the cost of
    generalisation on this task is visible rather than assumed. The trivial predictors appear
    under LODO, which is the only way a per-model
    centre can be computed honestly.

    Both centres are reported for the same reason the error metrics report both: the per-model
    mean and the per-model median are different orderings, and which is harder to beat is a
    question rather than an assumption.
    """
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    candidates: dict[str, np.ndarray] = {}
    for label, protocol in (
        ("equation (in-sample)", "in_sample"),
        ("equation (loo-dataset)", "loo_dataset"),
        ("equation (loo-model)", "loo_model"),
    ):
        prediction = _e3_predictions(frame, e3, protocol, config)
        if prediction is not None:
            candidates[label] = prediction
    doubly = doubly_held_out_predictions(frame, e3.n_terms, config)
    if doubly is not None:
        candidates["equation (loo-cell: both held out)"] = doubly
    # The trivial predictors cannot be computed under DHO (internally ``loo_cell``) at all: a model
    # held out of every fold has no rows to average, so "how well does this model usually do"
    # has no value. They appear under LODO, which is the only way they exist -- and
    # that hands them the model identity denied to the DHO row of the equation.
    candidates["per-model mean (loo-dataset)"] = baseline_group_centre(truth, datasets, models, centre="mean")
    candidates["per-model median (loo-dataset)"] = baseline_group_centre(truth, datasets, models, centre="median")
    # The opaque opponent under every protocol it has, so each of its rows can be read against
    # the equation row that was allowed to see the same things. The DHO rows are the
    # comparison that matters: there neither side has the dataset or the model, and the trivial
    # baselines cannot be computed at all.
    if opaque is not None:
        candidates |= _opaque_candidates(opaque)

    tables = {
        label: ranking_report(truth, prediction, datasets).sort("group") for label, prediction in candidates.items()
    }
    reference = tables["equation (loo-dataset)"] if "equation (loo-dataset)" in tables else next(iter(tables.values()))

    rows: list[dict[str, object]] = []
    for label, table in tables.items():
        row: dict[str, object] = {
            "predictor": label,
            **{
                column: float(table[column].to_numpy().mean())
                for column in ("ap", "mrr", "hit_at_1", "regret")
                if column in table.columns
            },
            "datasets": table.height,
        }
        # The mean over twenty datasets is not the comparison. hit@1 moves only in steps of
        # 0.05 on twenty folds, so a single dataset flipping its top pick shifts it by more
        # than the gaps being read. The paired test against the reported equation is what
        # licenses any claim from this table.
        if label != "equation (loo-dataset)" and "ap" in table.columns:
            paired = paired_comparison(reference["ap"].to_numpy(), table["ap"].to_numpy())
            row["ap_vs_e3_p"] = paired.p_value
            row["ap_vs_e3_significant"] = paired.significant
        rows.append(row)
    return pl.DataFrame(rows)


def decision_baselines(
    frame: pl.DataFrame,
    config: Configuration,
    known: dict[int, CrossValidation] | None = None,
    e3: EquationReport | None = None,
    opaque: OpaqueRun | None = None,
) -> pl.DataFrame:
    """The above-or-below-threshold decision, for the equation and both trivial centres.

    Every predictor under the same protocol, as in `ranking_baselines`, and the equation under
    all four so the cost of generalisation on the decision is measured rather than assumed.
    `decision_report` already carries a majority-class column, which is the floor any rule has
    to clear; these are the harder comparison.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)

    candidates: dict[str, np.ndarray] = {}
    if e3 is not None:
        for label, protocol in (
            ("equation (in-sample)", "in_sample"),
            ("equation (loo-dataset)", "loo_dataset"),
            ("equation (loo-model)", "loo_model"),
        ):
            prediction = _e3_predictions(frame, e3, protocol, config)
            if prediction is not None:
                candidates[label] = prediction
        doubly = doubly_held_out_predictions(frame, e3.n_terms, config)
        if doubly is not None:
            candidates["equation (loo-cell: both held out)"] = doubly
    else:
        path = known
        if path is None or (e3 is not None and e3.n_terms not in path):
            path, _ = _fixed_form_path(columns, truth, datasets, config)
        candidates["equation (loo-dataset)"] = path[e3.n_terms if e3 is not None else max(path)].predictions
    candidates["per-model mean (loo-dataset)"] = baseline_group_centre(truth, datasets, models, centre="mean")
    candidates["per-model median (loo-dataset)"] = baseline_group_centre(truth, datasets, models, centre="median")
    if opaque is not None:
        candidates |= _opaque_candidates(opaque)

    rows: list[dict[str, object]] = []
    for label, prediction in candidates.items():
        for row in decision_report(truth, prediction, datasets).to_dicts():
            rows.append({"predictor": label, **row})
    return pl.DataFrame(rows)


@dataclass
class Report:
    """Everything the experiment produces."""

    e1: EquationReport
    e3: EquationReport
    e2: EquationReport
    #: E3-MAX: the same configuration under the wider grammar at the raw floor maximum.
    #: Reported to show how far the additive form reaches, not as the study's
    #: recommendation. See `run_capability`.
    e3_capability: EquationReport
    #: E3-Valid and E3-MAX side by side, with the rule each used. See `selection_summary`.
    grammars: pl.DataFrame
    #: Feature-level effects and stability before evidence thresholds are applied.
    feature_practices: pl.DataFrame
    practices: pl.DataFrame
    effects: pl.DataFrame
    shares: pl.DataFrame
    decomposition: pl.DataFrame
    correlations: pl.DataFrame
    #: What each raw feature carries on its own, and the three-level ceiling the grammar
    #: implies before any search runs. See `analysis.grammar_ceiling`.
    reach: pl.DataFrame
    ceiling: pl.DataFrame
    baselines: pl.DataFrame
    comparison: pl.DataFrame
    leakage: pl.DataFrame
    selection: pl.DataFrame
    #: Every length paired against E3-Valid as a sensitivity analysis.
    length_choice: pl.DataFrame
    oracles: pl.DataFrame
    #: How far the equation's own interactions lie along the leading component the
    #: oracle ladder finds. Without it the ladder measures a ceiling and says nothing
    #: about whether the equation reaches any of it.
    interaction: pl.DataFrame
    decision: pl.DataFrame
    #: The ranking and threshold decisions against the trivial predictors, at both
    #: centres. An evaluation without them says how well the equation does and not
    #: whether it beats ordering the models by how well they usually do.
    ranking_baselines: pl.DataFrame
    decision_baselines: pl.DataFrame
    #: Standard opaque regressors on the same rows under the same protocols. The priced
    #: other side of the study's trade: see `ml_meta_perf.opaque`.
    opaque: pl.DataFrame
    #: What per-model *identity* adds to the reported path, which is the ceiling on what any
    #: model descriptor could reach. See `identity_ceiling`.
    identity: pl.DataFrame
    #: What an unpenalised least-squares fit over the *whole* library reaches under IS and
    #: held out. The control for the selection stage: if this transferred, the beam search
    #: and the length rule would be machinery in search of a problem. See
    #: `analysis.saturated_fit`.
    saturated: pl.DataFrame
    #: LODO with E3-Valid's terms re-chosen in every fold, beside the reported fixed form: how
    #: much of the transfer number belongs to choosing the form on all rows. See
    #: `selection_optimism`.
    optimism: pl.DataFrame


def run(
    path: str | None = None,
    *,
    config: Configuration | None = None,
    selection: SelectionConfig | None = None,
    opaque: OpaqueConfig | None = None,
    opaque_models: tuple[tuple[str, OpaqueBuilder], ...] | None = None,
) -> Report:
    """Run the whole study.

    ``config``, ``selection`` and ``opaque`` default to ``config/study.json``. **One
    configuration serves all three equations**: every table that mentions an equation is
    computed from the same object, so an override changes the study consistently.

    ``opaque_models`` replaces the estimators the comparison is run against, in the shape
    `opaque.evaluate` takes. The opaque side is scikit-learn's and the DHO refit around it is
    this project's, so a caller checking the wiring can exercise the second without paying
    for the first -- which at 476 refits per estimator is most of what a run costs.
    """
    frame = load(path)
    config, selection = _resolve(config, selection)
    e1 = run_e1(frame, config, selection)
    e2 = run_e2(frame, config, selection)
    e3 = run_e3(frame, config, selection)
    e3_capability = run_capability(frame, config, selection)
    config_e3 = config
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    reach, ceiling = reach_analysis(frame, config_e3)
    # Fitted once and shared: the folds are the expensive part, and the regression table and
    # the two decision comparisons have to be scored from the same predictions or they can
    # disagree with each other.
    if opaque_models is None:
        opaque_models = opaque_estimators(opaque or load_config().opaque)
    opaque_run = opaque_evaluate(frame, models=opaque_models)
    return Report(
        opaque=opaque_run.table,
        saturated=pl.DataFrame([saturated_analysis(frame, config_e3)]),
        optimism=selection_optimism(frame, e3, config_e3),
        e1=e1,
        e2=e2,
        e3=e3,
        e3_capability=e3_capability,
        grammars=selection_summary(e3, e3_capability, config, selection),
        feature_practices=feature_practices(e3.equation, columns, e3.stability),
        practices=best_practices(e3.equation, columns, e3.stability),
        effects=term_effects(e3.equation, columns, DATASET_FEATURES, MODEL_FEATURES),
        shares=group_shares(e3.equation, columns, DATASET_FEATURES, MODEL_FEATURES),
        decomposition=variance_decomposition(target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)),
        correlations=correlation_analysis(frame, config_e3),
        reach=reach,
        ceiling=ceiling,
        baselines=baselines(frame),
        comparison=comparison(frame, e1, e3, e2, e3_capability),
        leakage=leakage_demonstration(frame, e3.n_terms, config_e3, e3.paths),
        selection=model_selection(frame, e3, config_e3, "loo_cell"),
        decision=decision_quality(frame, e3.n_terms, config_e3, e3.paths.get("loo_dataset")),
        ranking_baselines=ranking_baselines(frame, e3, config_e3, opaque_run),
        decision_baselines=decision_baselines(frame, config_e3, e3.paths.get("loo_dataset"), e3, opaque_run),
        length_choice=length_comparison(frame, e3, config_e3),
        oracles=oracle_ladder(target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)),
        interaction=interaction_reached(frame, e3),
        identity=identity_ceiling(frame, e3),
    )
