"""The end-to-end experiment: fit E1, E2 and E3, validate all three, and report.

The three equations differ only in which features they may draw on -- E1 sees dataset
meta-features, E2 sees model meta-features, E3 sees both -- so the gaps between them
measure what each half of the meta-data is worth.

The defaults below are not arbitrary. They are the configuration that survived a sweep
over the term stability cap, the ridge penalty and the equation length, scored on
leave-one-dataset-out rather than on fit. The sweep is reproducible through
``sweep_configurations``; ``DEFAULT_E1`` and ``DEFAULT_E3`` are simply where it landed.

Study chapter: [6. Results](../../assets/docs/06-results.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ml_meta_perf.analysis import screen
from ml_meta_perf.attribution import group_shares, term_effects, variance_decomposition
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
from ml_meta_perf.fit import fit, prune
from ml_meta_perf.model import Equation
from ml_meta_perf.practices import best_practices
from ml_meta_perf.selection import pareto_table, recommend
from ml_meta_perf.terms import Library, build_library
from ml_meta_perf.validate import (
    CrossValidation,
    Scores,
    additive_oracle,
    baseline_group_mean,
    cross_validate_fixed_form,
    decision_report,
    fold_selections,
    interaction_capture,
    oracle_ladder,
    random_kfold_groups,
    ranking_report,
    score,
    term_stability,
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
    max_arity: int = 3


# Dataset features only, fitted on all 476 rows like the other two. An earlier version
# fitted it on the 20 aggregated per-dataset means, on the grounds that a predictor which
# is constant inside a group can only ever predict that group's mean anyway. That was
# true and it was still the wrong choice: it put E1's R2 on a 20-point denominator, so
# its headline could not be compared with E3's without a paragraph of explanation, and
# the 0.506 it produced read as *better* transfer than E3's 0.466 when on the common
# scale it is 0.217. Fitting all three the same way costs 0.03 of in-sample R2 and
# removes the caveat entirely. See [chapter 6](../../assets/docs/06-results.md).
DEFAULT_E1 = Configuration(max_abs_zscore=3.0, penalty=20.0, pool_size=200, max_terms=8, headline_terms=7)

# Model features only. Re-swept over penalty x length x z-cap x arity on the fixed-form
# protocol, after `MODEL_FEATURES` was replaced and the reported protocol changed on
# 2026-09-05.
#
# Length falls from 8 to 6 on 2026-09-07, when `terms.Library.feature_groups` made one term
# per feature combination a rule of the protocol. E2 draws on six features, so it has only
# fifteen pairs to spend a term on and the constraint binds hardest here. Six is the knee by
# the same rule that chose eight before it -- every longer equation gains under 0.005 on
# either transfer protocol -- and six is now also *better* than eight on both of them
# (0.185 against 0.183 leave-one-dataset-out, 0.229 against 0.226 leave-one-model-out) for
# two fewer terms.
DEFAULT_E2 = Configuration(max_abs_zscore=3.0, penalty=5.0, pool_size=100, max_terms=12, headline_terms=6, max_arity=2)

# Re-swept over penalty x length x z-cap x arity after `MODEL_FEATURES` was replaced and the
# reported protocol changed to fixed form, both on 2026-09-05. All four knobs moved.
#
# `penalty` falls from 20 to 15, and would fall further on fit alone. Under the previous
# protocol the ridge did two jobs -- shrinking the weights *and* scoring which subset the beam
# chose, since the penalty sits in the selection score. With the form fixed it only does the
# first, so heavy shrinkage stopped paying for itself. Fifteen rather than the sweep's optimum
# of one or three: across that whole range the difference is under 0.01 on either transfer
# protocol, and an essentially unregularised ridge on a 476-row design is not worth that.
#
# `max_arity` drops from 3 to 2. The third arity buys `(f1+f2)/f3` and it is not selected --
# the best arity-2 point matches the best arity-3 point to within 0.007. A smaller grammar
# that scores the same is not a trade.
#
# `max_abs_zscore` rises from 3.0 to 4.25. The cap exists to stop a term being carried by a
# handful of extreme rows, and it is safe to loosen here in a way it was not before: all six
# model features are positive, bounded and fully supported, with none of the low-support tail
# that made a loose cap dangerous when hyperparameter columns were in the pool.
#
# Length stays at 16 under the one-term-per-feature-combination protocol, and the reason is
# brevity rather than accuracy, because on accuracy the two candidates cannot be separated.
#
# Twenty terms is the maximum of the transfer curve: 0.638 against 0.627 leave-one-dataset-out
# and 0.677 against 0.665 in-sample. It is also better *within* a dataset, which is the half
# of the variance a ranking sees -- 0.468 against 0.460 once each dataset's mean is removed
# from both sides. Nothing about a longer equation is hurting the fit.
#
# The 20-term equation appears to rank worse -- average precision 0.778 against 0.819, hit@1
# 0.700 against 0.800 -- and that appearance does not survive a paired test. It is two
# datasets of twenty flipping their top-1 pick, ASNM-CDX-2009 and KPI-KQI; on twenty folds
# hit@1 moves only in steps of 0.05, so 0.80 to 0.70 *is* those two. Per dataset, twenty terms
# has the better average precision on nine of the seventeen that changed at all -- sign test
# p = 1.000, bootstrap interval [-0.115, +0.023]. Do not read the ranking column as a trend in
# length; it is a mean over twenty groups and a single group moves it by 0.03.
#
# So neither R2 nor ranking decides this, and what is left is the study's standing tie-break:
# at indistinguishable measured performance, the shorter equation wins.
# `equation_search.OBJECTIVE_WEIGHTS` agrees -- 0.689 at sixteen against 0.679 at twenty --
# though that margin is drawn from the same twenty folds and should not be read as decisive
# on its own either.
#
# Penalty, z-cap and arity are unchanged from the 2026-09-05 sweep. A full
# penalty x length x z-cap x arity sweep under the constraint has not been run, and
# `ml-meta-perf-search` is where it belongs -- see `TODO.md`.
DEFAULT_E3 = Configuration(
    max_abs_zscore=4.25, penalty=15.0, pool_size=600, max_terms=32, headline_terms=16, max_arity=2
)

SWEEP_SIZES: tuple[int, ...] = (2, 4, 8, 12, 16, 20, 24, 26, 28, 32)


@dataclass
class EquationReport:
    """One equation together with everything said about it."""

    equation: Equation
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

    In-sample gets the same three metrics as the cross-validated columns, not just R2:
    an error curve that omits the in-sample line cannot show how far apart fit and
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
            MODEL_FEATURES,
            columns,
            max_arity=config.max_arity,
            max_abs_zscore=config.max_abs_zscore,
        )
    result = fit(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
    )
    path = cross_validate_fixed_form(library, columns, truth, labels, result.equations, penalty=config.penalty)
    return path, library


def run_equation(
    frame: pl.DataFrame,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    config: Configuration,
    name: str,
    sizes: tuple[int, ...] | None = None,
) -> EquationReport:
    """Fit one equation and validate it. The three equations differ **only** in the
    features they may draw on, and this is the single code path that says so.

    Every one of them is fitted on all 476 rows and scored on all 476 rows, under both
    leave-one-group-out protocols. That uniformity is the point: the gaps between E1, E2
    and E3 are only evidence about what each half of the meta-data is worth if nothing
    else differs between them -- not the fitting scale, not the protocol, not the
    denominator of the R2.
    """
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
    result = fit(
        library,
        truth,
        max_terms=config.max_terms,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
        name=name,
    )
    available = max(result.equations)
    size = min(config.headline_terms, available)
    # Fixed form: the terms are chosen once, here, and only the weights are refit in each
    # fold. See `validate.cross_validate_fixed_form` for why that is the reported protocol.
    paths = {
        label: cross_validate_fixed_form(library, columns, truth, labels, result.equations, penalty=config.penalty)
        for label, labels in (("loo_dataset", datasets), ("loo_model", models))
    }
    # The same folds with selection re-run inside them, kept only for `stability`: the share
    # of the equation's terms that survive when a fifth of the data is removed. That is what
    # licenses fixing the form, and it is not part of any reported score.
    reselected = fold_selections(
        library,
        truth,
        datasets,
        n_terms=size,
        penalty=config.penalty,
        pool_size=config.pool_size,
        beam_width=config.beam_width,
    )
    in_sample = {k: score(truth, eq.predict(columns)) for k, eq in result.equations.items()}
    equation = prune(result.equations[size], columns, truth, penalty=config.penalty)

    return EquationReport(
        equation=equation,
        in_sample=score(truth, equation.predict(columns)).as_dict(),
        curve=_curve(sizes or tuple(sorted(result.equations)), in_sample, paths, truth),
        cross_validated={
            label: path[size].scores(truth).as_dict() | path[size].dispersion() for label, path in paths.items()
        },
        stability=term_stability(reselected),
        paths=paths,
    )


def run_e1(frame: pl.DataFrame, config: Configuration = DEFAULT_E1) -> EquationReport:
    """Dataset features only -- how much of MCC the data alone explains.

    Every model on a given dataset shares one feature vector, so this equation can only
    ever predict a per-dataset constant. That is not a flaw to be corrected, it is the
    control: whatever E3 reaches beyond this is what knowing the model buys.

    Least squares finds that per-dataset constant on its own, so the aggregation an
    earlier version performed up front was unnecessary as well as harmful to the
    comparison -- see the note on `DEFAULT_E1`.
    """
    return run_equation(frame, DATASET_FEATURES, (), config, "E1")


def run_e2(frame: pl.DataFrame, config: Configuration = DEFAULT_E2) -> EquationReport:
    """Model features only -- the mirror image of E1.

    There are six model features and five of them are constant per model, so the library
    is tiny and the equation is short by necessity rather than by choice. That is itself
    the finding: the meta-data describes datasets far better than it describes models.
    The constraint on repeated feature combinations binds hardest here for the same
    reason -- six features offer only fifteen pairs -- which is why E2 is six terms.

    Aggregating this one to 25 per-model means -- the mirror of what E1 used to do -- was
    measured and is worse. It was worse for a stronger reason before 2026-09-05, when
    three model features varied within a model; those are retired, and only
    `Processing Units Number` still does. Averaging it away still discards real variation,
    and the optimum collapses to a single term.
    """
    return run_equation(frame, (), MODEL_FEATURES, config, "E2")


def run_e3(frame: pl.DataFrame, config: Configuration = DEFAULT_E3) -> EquationReport:
    """Dataset and model features -- one input, one output, and the published equation."""
    return run_equation(frame, DATASET_FEATURES, MODEL_FEATURES, config, "E3", SWEEP_SIZES)


def correlation_analysis(frame: pl.DataFrame, config: Configuration = DEFAULT_E3, top: int = 15) -> pl.DataFrame:
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


def leakage_demonstration(
    frame: pl.DataFrame,
    config: Configuration = DEFAULT_E3,
    known: dict[str, dict[int, CrossValidation]] | None = None,
) -> pl.DataFrame:
    """The same equation scored under a random split and under a grouped split.

    Dataset features are constant across a dataset's rows, so a random k-fold split puts
    the same dataset on both sides of the fold. The equation then recognises the dataset
    rather than generalising to it. This table exists so the difference is visible in
    numbers rather than asserted in prose.

    ``known`` supplies grouped paths that `run_e3` has already computed. They are the same
    folds over the same library at the same settings, and beam search records its best
    subset at every size as it goes, so a path built to 32 terms contains the 24-term
    entry this table wants -- recomputing it produced identical numbers and was a third of
    the study's runtime.
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
        if path is None or config.headline_terms not in path:
            path, library = _fixed_form_path(columns, truth, labels, config, library)
        rows.append({"protocol": label, **path[config.headline_terms].scores(truth).as_dict()})
    return pl.DataFrame(rows)


def interaction_reached(frame: pl.DataFrame, e3: EquationReport) -> pl.DataFrame:
    """How much of the leading interaction pattern the fitted equation actually reaches.

    The oracle ladder prices interaction components; this says whether E3 gets any of
    them. Reported under both the fit and the held-out protocol, because an equation can
    align with a pattern in-sample and lose it out of fold -- and that difference is the
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
) -> pl.DataFrame:
    """E1 against E3 on the one scale where they are comparable: all rows.

    All three equations are already fitted and scored on the same 476 rows, so this table
    is not correcting a denominator -- an earlier version of E1 was fitted on 20 aggregated
    dataset means and it was. What it adds is the two *ceilings* beside the equations: the
    true per-dataset and per-model means, which are the most any equation restricted to
    that half of the meta-data could reach. Without them a reader compares E1's 0.349 with
    E3's 0.665 and concludes the dataset side is weak, when 0.349 against a ceiling of
    0.354 means E1 is finished.
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
            {"equation": "E1 (dataset only)", **score(truth, e1.equation.predict(columns)).as_dict()},
            {
                "equation": "E1 ceiling (true dataset means)",
                **score(truth, dataset_ceiling).as_dict(),
            },
            {
                "equation": "E2 (model only)",
                **score(truth, e2.equation.predict(columns)).as_dict(),
            },
            {
                "equation": "E2 ceiling (true model means)",
                **score(truth, model_ceiling).as_dict(),
            },
            {"equation": "E3 (dataset + model)", **score(truth, e3.equation.predict(columns)).as_dict()},
            {
                "equation": "additive oracle (ceiling)",
                **score(truth, additive_oracle(truth, datasets, models)).as_dict(),
            },
        ]
    )


def decision_quality(
    frame: pl.DataFrame,
    config: Configuration = DEFAULT_E3,
    known: dict[int, CrossValidation] | None = None,
) -> pl.DataFrame:
    """Go/no-go decision quality, scored on held-out datasets rather than in-sample.

    ``known`` is `run_e3`'s leave-one-dataset-out path, reused for the same reason as in
    `leakage_demonstration`.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    path = known
    if path is None or config.headline_terms not in path:
        path, _ = _fixed_form_path(columns, truth, groups(frame, DATASET_COLUMN), config)
    return decision_report(truth, path[config.headline_terms].predictions, groups(frame, DATASET_COLUMN))


def model_selection(frame: pl.DataFrame, e3: EquationReport) -> pl.DataFrame:
    """Can the equation pick a good model for a dataset it has never seen?"""
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    return ranking_report(truth, e3.equation.predict(columns), datasets)


@dataclass
class Report:
    """Everything the experiment produces."""

    e1: EquationReport
    e3: EquationReport
    e2: EquationReport
    practices: pl.DataFrame
    effects: pl.DataFrame
    shares: pl.DataFrame
    decomposition: pl.DataFrame
    correlations: pl.DataFrame
    baselines: pl.DataFrame
    comparison: pl.DataFrame
    leakage: pl.DataFrame
    selection: pl.DataFrame
    term_choice: pl.DataFrame
    pareto: pl.DataFrame
    oracles: pl.DataFrame
    #: How far the equation's own interactions lie along the leading component the
    #: oracle ladder finds. Without it the ladder measures a ceiling and says nothing
    #: about whether the equation reaches any of it.
    interaction: pl.DataFrame
    decision: pl.DataFrame


# Small enough to run in a couple of seconds. Intended for smoke-testing the wiring,
# not for reporting: the equations it produces are far shorter than the studied ones.
QUICK_E1 = Configuration(max_abs_zscore=3.0, penalty=1.0, pool_size=40, max_terms=3, headline_terms=3)
QUICK_E3 = Configuration(max_abs_zscore=3.0, penalty=20.0, pool_size=40, max_terms=3, headline_terms=3)


def run(
    path: str | None = None,
    *,
    quick: bool = False,
    config_e1: Configuration | None = None,
    config_e3: Configuration | None = None,
) -> Report:
    """Run the whole study.

    The three configurations default to the tuned ones (or to the quick ones under
    ``quick``). Passing them explicitly is how the command line exposes the knobs: a
    caller who overrides ``config_e3`` gets a study that is internally consistent, since
    every table that mentions E3 is computed from the same configuration object.
    """
    frame = load(path)
    config_e1 = config_e1 or (QUICK_E1 if quick else DEFAULT_E1)
    config_e3 = config_e3 or (QUICK_E3 if quick else DEFAULT_E3)
    e1 = run_e1(frame, config_e1)
    e3 = run_e3(frame, config_e3)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    e2 = run_e2(frame)
    return Report(
        e1=e1,
        e2=e2,
        e3=e3,
        practices=best_practices(e3.equation, columns, e3.stability),
        effects=term_effects(e3.equation, columns, DATASET_FEATURES, MODEL_FEATURES),
        shares=group_shares(e3.equation, columns, DATASET_FEATURES, MODEL_FEATURES),
        decomposition=variance_decomposition(target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)),
        correlations=correlation_analysis(frame, config_e3),
        baselines=baselines(frame),
        comparison=comparison(frame, e1, e3, e2),
        leakage=leakage_demonstration(frame, config_e3, e3.paths),
        selection=model_selection(frame, e3),
        decision=decision_quality(frame, config_e3, e3.paths.get("loo_dataset")),
        term_choice=recommend(e3.curve),
        pareto=pareto_table(e3.curve),
        oracles=oracle_ladder(target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)),
        interaction=interaction_reached(frame, e3),
    )
