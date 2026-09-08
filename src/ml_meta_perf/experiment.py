"""The end-to-end experiment: fit E1, E2 and E3, validate all three, and report.

The three equations differ only in which features they may draw on -- E1 sees dataset
meta-features, E2 sees model meta-features, E3 sees both -- so the gaps between them
measure what each half of the meta-data is worth.

The defaults below are not arbitrary. They are the configuration that survived a sweep
over the term stability cap, the ridge penalty and the equation length, scored on
leave-one-dataset-out rather than on fit. The sweep is reproducible through
``sweep_configurations``; ``DEFAULT_E1`` and ``DEFAULT_E3`` are simply where it landed.

Study chapter: [4. The equation](../../assets/docs/04-equation.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ml_meta_perf.analysis import feature_reach, grammar_ceiling, saturated_fit, screen
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
from ml_meta_perf.identity import correct_out_of_fold
from ml_meta_perf.model import Equation
from ml_meta_perf.opaque import OpaqueRun
from ml_meta_perf.opaque import evaluate as opaque_evaluate
from ml_meta_perf.practices import best_practices
from ml_meta_perf.selection import best_length, pareto_table, recommend
from ml_meta_perf.stats import mae, r2_score
from ml_meta_perf.terms import Library, build_library
from ml_meta_perf.validate import (
    CrossValidation,
    Scores,
    additive_oracle,
    baseline_group_centre,
    cross_validate_doubly_held_out,
    cross_validate_fixed_form,
    decision_report,
    fold_selections,
    interaction_capture,
    leave_one_group_out,
    oracle_ladder,
    paired_comparison,
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
# removes the caveat entirely. See [chapter 6](../../assets/docs/04-equation.md).
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
# CONFIRMED BY THE FULL SWEEP, 2026-09-07 (Slurm job 15335, 48,576 points, 71 minutes on 62
# cores). This configuration is now what a search chose, not a length read off a curve.
#
# `max_arity=2` is confirmed outright: the best arity-2 point scores 0.7297 against 0.6866 for
# the best arity-3 point, and every configuration in the top band is arity 2.
#
# **The sweep's top row is not the answer, and this is the case the warning was written for.**
# It ranks a 9-term equation first (objective 0.7297 against this one's 0.6861). Paired over
# the twenty held-out datasets on per-dataset MAE, that equation is *significantly worse*:
# the incumbent wins on 16 of 20, sign test p = 0.012, bootstrap CI [+0.0082, +0.0303] entirely
# above zero. The standing rule is the shortest configuration that is **not significantly
# worse**, and nine terms does not qualify. Sixteen stands.
#
# What makes the objective prefer it is `stability`, not brevity. Decomposed against
# `OBJECTIVE_WEIGHTS`: stability contributes +0.0525 of the +0.0379 net gap and brevity only
# +0.0146, against -0.029 summed over the five accuracy components. The 9-term form reselects
# in 88% of folds where this one reselects in 53%. That is a real property and a real tension
# -- a shorter form is more stable and transfers worse -- but `stability` is weighted 0.15
# against 0.40 for the three R2 combined, so it should not be able to overturn an accuracy gap
# this size. Treat the objective as a shortlisting device and the paired test as the decision.
#
# The sweep also prefers a four-feature subset -- `Model Capability`, `Processing Units
# Number`, `Fitting Regime`, `Input Distribution Modelling` -- dropping `Solution
# Stochasticity` and `Loss Margin Behaviour` from the *equation*. **That is admissible, and it
# is the mechanism working rather than a loss.** The two stages have different criteria: the
# corpus is designed for *identification* and keeps all six columns, which is what makes every
# learner distinguishable on every dataset; the equation is judged on *compression*, and an
# equation that used every column would be one that had failed to generalise. See `data`'s
# module docstring. What must not change is the corpus: `MODEL_FEATURES` is the schema
# `load` validates and the set `tests/test_model_features.py` checks identification against.
#
# On the sweep's own numbers the four-feature pool dominates on every axis -- best objective
# 0.7297 against 0.7222 for six, best leave-one-dataset-out 0.6855 against 0.6716, best
# leave-one-model-out 0.6543 against 0.6404.
#
# Penalty and z-cap: no candidate significantly beats this one. The nearest, penalty 20 with
# z-cap 4.50 on the four-feature subset, reaches 0.6317 leave-one-dataset-out against 0.6270
# and is a tie when paired (p = 0.115) -- and on all six features the same knobs score 0.5798,
# so the apparent gain is the feature drop rather than the shrinkage. Left unchanged.
DEFAULT_E3 = Configuration(
    max_abs_zscore=4.25, penalty=20.0, pool_size=600, max_terms=32, headline_terms=15, max_arity=2
)

# The same corpus and the same four features under the **full** grammar, arity 3. It is the
# more accurate equation and it is reported beside the published one to show how far the
# additive form reaches when brevity is not traded for: 23 terms, and on the 2026-09-07 sweep
# the best leave-one-dataset-out of any configuration searched.
#
# Both lengths come from one rule -- `selection.best_length`, the argmax of the consensus
# curve -- applied under each grammar. Neither 15 nor 23 is written down; both fall out.
#
# What it costs is what the published equation is buying. Twenty-three terms over a grammar
# that also admits `(f1+f2)/f3` is a longer and less readable statement, and its form is far
# less stable: terms reselect in roughly a seventh of the folds against the published
# equation's third, and form stability is what licenses fixing the form at all. So this is
# reported as a *capability measurement* rather than as the study's recommendation.
DEFAULT_E3_CAPABILITY = Configuration(
    max_abs_zscore=4.25, penalty=3.0, pool_size=600, max_terms=32, headline_terms=23, max_arity=3
)

#: The model features the **equation** may build terms from.
#:
#: Deliberately a subset of `data.MODEL_FEATURES`, which is the *corpus* schema. The two
#: stages have different criteria and `data`'s module docstring sets them out: designing the
#: corpus requires **identification**, so it carries all six columns and every learner is
#: distinguishable on every dataset; fitting the equation requires **compression**, and an
#: equation that used every available column would be one that had failed to generalise.
#: Restricting the term pool removes nothing from the corpus.
#:
#: The 2026-09-07 sweep chose this subset, and it dominates the full six on every axis:
#: objective 0.7297 against 0.7222, leave-one-dataset-out 0.6855 against 0.6716,
#: leave-one-model-out 0.6543 against 0.6404. `Solution Stochasticity` and
#: `Loss Margin Behaviour` stay in the corpus and out of the equation.
EQUATION_MODEL_FEATURES: tuple[str, ...] = (
    "Model Capability",
    "Processing Units Number",
    "Fitting Regime",
    "Input Distribution Modelling",
)

#: Lengths the E3 curve is reported at.
#:
#: **Every length, not a hand-picked subset.** This was
#: ``(2, 4, 8, 12, 16, 20, 24, 26, 28, 32)`` -- non-uniform, and it skipped 13, 15, 17 and 31,
#: which are exactly the four lengths where the leave-one-dataset-out curve craters (0.532,
#: 0.393, 0.562, 0.539 against neighbours around 0.62). Whatever the intent, the published
#: curve was far smoother than the real one, and a knee detected on a non-uniform grid is
#: partly reporting the grid: the same detector returns 4 on the ragged grid and 6 on the
#: dense one.
#:
#: It costs nothing. ``fit`` already builds the whole path up to ``max_terms`` and
#: ``run_equation`` was subsampling it for E3 alone.
SWEEP_SIZES: tuple[int, ...] | None = None


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
    model_features: tuple[str, ...] = EQUATION_MODEL_FEATURES,
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


def run_e3_capability(
    frame: pl.DataFrame, config: Configuration = DEFAULT_E3_CAPABILITY
) -> EquationReport:
    """The same features under the full grammar: how far the additive form reaches.

    Not the study's recommendation and not what the chapters analyse term by term. It exists
    so that the published equation's accuracy can be read against what the *form* could do
    rather than only against oracles and baselines -- the question "is the additive model out
    of room, or is this equation short of it" needs an answer, and this is it.
    """
    return run_equation(frame, DATASET_FEATURES, EQUATION_MODEL_FEATURES, config, "E3-capability", SWEEP_SIZES)


def run_e3(frame: pl.DataFrame, config: Configuration = DEFAULT_E3) -> EquationReport:
    """Dataset and model features -- one input, one output, and the published equation."""
    return run_equation(frame, DATASET_FEATURES, EQUATION_MODEL_FEATURES, config, "E3", SWEEP_SIZES)


def reach_analysis(frame: pl.DataFrame, config: Configuration = DEFAULT_E3) -> tuple[pl.DataFrame, pl.DataFrame]:
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


def saturated_analysis(frame: pl.DataFrame, config: Configuration = DEFAULT_E3) -> dict[str, float]:
    """`analysis.saturated_fit` over the library E3 actually searches.

    The library rather than the screened pool, and the E3 grammar rather than the capability
    one, because the claim it supports is about the procedure the study publishes: handing
    *these* candidates to least squares in one go is what selection is being compared with.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
    library = build_library(
        DATASET_FEATURES,
        EQUATION_MODEL_FEATURES,
        columns,
        max_arity=config.max_arity,
        max_abs_zscore=config.max_abs_zscore,
    )
    return saturated_fit(library, target(frame), groups(frame, DATASET_COLUMN))


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
            "baseline": "additive oracle (ceiling, in-sample)",
            **score(truth, additive_oracle(truth, datasets, models)).as_dict(),
        }
    )
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


def identity_ceiling(frame: pl.DataFrame, e3: EquationReport) -> pl.DataFrame:
    """The upper bound on what any model descriptor could add, measured rather than argued.

    Under leave-one-dataset-out **every one of the 25 models appears in every training fold**,
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
        f"none (E3, {size} terms)": fold.predictions,
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
            row |= {"gain": 0.0, "ci_low": float("nan"), "ci_high": float("nan"),
                    "sign_p": float("nan"), "wins": 0, "verdict": "baseline"}
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
    e3_capability: EquationReport | None = None,
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
            {"equation": f"E1, dataset only ({e1.equation.n_terms} terms)", "n_terms": e1.equation.n_terms,
             **score(truth, e1.equation.predict(columns)).as_dict()},
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
            {"equation": f"E3, dataset + model ({e3.equation.n_terms} terms)", "n_terms": e3.equation.n_terms,
             **score(truth, e3.equation.predict(columns)).as_dict()},
            *(
                [
                    {
                        "equation": f"E3 capability, arity 3 ({len(e3_capability.equation.terms)} terms)",
                        "n_terms": len(e3_capability.equation.terms),
                        **score(truth, e3_capability.equation.predict(columns)).as_dict(),
                    }
                ]
                if e3_capability is not None
                else []
            ),
            {
                "equation": "reference: additive oracle",
                "n_terms": None,
                **score(truth, additive_oracle(truth, datasets, models)).as_dict(),
            },
        ]
    )


def decision_quality(
    frame: pl.DataFrame,
    config: Configuration = DEFAULT_E3,
    known: dict[int, CrossValidation] | None = None,
) -> pl.DataFrame:
    """Go/no-go decision quality, with both the dataset and the model held out.

    Same requirement as `model_selection` and for the same reason: the question is asked about
    a pair nobody has run, so neither half of it may be in the training set. ``known`` is
    `run_e3`'s leave-one-dataset-out path and is the fallback when the doubly-held-out fit
    cannot be built.
    """
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    doubly = doubly_held_out_predictions(frame, config)
    if doubly is not None:
        return decision_report(truth, doubly, datasets)
    columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
    path = known
    if path is None or config.headline_terms not in path:
        path, _ = _fixed_form_path(columns, truth, datasets, config)
    return decision_report(truth, path[config.headline_terms].predictions, datasets)


def length_comparison(frame: pl.DataFrame, e3: EquationReport, config: Configuration = DEFAULT_E3) -> pl.DataFrame:
    """Every equation length paired against the published one, per held-out dataset.

    **This is the rule that chooses the length**, and it replaced knee detection on
    2026-09-07. Knee detectors, gRDP-smoothed knee detectors and the Pareto-front knee all
    place the bend of this curve at four to eight terms, and every one of those lengths is
    significantly worse than the published equation when the two are compared fold by fold.
    A knee measures where the *marginal* return per term collapses; it does not measure
    whether the accuracy still being added is real. `selection` keeps the geometric readings
    as diagnostics and this is what decides.

    Per-dataset **MAE** rather than per-dataset R2, because three datasets here have almost no
    within-dataset variance -- `5G_Slicing`'s 25 models all score about 0.986 -- and an R2
    over a near-constant target is dominated by its denominator. See
    `validate.CrossValidation.dispersion`.

    The published length is the shortest whose interval against the incumbent spans zero. A
    row marked ``worse`` is one no brevity argument can justify.
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
    chosen = best_length(e3.curve)
    if chosen not in path:
        chosen = min(config.headline_terms, max(path))
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
        verdict = "selected" if size == published else ("worse" if result.significant and result.mean > 0 else
                                                       ("better" if result.significant else "tie"))
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


def doubly_held_out_predictions(
    frame: pl.DataFrame, config: Configuration = DEFAULT_E3
) -> np.ndarray | None:
    """E3's predictions with **both** the dataset and the model of each cell held out.

    Used for the ranking and the threshold decision and for nothing else. Those two are the
    questions a practitioner actually asks -- *which model should I run on this data*, and
    *will it clear my bar* -- and both are asked about a pair that has not been run. Neither
    single-group protocol answers that: leave-one-dataset-out has seen the learner on nineteen
    other problems, leave-one-model-out has seen the dataset.

    The R2 curve and the length rule stay on the single-group protocols, which is the right
    scope for them: they are about how the equation degrades as one axis becomes unfamiliar,
    and a per-cell refit would answer a different question at 32 times the cost.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)
    library = build_library(
        DATASET_FEATURES, EQUATION_MODEL_FEATURES, columns,
        max_arity=config.max_arity, max_abs_zscore=config.max_abs_zscore,
    )
    result = fit(
        library, truth, max_terms=config.max_terms, penalty=config.penalty,
        pool_size=config.pool_size, beam_width=config.beam_width,
    )
    size = min(config.headline_terms, max(result.equations))
    if size not in result.equations:
        return None
    path = cross_validate_doubly_held_out(
        library, columns, truth, datasets, models,
        {size: result.equations[size]}, penalty=config.penalty,
    )
    return path[size].predictions if size in path else None


def _e3_predictions(
    frame: pl.DataFrame, e3: EquationReport, protocol: str, config: Configuration
) -> np.ndarray | None:
    """E3's predictions under one protocol.

    ``in_sample``, ``loo_dataset``, ``loo_model``, or ``loo_cell`` -- the last holding out both
    the dataset and the model of every cell, which is the protocol the ranking and the
    threshold decision are reported under.
    """
    if protocol == "loo_cell":
        return doubly_held_out_predictions(frame, config)
    if protocol == "in_sample":
        columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
        return e3.equation.predict(columns)
    path = e3.paths.get(protocol)
    if not path:
        return None
    size = min(config.headline_terms, max(path))
    return path[size].predictions if size in path else None


def model_selection(
    frame: pl.DataFrame,
    e3: EquationReport,
    protocol: str = "loo_cell",
    config: Configuration = DEFAULT_E3,
) -> pl.DataFrame:
    """Can the equation pick a good model for a dataset it has never run it on?

    **Reported under ``loo_cell``: both the dataset and the model of every cell are out of the
    training set.** Anything weaker answers a different question. Leave-one-dataset-out has
    seen the learner on nineteen other problems; leave-one-model-out has seen the dataset. A
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
    config: Configuration = DEFAULT_E3,
    opaque: OpaqueRun | None = None,
) -> pl.DataFrame:
    """The equation's ranking against the trivial rankings, averaged over datasets.

    **Every predictor is scored under the same protocol**, which is what makes the comparison
    a comparison. The equation appears three times -- in-sample and under both leave-one-group-out
    protocols -- so the cost of generalisation on this task is visible rather than assumed, and
    the trivial predictors appear leave-one-dataset-out, which is the only way a per-model
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
    doubly = doubly_held_out_predictions(frame, config)
    if doubly is not None:
        candidates["equation (loo-cell: both held out)"] = doubly
    # The trivial predictors cannot be computed under the loo-cell protocol at all: a model
    # held out of every fold has no rows to average, so "how well does this model usually do"
    # has no value. They appear leave-one-dataset-out, which is the only way they exist -- and
    # that hands them the model identity the loo-cell row of the equation is denied.
    candidates["per-model mean (loo-dataset)"] = baseline_group_centre(truth, datasets, models, centre="mean")
    candidates["per-model median (loo-dataset)"] = baseline_group_centre(truth, datasets, models, centre="median")
    # The opaque opponent under every protocol it has, so each of its rows can be read against
    # the equation row that was allowed to see the same things. The loo-cell rows are the
    # comparison that matters: there neither side has the dataset or the model, and the trivial
    # baselines cannot be computed at all.
    if opaque is not None:
        candidates |= _opaque_candidates(opaque)

    tables = {
        label: ranking_report(truth, prediction, datasets).sort("group")
        for label, prediction in candidates.items()
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
    config: Configuration = DEFAULT_E3,
    known: dict[int, CrossValidation] | None = None,
    e3: EquationReport | None = None,
    opaque: OpaqueRun | None = None,
) -> pl.DataFrame:
    """The above-or-below-threshold decision, for the equation and both trivial centres.

    Every predictor under the same protocol, as in `ranking_baselines`, and the equation under
    all three so the cost of generalisation on the decision is measured rather than assumed.
    `decision_report` already carries a majority-class column, which is the floor any rule has
    to clear; these are the harder comparison.
    """
    columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
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
        doubly = doubly_held_out_predictions(frame, config)
        if doubly is not None:
            candidates["equation (loo-cell: both held out)"] = doubly
    else:
        path = known
        if path is None or config.headline_terms not in path:
            path, _ = _fixed_form_path(columns, truth, datasets, config)
        candidates["equation (loo-dataset)"] = path[config.headline_terms].predictions
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
    #: The same features under the full grammar (arity 3). Reported to show how far the
    #: additive form reaches, not as the study's recommendation. See `run_e3_capability`.
    e3_capability: EquationReport
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
    term_choice: pl.DataFrame
    #: Every length paired against the published one. This is the rule that chooses the
    #: length; `term_choice` holds the geometric readings that disagree with it.
    length_choice: pl.DataFrame
    pareto: pl.DataFrame
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
    #: What an unpenalised least-squares fit over the *whole* library reaches, in-sample and
    #: held out. The control for the selection stage: if this transferred, the beam search
    #: and the length rule would be machinery in search of a problem. See
    #: `analysis.saturated_fit`.
    saturated: pl.DataFrame


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
    e3_capability = run_e3_capability(frame)
    reach, ceiling = reach_analysis(frame, config_e3)
    # Fitted once and shared: the folds are the expensive part, and the regression table and
    # the two decision comparisons have to be scored from the same predictions or they can
    # disagree with each other.
    opaque_run = opaque_evaluate(frame)
    return Report(
        opaque=opaque_run.table,
        saturated=pl.DataFrame([saturated_analysis(frame, config_e3)]),
        e1=e1,
        e2=e2,
        e3=e3,
        e3_capability=e3_capability,
        practices=best_practices(e3.equation, columns, e3.stability),
        effects=term_effects(e3.equation, columns, DATASET_FEATURES, MODEL_FEATURES),
        shares=group_shares(e3.equation, columns, DATASET_FEATURES, MODEL_FEATURES),
        decomposition=variance_decomposition(target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)),
        correlations=correlation_analysis(frame, config_e3),
        reach=reach,
        ceiling=ceiling,
        baselines=baselines(frame),
        comparison=comparison(frame, e1, e3, e2, e3_capability),
        leakage=leakage_demonstration(frame, config_e3, e3.paths),
        selection=model_selection(frame, e3, "loo_cell", config_e3),
        decision=decision_quality(frame, config_e3, e3.paths.get("loo_dataset")),
        ranking_baselines=ranking_baselines(frame, e3, config_e3, opaque_run),
        decision_baselines=decision_baselines(frame, config_e3, e3.paths.get("loo_dataset"), e3, opaque_run),
        term_choice=recommend(e3.curve, published=len(e3.equation.terms)),
        length_choice=length_comparison(frame, e3, config_e3),
        pareto=pareto_table(e3.curve),
        oracles=oracle_ladder(target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)),
        interaction=interaction_reached(frame, e3),
        identity=identity_ceiling(frame, e3),
    )
