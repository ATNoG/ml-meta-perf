"""The end-to-end experiment: fit E1, E2 and E3, validate all three, and report.

The three equations differ only in which features they may draw on -- E1 sees dataset
meta-features, E2 sees model meta-features, E3 sees both -- so the gaps between them
measure what each half of the meta-data is worth.

The defaults below are the configuration selected by the corrected-corpus sweep. The broad
stage uses a composite objective to shortlist candidates. E3-Valid uses the retained
Combined-R² plateau rule, while E3-MAX independently maximises the worst R² over in-sample
(IS), leave-one-dataset-out (LODO), leave-one-model-out (LOMO), and doubly held-out (DHO)
validation.

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
from ml_meta_perf.opaque import evaluate as opaque_evaluate
from ml_meta_perf.practices import best_practices, feature_practices
from ml_meta_perf.search import prune, search
from ml_meta_perf.selection import (
    PLATEAU_TOLERANCE,
    PLATEAU_WINDOW,
    complexity,
    floor_argmax,
    floor_curve,
    most_capable,
    plateau_configuration,
    protocol_spread,
)
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
    oracle_ladder,
    paired_comparison,
    random_kfold_groups,
    ranking_report,
    score,
    term_stability,
)


@dataclass(frozen=True)
class Configuration:
    """The search knobs retained from the configuration sweep."""

    max_abs_zscore: float
    penalty: float
    pool_size: int
    #: The search horizon: the longest equation `fit` explores, and the range the curve covers.
    #: **Not the published length** -- selection derives that from the complete curve and
    #: carries it on `EquationReport.n_terms`.
    max_terms: int
    beam_width: int = 6
    max_arity: int = 3


#: **One configuration, shared by all three equations.** This keeps the E1/E2/E3 comparison
#: attributable to the feature sets rather than to separate tuning runs. The values come from
#: the exhaustive corrected-corpus E3 sweep; E1 and E2 use the same base settings.

#: The ridge penalty on standardised terms. The corrected-corpus sweep evaluated sixteen
#: values from 0.1 through 80; the shared base configuration whose four-protocol floor was
#: highest uses 1.0. This is applied to all three equations so their differences remain
#: attributable to their feature sets.
PENALTY = 1.0

#: The largest standard score a term may reach before it is rejected as a spike. The
#: corrected-corpus sweep tested 3.0, 3.5, 4.0, 4.25, 4.5 and 5.0 and selected 5.0. The cap
#: still excludes terms supported only by extreme rows; 5.0 is the measured setting of the
#: winning shared base rather than a manual relaxation.
MAX_ABS_ZSCORE = 5.0

#: Terms surviving the correlation screen into the beam.
POOL_SIZE = 600

#: Beam width. Measured over the full beam sweep: eight times the search converges to the
#: fourth decimal, so the width is a cost control rather than a tuned knob.
BEAM_WIDTH = 6

#: The longest equation the search explores. **The horizon, not the published length** --
#: E3-Valid applies `selection.plateau_configuration` over this complete horizon. E3-MAX
#: independently uses the four-protocol floor.
MAX_TERMS = 25

#: The grammars the arity search covers, and the default for ``--arity``.
#:
#: **Arity 4 is deliberately out on readability grounds.** The four-feature ratio-of-sums
#: remains reachable through the flag for exploratory runs, while the default sweep stops at
#: terms a reader can reasonably hold in one expression. Arity 1 is
#: admissible too and is never worth a default: a grammar with no products cannot express the
#: conditional claims the study is about.
ARITIES: tuple[int, ...] = (2, 3)

# The corrected-corpus sweep uses the historical composite objective only to make a diverse
# shortlist. E3-Valid selects arity 2 at 18 terms, immediately before the Combined-R2 curve's
# first sustained plateau. E3-MAX independently selects arity 3 at 25 terms by maximising the
# four-protocol R2 floor, including the cell in which both dataset and model are held out.
#
# The selected four-feature subset is `Model Capability`, `Processing Units Number`,
# `Fitting Regime`, `Loss Margin Behaviour`. The corpus still carries all six model features
# for identification; the equation uses the subset the recalibration selected for compression.
# What must not change is the corpus: `MODEL_FEATURES` is the schema `load` validates and the
# set `tests/test_model_features.py` checks identification against.
#
# E1 is fitted on all 476 rows like the other two. An earlier version fitted it on the 20
# aggregated per-dataset means, on the grounds that a predictor constant inside a group can
# only ever predict that group's mean anyway. That was true and still the wrong choice: it put
# E1's R2 on a 20-point denominator, so its headline could not be compared with E3's without a
# paragraph of explanation, and the 0.506 it produced read as *better* transfer than E3's
# 0.466 when on the common scale it is 0.217. See [chapter 4](https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/04-equation.md).
DEFAULT = Configuration(
    max_abs_zscore=MAX_ABS_ZSCORE,
    penalty=PENALTY,
    pool_size=POOL_SIZE,
    beam_width=BEAM_WIDTH,
    max_terms=MAX_TERMS,
    # Where a caller that does *not* search starts: E1 and E2 are fitted once, over the most
    # parsimonious grammar in `ARITIES`. E3's arity is chosen by `search_grammars`, which
    # replaces this field per candidate grammar.
    max_arity=min(ARITIES),
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
#: The corrected-corpus sweep chose this subset. `Solution Stochasticity` and
#: `Input Distribution Modelling` remain in the corpus and out of the equation.
EQUATION_MODEL_FEATURES: tuple[str, ...] = (
    "Model Capability",
    "Processing Units Number",
    "Fitting Regime",
    "Loss Margin Behaviour",
)

#: Lengths the E3 curve is reported at.
#:
#: Every length is retained so the sustained-plateau rule reads a complete, uniformly spaced
#: curve. This costs nothing beyond the fitted path because ``fit`` already builds every
#: equation up to ``max_terms``.
SWEEP_SIZES: tuple[int, ...] | None = None


@dataclass
class EquationReport:
    """One equation together with everything said about it."""

    equation: Equation
    #: The selected length. E1 and E2 use `selection.floor_argmax`; E3-Valid uses the plateau
    #: rule and E3-MAX uses the cross-protocol floor. Every output reads the result from here.
    n_terms: int
    #: The grammar this equation was searched under. **Every helper that refits must be given
    #: it**, because several of them rebuild the library from a `Configuration` and the arity
    #: on that object is the default rather than the one `search_grammars` chose. Before this
    #: field existed, a run whose chosen grammar was not the configuration's -- `--arity 3`
    #: reproduces it -- published an arity-3 equation and scored its DHO row on an
    #: arity-2 refit.
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


def run_equation(
    frame: pl.DataFrame,
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
    config: Configuration,
    name: str,
    sizes: tuple[int, ...] | None = None,
    selected_size: int | None = None,
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
    # DHO at **every** length, not just the selected one. E1, E2 and
    # E3-MAX use it through `selection.floor_curve`; E3-Valid reports it without selecting on
    # it because its Combined-R2 plateau is defined over the other three protocols.
    paths["loo_cell"] = cross_validate_doubly_held_out(
        library, columns, truth, datasets, models, result.equations, penalty=config.penalty
    )
    in_sample = {k: score(truth, eq.predict(columns)) for k, eq in result.equations.items()}
    curve = _curve(sizes or tuple(sorted(result.equations)), in_sample, paths, truth)

    # **The length is derived, not asserted.** `fit` returns an equation at every length in one
    # pass and every one of them has just been cross-validated. Callers may provide the result
    # of a cross-grammar rule; otherwise the single-grammar floor chooses it.
    size = floor_argmax(curve) if selected_size is None else selected_size
    if size not in result.equations:
        raise ValueError(f"selected equation length {size} is unavailable")

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
    equation = prune(result.equations[size], columns, truth, penalty=config.penalty)

    return EquationReport(
        equation=equation,
        n_terms=size,
        arity=config.max_arity,
        in_sample=score(truth, equation.predict(columns)).as_dict(),
        curve=curve,
        cross_validated={
            label: path[size].scores(truth).as_dict() | path[size].dispersion() for label, path in paths.items()
        },
        stability=term_stability(reselected, tuple(term.name for term in equation.terms)),
        paths=paths,
    )


def run_e1(frame: pl.DataFrame, config: Configuration = DEFAULT) -> EquationReport:
    """Dataset features only -- how much of MCC the data alone explains.

    Every model on a given dataset shares one feature vector, so this equation can only
    ever predict a per-dataset constant. That is not a flaw to be corrected, it is the
    control: whatever E3 reaches beyond this is what knowing the model buys.

    Least squares finds that per-dataset constant on its own, so the aggregation an
    earlier version performed up front was unnecessary as well as harmful to the
    comparison -- see the note on `DEFAULT`.
    """
    return run_equation(frame, DATASET_FEATURES, (), config, "E1")


def run_e2(frame: pl.DataFrame, config: Configuration = DEFAULT) -> EquationReport:
    """Model features only -- the mirror image of E1.

    There are six model features and five of them are constant per model, so the library
    is tiny and the equation is short by necessity rather than by choice. That is itself
    the finding: the meta-data describes datasets far better than it describes models.
    The constraint on repeated feature combinations binds hardest here for the same
    reason -- six features offer only fifteen pairs -- which is why E2 is six terms.

    Aggregating this one to 25 per-model means -- the mirror of what E1 used to do -- was
    measured and is worse. `Processing Units Number` is intentionally allowed to vary with
    the dataset because it is a capacity calculation over that dataset's shape. Averaging
    it away discards real variation, and the optimum collapses to a single term.
    """
    return run_equation(frame, (), MODEL_FEATURES, config, "E2")


@dataclass(frozen=True)
class GrammarSearch:
    """The retained E3-Valid and E3-MAX fits and their selection record."""

    #: One `EquationReport` per arity searched, each already at its own derived length.
    reports: dict[int, EquationReport]
    #: The arity selected by the E3-Valid plateau rule.
    valid: int
    #: The arity `selection.most_capable` chose: how far the additive form reaches.
    maximum: int
    valid_report: EquationReport
    maximum_report: EquationReport
    #: The selected equations and the metrics used to distinguish their roles.
    candidates: pl.DataFrame


def search_grammars(
    frame: pl.DataFrame,
    config: Configuration = DEFAULT,
    arities: tuple[int, ...] = ARITIES,
) -> GrammarSearch:
    """Fit E3 once per grammar and select E3-Valid and E3-MAX reproducibly.

    **Every grammar is fitted under the same configuration bar the arity.** That is the point:
    the study used to fit its published equation at arity 2 with penalty 20 and its capability
    bound at arity 3 with penalty 3, and a comparison between two equations tuned differently
    is not a comparison. It moves the bound's numbers -- it is no longer allowed its own
    shrinkage -- and what it buys is that "arity 3 reaches further" becomes a statement about
    the grammar rather than about two hyperparameter sets.

    E3-Valid uses the combined-R2 plateau rule retained by the configuration search. E3-MAX
    independently maximises the four-protocol R2 floor as a capability bound.
    """
    reports = {
        arity: run_equation(
            frame,
            DATASET_FEATURES,
            EQUATION_MODEL_FEATURES,
            dataclasses.replace(config, max_arity=arity),
            f"E3-arity{arity}",
            SWEEP_SIZES,
        )
        for arity in arities
    }
    curves = {arity: report.curve for arity, report in reports.items()}
    valid_arity, valid_size = plateau_configuration(
        curves,
        tolerance=PLATEAU_TOLERANCE,
        window=PLATEAU_WINDOW,
    )
    max_arity, max_size = most_capable(curves)

    valid_report = reports[valid_arity]
    if valid_report.n_terms != valid_size:
        valid_report = run_equation(
            frame,
            DATASET_FEATURES,
            EQUATION_MODEL_FEATURES,
            dataclasses.replace(config, max_arity=valid_arity),
            f"E3-arity{valid_arity}",
            SWEEP_SIZES,
            selected_size=valid_size,
        )
    maximum_report = reports[max_arity]

    # Named for the role the rule gave them, not for the grammar they were searched under. The
    # arity is a search detail; "E3" and "E3-capability" are what the chapters and the saved
    # equations refer to, and they must not change name because the search that found them did.
    if (max_arity, max_size) == (valid_arity, valid_size):
        valid_report.equation = dataclasses.replace(
            valid_report.equation,
            name=valid_report.equation.name.replace(f"E3-arity{valid_arity}", "E3"),
        )
        maximum_report = valid_report
    else:
        for report, arity, role in (
            (maximum_report, max_arity, "E3-capability"),
            (valid_report, valid_arity, "E3"),
        ):
            report.equation = dataclasses.replace(
                report.equation, name=report.equation.name.replace(f"E3-arity{arity}", role)
            )

    rows: list[dict[str, object]] = []
    selected_reports = [("E3-Valid", valid_report)]
    if maximum_report is valid_report:
        selected_reports[0] = ("E3-Valid + E3-MAX", valid_report)
    else:
        selected_reports.append(("E3-MAX", maximum_report))
    for role, report in selected_reports:
        arity = report.arity
        size = report.n_terms
        position = list(curves[arity]["n_terms"]).index(size)
        rows.append(
            {
                "arity": arity,
                "n_terms": size,
                "complexity": complexity(arity, size),
                "r2_in_sample": float(curves[arity]["r2_in_sample"][position]),
                "r2_loo_dataset": float(curves[arity]["r2_loo_dataset"][position]),
                "r2_loo_model": float(curves[arity]["r2_loo_model"][position]),
                "combined_r2": float(
                    np.median(
                        [
                            curves[arity][column][position]
                            for column in ("r2_in_sample", "r2_loo_dataset", "r2_loo_model")
                        ]
                    )
                ),
                "floor": float(floor_curve(curves[arity])[position]),
                "spread": float(protocol_spread(curves[arity])[position]),
                "role": role,
            }
        )
    return GrammarSearch(
        reports=reports,
        valid=valid_arity,
        maximum=max_arity,
        valid_report=valid_report,
        maximum_report=maximum_report,
        candidates=pl.DataFrame(rows),
    )


def run_e3(frame: pl.DataFrame, config: Configuration = DEFAULT) -> EquationReport:
    """Dataset and model features, selected by the retained plateau rule for one arity."""
    report = run_equation(frame, DATASET_FEATURES, EQUATION_MODEL_FEATURES, config, "E3", SWEEP_SIZES)
    _, size = plateau_configuration(
        {config.max_arity: report.curve},
        tolerance=PLATEAU_TOLERANCE,
        window=PLATEAU_WINDOW,
    )
    if report.n_terms == size:
        return report
    return run_equation(
        frame,
        DATASET_FEATURES,
        EQUATION_MODEL_FEATURES,
        config,
        "E3",
        SWEEP_SIZES,
        selected_size=size,
    )


def reach_analysis(frame: pl.DataFrame, config: Configuration = DEFAULT) -> tuple[pl.DataFrame, pl.DataFrame]:
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


def saturated_analysis(frame: pl.DataFrame, config: Configuration = DEFAULT) -> dict[str, float]:
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


def correlation_analysis(frame: pl.DataFrame, config: Configuration = DEFAULT, top: int = 15) -> pl.DataFrame:
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
    config: Configuration = DEFAULT,
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
    config: Configuration = DEFAULT,
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
    columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
    path = known
    if path is None or n_terms not in path:
        path, _ = _fixed_form_path(columns, truth, datasets, config)
    size = n_terms if n_terms in path else max(path)
    return decision_report(truth, path[size].predictions, datasets)


def length_comparison(frame: pl.DataFrame, e3: EquationReport, config: Configuration = DEFAULT) -> pl.DataFrame:
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


def doubly_held_out_predictions(
    frame: pl.DataFrame, n_terms: int, config: Configuration = DEFAULT
) -> np.ndarray | None:
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
    columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)
    library = build_library(
        DATASET_FEATURES,
        EQUATION_MODEL_FEATURES,
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
        columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
        return e3.equation.predict(columns)
    path = e3.paths.get(protocol)
    if not path:
        return None
    size = e3.n_terms if e3.n_terms in path else max(path)
    return path[size].predictions if size in path else None


def model_selection(
    frame: pl.DataFrame,
    e3: EquationReport,
    protocol: str = "loo_cell",
    config: Configuration = DEFAULT,
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
    config: Configuration = DEFAULT,
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
    config: Configuration = DEFAULT,
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
    #: The same features under the full grammar (arity 3). Reported to show how far the
    #: additive form reaches, not as the study's recommendation. See `search_grammars`.
    #: **It is the same object as `e3` when the rule picks one grammar for both**, which is a
    #: legitimate outcome: the bound and the equation coincide when the wider grammar earns
    #: its additional complexity.
    e3_capability: EquationReport
    #: One row per grammar searched, with the floor, the spread and the margin against the
    #: best -- `search_grammars` shows the rule's working.
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


def run(
    path: str | None = None,
    *,
    config_e1: Configuration | None = None,
    config_e2: Configuration | None = None,
    config_e3: Configuration | None = None,
    arities: tuple[int, ...] = ARITIES,
    opaque_models: tuple[tuple[str, OpaqueBuilder], ...] | None = None,
) -> Report:
    """Run the whole study.

    The three configurations default to the retained sweep settings. Passing them explicitly is how the
    command line exposes the knobs: a caller who overrides ``config_e3`` gets a study that is
    internally consistent, since every table that mentions E3 is computed from the same
    configuration object.

    **There is no ``quick`` any more.** It was a preset of three flags that already exist,
    and it silently reached two things they did not -- E2's configuration and the opaque
    ensemble sizes -- which is how it came to promise "seconds" while paying 226 s for a
    comparison. Every knob it set is now a parameter here, so a caller that wants a cheap run
    says which parts are cheap and the reader of that call site can see it.

    ``opaque_models`` replaces the estimators the comparison is run against, in the shape
    `opaque.evaluate` takes. It exists for the same reason that parameter does: the opaque
    side is scikit-learn's, the DHO refit around it is this project's, and a
    caller checking the wiring should be able to exercise the second without paying for the
    first -- which at 476 refits per estimator is most of what a run costs.
    """
    frame = load(path)
    config_e1 = config_e1 or DEFAULT
    config_e2 = config_e2 or DEFAULT
    config_e3 = config_e3 or DEFAULT
    e1 = run_e1(frame, config_e1)
    # One fit per grammar and the rule picks, rather than two hand-fixed arities. `e3` and
    # `e3_capability` are both drawn from this: the same search, the same configuration, the
    # arity the only difference between them.
    grammars = search_grammars(frame, config_e3, arities)
    e3 = grammars.valid_report
    e3_capability = grammars.maximum_report
    # **Every table below is built from the grammar the rule chose, not the one the
    # configuration happened to carry.** Several of these helpers rebuild the library and
    # refit; handing them `config_e3` scored the published equation's strictest protocol on a
    # different grammar whenever the two disagreed.
    config_e3 = dataclasses.replace(config_e3, max_arity=e3.arity)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    e2 = run_e2(frame, config_e2)
    reach, ceiling = reach_analysis(frame, config_e3)
    # Fitted once and shared: the folds are the expensive part, and the regression table and
    # the two decision comparisons have to be scored from the same predictions or they can
    # disagree with each other.
    opaque_run = opaque_evaluate(frame, models=opaque_models)
    return Report(
        opaque=opaque_run.table,
        saturated=pl.DataFrame([saturated_analysis(frame, config_e3)]),
        e1=e1,
        e2=e2,
        e3=e3,
        e3_capability=e3_capability,
        grammars=grammars.candidates,
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
        selection=model_selection(frame, e3, "loo_cell", config_e3),
        decision=decision_quality(frame, e3.n_terms, config_e3, e3.paths.get("loo_dataset")),
        ranking_baselines=ranking_baselines(frame, e3, config_e3, opaque_run),
        decision_baselines=decision_baselines(frame, config_e3, e3.paths.get("loo_dataset"), e3, opaque_run),
        length_choice=length_comparison(frame, e3, config_e3),
        oracles=oracle_ladder(target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)),
        interaction=interaction_reached(frame, e3),
        identity=identity_ceiling(frame, e3),
    )
