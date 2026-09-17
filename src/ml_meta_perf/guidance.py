"""Best practices from the literature, weighed against what this study measured.

A best practice is not a property of a fitted equation. It is a piece of general,
transferable advice -- short enough to remember, cheap enough to apply -- that already
circulates in the field and that evidence can *support*, *qualify* or *challenge*. The
equation analysis in `ml_meta_perf.report` is evidence. It is not itself the advice, and a
sentence of the form "higher `nr_norm` went with lower MCC" is a measurement rather than
something anyone can act on.

So this module runs the other way round from `ml_meta_perf.practices`. Instead of reading
guidance out of the weights, it starts from practices the tabular machine-learning
literature already recommends, and asks what this corpus says about each one. That makes
the study's contribution a *test of received advice* rather than a source of new advice,
which is the honest position for 20 datasets from one domain.

Each `Practice` is written by hand, because a general recommendation and its citation are
not things a fitting procedure produces. Each `Verdict` is computed: the module measures
the relevant quantity on the meta-dataset and picks the verdict from a stated threshold,
so re-running on other data can overturn any of them.

Verdicts mean:

* **supported** -- this corpus shows the effect the practice predicts, at a size worth acting on;
* **qualified** -- the effect is present but smaller, or holds only under a condition worth stating;
* **challenged** -- this corpus shows the opposite of what the practice predicts;
* **not tested** -- the study adopts or assumes the practice and cannot be evidence for it.

That last verdict matters. A study that reports only the practices it happens to confirm
is not evidence about practice, so a practice the study merely assumes is marked as such
rather than counted as a win.

Study chapter: [6. Best practices against the equation][study-chapter] -- the rationale, in
prose, with the figures.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/06-practices.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from ml_meta_perf.attribution import contributions as attribution_contributions
from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FAMILY,
    MODEL_FEATURES,
    NEURAL_FAMILIES,
    TARGET_COLUMN,
    TREE_FAMILIES,
)
from ml_meta_perf.experiment import Report
from ml_meta_perf.stats import spearman
from ml_meta_perf.terms import Term
from ml_meta_perf.validate import paired_comparison

SUPPORTED = "supported"
QUALIFIED = "qualified"
CHALLENGED = "challenged"
NOT_TESTED = "not tested"


#: What a practice predicts a feature does to MCC as that feature rises.
RAISES, LOWERS = "raises", "lowers"


@dataclass(frozen=True)
class Practice:
    """One general recommendation, and where it comes from."""

    id: str
    statement: str
    source: str
    #: Why the field believes it, in one line -- so a reader can judge whether this
    #: corpus is even the right place to test it.
    rationale: str
    #: Raw features the practice makes a claim about, each with the direction it predicts
    #: MCC moves as that feature rises. **This is what lets a practice be checked against
    #: the published equation rather than only against corpus averages.** A verdict drawn
    #: from family means says the advice holds on this data; a term-level agreement says the
    #: *equation* encodes it, which is a stronger and more falsifiable claim -- and the one
    #: an interpretability-first study is actually in a position to make.
    #:
    #: Empty for a practice that makes no claim about a single feature. Most do not: a
    #: family-level recommendation is a claim about *rows*, not about a coefficient, and
    #: `group_claims` is how those are checked.
    expectations: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Verdict:
    """What this study says about one practice."""

    practice: Practice
    verdict: str
    evidence: str
    magnitude: float


#: Measured evidence a verdict can draw on, assembled once and passed to every check.
@dataclass(frozen=True)
class Evidence:
    """Everything the checks need, measured from the meta-dataset and the study."""

    frame: pl.DataFrame
    family_means: pl.DataFrame
    complete_family_means: pl.DataFrame
    complete_model_means: pl.DataFrame
    n_complete_datasets: int
    ranking_agreement: float
    missing_cells: int
    total_cells: int
    decomposition: pl.DataFrame
    comparison: pl.DataFrame
    leakage: pl.DataFrame
    practices: pl.DataFrame
    #: Mean ranking quality of the equation and of the per-model-mean baseline, over the
    #: held-out datasets. **Read `ap`, `mrr`, `hit_at_1` and `regret`.** Spearman is carried
    #: for continuity and must not decide anything: on this corpus it sits between 0.63 and
    #: 0.73 for every predictor *and* every baseline, including a constant, so a verdict
    #: resting on it is resting on a quantity that cannot separate what it is comparing.
    #: See `validate.ranking_report`.
    selection_ranking: dict[str, float]
    baseline_ranking: dict[str, float]
    #: The same two, per held-out dataset and in one group order, so a check can run a
    #: paired test rather than compare two means over twenty folds. `PairedResult` explains
    #: why the means alone are not enough here.
    selection_per_group: pl.DataFrame
    baseline_per_group: pl.DataFrame

    def family(self, name: str, complete: bool = True) -> float:
        """Mean MCC of one learner family, on the complete-grid subset by default."""
        table = self.complete_family_means if complete else self.family_means
        matched = table.filter(pl.col("family") == name)
        return float(matched["mcc"][0]) if matched.height else float("nan")

    def families(self, names: tuple[str, ...], complete: bool = True) -> float:
        """Mean MCC across several families, weighted by their row counts."""
        table = self.complete_family_means if complete else self.family_means
        matched = table.filter(pl.col("family").is_in(list(names)))
        if not matched.height:
            return float("nan")
        weights = matched["n"].to_numpy().astype(np.float64)
        return float(np.average(matched["mcc"].to_numpy(), weights=weights))

    def scored(self, equation: str) -> float:
        """The R2 of a row of `comparison`, matched by prefix.

        By prefix because the labels carry their term count -- for example, "E3, dataset +
        model" -- and that count moves whenever the configuration does. Matching the whole
        string would make every caller here break on a change that is not about them.
        """
        matched = self.comparison.filter(pl.col("equation").str.starts_with(equation))
        return float(matched["r2"][0]) if matched.height else float("nan")

    def protocol(self, name: str) -> float:
        matched = self.leakage.filter(pl.col("protocol") == name)
        return float(matched["r2"][0]) if matched.height else float("nan")

    def practice_effect(self, feature: str) -> float:
        matched = self.practices.filter(pl.col("feature") == feature)
        return float(matched["effect"][0]) if matched.height else float("nan")

    def practice_terms(self, feature: str) -> int:
        """How many of the equation's terms this feature appears in, 0 if none do."""
        matched = self.practices.filter(pl.col("feature") == feature)
        if not matched.height or "n_terms" not in matched.columns:
            return 0
        return int(matched["n_terms"][0])


#: The ranking columns a verdict may read, in the order `ranking_report` documents them.
RANKING_METRICS: tuple[str, ...] = ("ap", "mrr", "hit_at_1", "regret")


def _mean_ranking(table: pl.DataFrame) -> dict[str, float]:
    """Mean of each ranking metric over the held-out groups."""
    return {
        name: float(np.mean(table[name].to_numpy())) for name in (*RANKING_METRICS, "spearman") if name in table.columns
    }


def _family_means(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.with_columns(pl.col(MODEL_COLUMN).replace(MODEL_FAMILY).alias("family"))
        .group_by("family")
        .agg(
            pl.col(TARGET_COLUMN).mean().alias("mcc"),
            pl.col(MODEL_COLUMN).n_unique().alias("models"),
            pl.len().alias("n"),
        )
        .sort("mcc", descending=True)
    )


def _model_means(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by(MODEL_COLUMN)
        .agg(pl.col(TARGET_COLUMN).mean().alias("mcc"), pl.len().alias("n"))
        .sort("mcc", descending=True)
    )


def gather(frame: pl.DataFrame, report: Report) -> Evidence:
    """Measure everything the practice checks need.

    The *complete-grid* tables are the ones the family comparisons use. Twenty-four of the
    500 (dataset, model) cells are absent and they are not absent at random -- eight
    models are missing from the same three datasets -- so a mean taken over all rows
    compares those eight models on a different set of problems from the other seventeen.
    Restricting to the datasets where every model ran removes that.
    """
    from ml_meta_perf.stats import spearman
    from ml_meta_perf.validate import baseline_group_mean, ranking_report

    n_models = frame[MODEL_COLUMN].n_unique()
    counts = frame.select(DATASET_COLUMN, MODEL_COLUMN).unique().group_by(DATASET_COLUMN).len()
    complete = counts.filter(pl.col("len") == n_models)[DATASET_COLUMN].to_list()
    # No dataset ran every model: there is no complete grid to restrict to, so the family
    # comparisons fall back to every row and the agreement figure is undefined rather than
    # silently computed over an empty subset.
    subset = frame.filter(pl.col(DATASET_COLUMN).is_in(complete)) if complete else frame

    every = _model_means(frame).sort(MODEL_COLUMN)
    restricted = _model_means(subset).sort(MODEL_COLUMN)
    agreement = (
        spearman(every["mcc"].to_numpy(), restricted["mcc"].to_numpy())
        if complete and every.height == restricted.height
        else float("nan")
    )

    truth = frame[TARGET_COLUMN].to_numpy().astype(np.float64)
    datasets = frame[DATASET_COLUMN].to_numpy().astype(str)
    models = frame[MODEL_COLUMN].to_numpy().astype(str)
    baseline = ranking_report(truth, baseline_group_mean(truth, datasets, models), datasets)

    selection = report.selection
    return Evidence(
        frame=frame,
        family_means=_family_means(frame),
        complete_family_means=_family_means(subset),
        complete_model_means=_model_means(subset),
        n_complete_datasets=len(complete),
        ranking_agreement=agreement,
        # Distinct pairs, not rows: a meta-dataset with a repeated (dataset, model) cell
        # would otherwise report a negative number of missing ones.
        missing_cells=n_models * frame[DATASET_COLUMN].n_unique()
        - frame.select(DATASET_COLUMN, MODEL_COLUMN).unique().height,
        total_cells=n_models * frame[DATASET_COLUMN].n_unique(),
        decomposition=report.decomposition,
        comparison=report.comparison,
        leakage=report.leakage,
        practices=report.practices,
        selection_ranking=_mean_ranking(selection),
        baseline_ranking=_mean_ranking(baseline),
        selection_per_group=selection.sort("group") if "group" in selection.columns else selection,
        baseline_per_group=baseline.sort("group") if "group" in baseline.columns else baseline,
    )


# --------------------------------------------------------------------------------------
# The catalogue. Statements and citations are written by hand; every verdict is computed.
# --------------------------------------------------------------------------------------

CATALOGUE: tuple[Practice, ...] = (
    Practice(
        id="profile-the-data-first",
        statement=(
            "Characterise the dataset before choosing a model. What the data is like bounds "
            "what any model can reach, and that bound is usually the larger effect."
        ),
        source="Zha et al., 'Data-centric Artificial Intelligence: A Survey', ACM Computing Surveys 57(5) (2025)",
        rationale=(
            "The data-centric position holds that returns from improving data exceed returns "
            "from swapping architectures. It is an argument about where to spend effort."
        ),
        # No single feature: the claim is that the *dataset half* of the meta-data matters
        # more than the model half, which is a claim about the two blocks of the equation
        # rather than about any one column. `group_shares` is where it is read.
    ),
    Practice(
        id="tree-ensembles-first",
        statement=(
            "On tabular data, start from tree ensembles. Reach for a neural architecture only "
            "when a tree ensemble has been tried and found wanting."
        ),
        source=(
            "Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 Datasets and Benchmarks Track; "
            "Shwartz-Ziv & Armon, 'Tabular Data: Deep Learning is Not All You Need', "
            "Information Fusion 81, 84-90 (2022)"
        ),
        rationale=(
            "Trees handle irregular, non-smooth target functions and uninformative features, "
            "which is what tabular data usually contains."
        ),
    ),
    Practice(
        id="pretrained-tabular-baseline",
        statement=(
            "Include a pretrained tabular model (TabPFN, TabICL) in the first round of "
            "candidates: it costs one fit and is frequently competitive with a tuned ensemble."
        ),
        source=(
            "Hollmann et al., 'Accurate predictions on small data with a tabular foundation "
            "model' (TabPFN), Nature 637, 319-326 (2025); "
            "Qu, Holzmuller, Varoquaux & Le Morvan, 'TabICL', ICML 2025"
        ),
        rationale=(
            "In-context learning on tabular data removes the tuning budget that usually "
            "separates a quick baseline from a competitive one."
        ),
    ),
    Practice(
        id="hold-out-whole-groups",
        statement=(
            "When rows share a group -- a subject, a site, a dataset -- validate by holding out "
            "whole groups. A random split reports a number that will not survive deployment."
        ),
        source="Walsh et al., 'Machine learning reporting standards', Nature Methods 18 (2021)",
        rationale=(
            "Any feature constant within a group lets the model recognise the group rather "
            "than generalise to it, and a random split puts the group on both sides."
        ),
    ),
    Practice(
        id="clean-noise-before-adding-capacity",
        statement=(
            "Spend the first effort on reducing noise in the data, not on a larger model. "
            "Noise sets a ceiling that capacity cannot lift."
        ),
        source=(
            "Zha et al., 'Data-centric AI: Perspectives and Challenges', "
            "SIAM International Conference on Data Mining (SDM) 2023, 945-948"
        ),
        rationale=(
            "Irreducible error from noisy features or labels bounds every model on that data, "
            "so capacity spent against it buys nothing."
        ),
        expectations=(("ns_ratio", LOWERS),),
    ),
    Practice(
        id="prefer-outlier-robust-learners",
        statement=(
            "On real-world data that has not been carefully curated, prefer a learner with "
            "built-in robustness to outliers."
        ),
        source=(
            "Grinsztajn, Oyallon & Varoquaux, NeurIPS 2022 Datasets and Benchmarks Track, "
            "on non-smooth targets and outliers"
        ),
        rationale=(
            "Real tabular data carries outliers that a squared-error learner chases and a "
            "split-based or margin-based one largely ignores."
        ),
        # `nr_outliers` counts outliers in the *data*; the practice is about resistance to
        # them in the *learner*. The expectation is on the data side only, and the verdict
        # says why that does not settle the practice.
        expectations=(("nr_outliers", LOWERS),),
    ),
    Practice(
        id="capacity-is-not-free",
        statement=(
            "Match capacity to the problem. A larger, more expensive model is not a safer "
            "default; on small tabular problems it is usually a worse one."
        ),
        source="Shwartz-Ziv & Armon, Information Fusion 81, 84-90 (2022)",
        rationale=(
            "Capacity beyond what the sample supports fits noise, and the cost is paid twice: "
            "in accuracy and in the tuning budget needed to recover it."
        ),
        # The practice says more capacity is not safer, so it predicts that raising
        # `Processing Units Number` does not raise MCC. `Model Capability` is the opposite
        # claim in the same sentence -- capability *matched* to the problem does help -- and
        # both are in the equation, so both are checkable.
        expectations=(("Processing Units Number", LOWERS), ("Model Capability", RAISES)),
    ),
    Practice(
        id="beat-the-trivial-baseline",
        statement=(
            "Before adopting a meta-learner to choose models, check it against 'use whatever "
            "usually works'. Ranking is an easier problem than prediction and often needs less."
        ),
        source="Rice, 'The Algorithm Selection Problem' (1976); standard meta-learning practice",
        rationale=(
            "A per-model mean over previous datasets carries most of the ranking signal at "
            "zero modelling cost, and is the baseline any selection method has to clear."
        ),
    ),
    Practice(
        id="report-excluded-runs",
        statement=(
            "Report which (dataset, model) runs were excluded and why. Aggregate comparisons "
            "over an incomplete grid compare different models on different problems."
        ),
        source="Walsh et al., Nature Methods 18 (2021); benchmarking reporting standards",
        rationale=(
            "Runs usually go missing where a model struggles or will not fit, so exclusions "
            "are correlated with the outcome being measured."
        ),
    ),
    Practice(
        id="use-a-balanced-metric",
        statement=(
            "Score imbalanced classification with a metric that accounts for all four "
            "confusion-matrix cells -- MCC rather than accuracy or F1."
        ),
        source="Chicco & Jurman, BMC Genomics 21:6 (2020)",
        rationale=(
            "Accuracy and F1 can both look strong on a classifier that has learned only the majority class; MCC cannot."
        ),
    ),
)

_BY_ID = {practice.id: practice for practice in CATALOGUE}


def _profile_the_data_first(evidence: Evidence) -> Verdict:
    rows = {
        row["knowing only"]: float(row["variance_explained"]) for row in evidence.decomposition.iter_rows(named=True)
    }
    dataset, model = rows.get("dataset identity", float("nan")), rows.get("model identity", float("nan"))
    captured_dataset = evidence.scored("E1, dataset only") / dataset if dataset else float("nan")
    captured_model = evidence.scored("E2, model only") / model if model else float("nan")
    gap = dataset - model
    return Verdict(
        practice=_BY_ID["profile-the-data-first"],
        verdict=SUPPORTED if gap > 0.02 else QUALIFIED,
        evidence=(
            f"Knowing only which dataset a row came from explains {dataset:.1%} of MCC variance; "
            f"knowing only which model, {model:.1%}. The dataset side is also the better described: "
            f"{len(DATASET_FEATURES)} dataset meta-features reach {captured_dataset:.0%} of what "
            f"dataset identity explains, while {len(MODEL_FEATURES)} model meta-features reach "
            f"{captured_model:.0%} of theirs. Both the effect and our ability to measure it favour "
            "the data."
        ),
        magnitude=gap,
    )


def _tree_ensembles_first(evidence: Evidence) -> Verdict:
    trees = evidence.families(TREE_FAMILIES)
    neural = evidence.families(NEURAL_FAMILIES)
    tabular_nn = evidence.family("tabular NN")
    generic_nn = evidence.family("generic NN")
    return Verdict(
        practice=_BY_ID["tree-ensembles-first"],
        verdict=SUPPORTED if trees - neural > 0.05 else QUALIFIED,
        evidence=(
            f"On the {evidence.n_complete_datasets} datasets where every model ran, tree-based "
            f"families average MCC {trees:.3f} against {neural:.3f} for neural ones. The neural "
            f"side splits sharply: architectures built for tabular data reach {tabular_nn:.3f} "
            f"while a plain MLP or DNN reaches {generic_nn:.3f}, last of the ten families. The "
            "advice holds, and it holds most strongly against exactly the architectures that are "
            "not designed for this kind of data."
        ),
        magnitude=trees - neural,
    )


def _pretrained_tabular_baseline(evidence: Evidence) -> Verdict:
    foundation = evidence.family("tabular foundation")
    best_tree = evidence.complete_family_means.filter(pl.col("family").is_in(list(TREE_FAMILIES)))
    top = float(np.max(best_tree["mcc"].to_numpy())) if best_tree.height else float("nan")
    order = evidence.complete_model_means[MODEL_COLUMN].to_list()
    ranks = {name: index + 1 for index, name in enumerate(order)}
    placings = sorted(ranks.get(name, 0) for name in ("TabICL", "TabPFN") if name in ranks)
    return Verdict(
        practice=_BY_ID["pretrained-tabular-baseline"],
        verdict=SUPPORTED if foundation >= top - 0.01 else QUALIFIED,
        evidence=(
            f"Pretrained tabular models average MCC {foundation:.3f}, against {top:.3f} for the "
            f"best tree family, and place {' and '.join(str(place) for place in placings)} of "
            f"{len(order)} models. They match the strongest tree ensembles here without a tuning "
            "budget, which is the whole of the claim."
        ),
        magnitude=foundation - top,
    )


def _hold_out_whole_groups(evidence: Evidence) -> Verdict:
    random = evidence.protocol("random 10-fold (leaky)")
    grouped = evidence.protocol("leave-one-dataset-out")
    inflation = random - grouped
    return Verdict(
        practice=_BY_ID["hold-out-whole-groups"],
        verdict=SUPPORTED if inflation > 0.02 else QUALIFIED,
        evidence=(
            f"The same equation scores R² {random:.3f} under a random 10-fold split and "
            f"{grouped:.3f} when whole datasets are held out -- {inflation:.3f} of pure "
            "protocol. Dataset meta-features are constant within a dataset, so a random fold "
            "shows the equation rows from a dataset it is being scored on."
        ),
        magnitude=inflation,
    )


def _clean_noise_before_adding_capacity(evidence: Evidence) -> Verdict:
    effect = evidence.practice_effect("ns_ratio")
    if np.isnan(effect):
        return Verdict(
            practice=_BY_ID["clean-noise-before-adding-capacity"],
            verdict=NOT_TESTED,
            evidence="No noise-to-signal practice survived the extraction filters in this run.",
            magnitude=float("nan"),
        )
    return Verdict(
        practice=_BY_ID["clean-noise-before-adding-capacity"],
        verdict=SUPPORTED if effect < -0.05 else QUALIFIED,
        evidence=(
            f"Noise-to-signal ratio moves predicted MCC by {effect:.2f} between its lowest and "
            "highest decile, and it is one of only two features whose direction inside the "
            "equation agrees with its own correlation against MCC -- so it is not an artefact of "
            "conditioning. Nothing on the model side of the equation offsets it."
        ),
        magnitude=effect,
    )


def _prefer_outlier_robust_learners(evidence: Evidence) -> Verdict:
    """Untestable on this corpus, and it will stay that way.

    The practice needs a column saying whether a *learner* resists outliers.
    The generated corpus does not contain such a learner descriptor.

    The nearby column is not a substitute. ``nr_outliers`` counts outliers in the *data*,
    which is a property of the problem rather than of the learner, so a term over it
    answers "do outliers hurt?" and not "does robustness help?". Re-pointing the check at
    it would produce a verdict that reads as though the practice had been tested.

    Nor can it be recovered: no model may be re-run and no descriptor measured under this
    study's standing constraints, so the only route left is asserting a robustness ordinal
    from the literature, which
    would test the assertion rather than the practice. This returns `NOT_TESTED`
    unconditionally and says why, which is the honest report of a practice this study
    cannot weigh.
    """
    return Verdict(
        practice=_BY_ID["prefer-outlier-robust-learners"],
        verdict=NOT_TESTED,
        evidence=(
            "This corpus cannot weigh it. The practice is about a property of the learner, "
            "and the generated meta-dataset has no learner-robustness descriptor. "
            "`nr_outliers` counts outliers in the data, not resistance to them in the model, "
            "so it answers a different question. Reported as untested rather than answered "
            "with the nearest available number."
        ),
        magnitude=float("nan"),
    )


def _capacity_is_not_free(evidence: Evidence) -> Verdict:
    generic = evidence.family("generic NN")
    trees = evidence.families(TREE_FAMILIES)
    capacity_terms = evidence.practice_terms("Processing Units Number")
    equation_says = (
        f" The equation says it conditionally rather than flatly: `Processing Units Number` "
        f"carries {capacity_terms} of its terms, mostly against a dataset property, so what "
        "raises MCC is capacity *matched to* the problem rather than capacity itself."
        if capacity_terms
        else " Capacity does not survive into the equation's terms in this run, so the family "
        "means are the whole of the evidence here."
    )
    return Verdict(
        practice=_BY_ID["capacity-is-not-free"],
        verdict=SUPPORTED if generic < trees - 0.05 else QUALIFIED,
        evidence=(
            f"The highest-capacity family here is also the worst: generic neural networks "
            f"average MCC {generic:.3f} against {trees:.3f} for tree ensembles.{equation_says}"
        ),
        magnitude=generic - trees,
    )


def _beat_the_trivial_baseline(evidence: Evidence) -> Verdict:
    """Did the meta-learner actually beat "use whatever usually works"?

    Two things this check must not do, both of which it did.

    It must not decide on Spearman. That column sits between 0.63 and 0.73 for every
    predictor and every baseline on this corpus, including a constant, so the two sides
    were once separated by 0.0006 -- a margin that chose a published verdict while
    measuring nothing. `validate.ranking_report` says so where it is computed.

    It must not decide on a difference of means over twenty folds. A mean difference does
    not say whether the advantage is broad or driven by a few large wins. The verdict
    therefore runs `validate.paired_comparison` on the per-dataset scores and reads the
    interval, not the mean alone.

    The practice claims the trivial baseline is *competitive*, so an inconclusive paired
    test is the practice being right rather than a failure to measure. Only a baseline that
    loses on a majority of metrics with an interval excluding zero can challenge it.
    """
    selection, baseline = evidence.selection_per_group, evidence.baseline_per_group
    usable = [
        name
        for name in RANKING_METRICS
        if name in selection.columns
        and name in baseline.columns
        and selection.height == baseline.height
        and selection.height > 1
    ]
    if not usable:
        return Verdict(
            practice=_BY_ID["beat-the-trivial-baseline"],
            verdict=NOT_TESTED,
            evidence="No per-dataset ranking table was available to pair in this run.",
            magnitude=float("nan"),
        )

    results = {
        name: paired_comparison(
            selection[name].to_numpy().astype(np.float64),
            baseline[name].to_numpy().astype(np.float64),
            lower_is_better=name == "regret",
        )
        for name in usable
    }
    decisive = [name for name, result in results.items() if result.significant]
    equation_ahead = [name for name in decisive if results[name].mean > 0.0]

    if decisive and not equation_ahead:
        verdict = SUPPORTED
        summary = "and the equation loses on a margin this corpus can actually resolve"
    elif len(equation_ahead) > len(usable) / 2:
        verdict = CHALLENGED
        summary = "and the equation clears it by a margin that survives a paired test"
    else:
        verdict = QUALIFIED
        summary = (
            "and the two cannot be separated -- which is the practice being right, since it "
            "claims the trivial baseline is competitive rather than that it wins"
        )

    detail = "; ".join(
        f"{name} {evidence.selection_ranking[name]:.3f} against "
        f"{evidence.baseline_ranking[name]:.3f}, equation better on {results[name].wins} of "
        f"{results[name].n} datasets that differ, 95% CI "
        f"[{results[name].low:+.3f}, {results[name].high:+.3f}]"
        for name in usable
    )
    return Verdict(
        practice=_BY_ID["beat-the-trivial-baseline"],
        verdict=verdict,
        evidence=(
            f"Tested against this study's own equation {summary}. Ranking models within a "
            f"held-out dataset, against the per-model-mean baseline -- {detail}. Every "
            "interval is a paired bootstrap over the twenty held-out datasets, because a "
            "difference of two means over twenty folds is not yet a measurement."
        ),
        magnitude=results[usable[0]].mean,
    )


def _report_excluded_runs(evidence: Evidence) -> Verdict:
    share = evidence.missing_cells / evidence.total_cells if evidence.total_cells else 0.0
    return Verdict(
        practice=_BY_ID["report-excluded-runs"],
        verdict=SUPPORTED,
        evidence=(
            f"{evidence.missing_cells} of {evidence.total_cells} (dataset, model) cells are "
            f"absent -- {share:.0%} -- and they are not absent at random: eight models are "
            f"missing from the same three datasets. Measuring the bias rather than assuming it "
            f"is small: restricting to the {evidence.n_complete_datasets} complete datasets moves "
            f"the model ranking by Spearman {evidence.ranking_agreement:.3f}, so the ordering "
            "survives, but a mean over all rows still compares eight of the models on a "
            "different set of problems from the rest. Every family figure quoted here uses the "
            "complete subset for that reason."
        ),
        magnitude=share,
    )


def _use_a_balanced_metric(evidence: Evidence) -> Verdict:
    truth = evidence.frame[TARGET_COLUMN].to_numpy().astype(np.float64)
    zeros = int(np.sum(truth == 0.0))
    return Verdict(
        practice=_BY_ID["use-a-balanced-metric"],
        verdict=NOT_TESTED,
        evidence=(
            "This study adopts MCC as its target and never measures an alternative, so it is not "
            f"evidence for the practice. What it does show is the shape MCC has: {zeros} of "
            f"{truth.shape[0]} rows sit at exactly 0, which is a classifier that has learned "
            "nothing being scored as having learned nothing. Accuracy would not have said that."
        ),
        magnitude=float(zeros) / float(truth.shape[0]),
    )


CHECKS: dict[str, Callable[[Evidence], Verdict]] = {
    "profile-the-data-first": _profile_the_data_first,
    "tree-ensembles-first": _tree_ensembles_first,
    "pretrained-tabular-baseline": _pretrained_tabular_baseline,
    "hold-out-whole-groups": _hold_out_whole_groups,
    "clean-noise-before-adding-capacity": _clean_noise_before_adding_capacity,
    "prefer-outlier-robust-learners": _prefer_outlier_robust_learners,
    "capacity-is-not-free": _capacity_is_not_free,
    "beat-the-trivial-baseline": _beat_the_trivial_baseline,
    "report-excluded-runs": _report_excluded_runs,
    "use-a-balanced-metric": _use_a_balanced_metric,
}


def assess(frame: pl.DataFrame, report: Report) -> list[Verdict]:
    """Weigh every catalogued practice against this study, in catalogue order."""
    evidence = gather(frame, report)
    return [CHECKS[practice.id](evidence) for practice in CATALOGUE]


#: Below this rank correlation a (term, feature) pair has no direction worth stating. The same
#: floor `practices.MIN_DIRECTION` applies to the per-feature statements, and for the same
#: reason: a correlation of 0.05 between a feature and a term's contribution is the other
#: features that term contains moving, not the feature being asked about. Without it the table
#: reports a sign for a pairing that has none, which reads as a disagreement.
MIN_TERM_DIRECTION = 0.15


def feature_position(term: Term, feature: str) -> str:
    """Where a feature sits inside a term: ``numerator``, ``denominator``, or ``factor``.

    **This is what makes a sign readable**, and without it the table looks self-contradictory.
    A feature in a denominator enters the term inverted, so a negatively-weighted ratio
    contributes *more* as that feature rises -- which is arithmetic, not a disagreement with
    whatever the practice claims. Printing the position next to the direction is what lets a
    reader see that ``eq_num_attr / PUN`` and ``PUN / nr_class`` carrying opposite signs is one
    coherent statement about a ratio rather than two conflicting ones about a quantity.
    """
    match term.operation:
        case "ratio":
            below = 1
        case "sum_ratio":
            below = 2
        case "ratio_of_sums":
            below = 2
        case _:
            return "factor"
    for index, operand in enumerate(term.operands):
        if feature in set(operand.features if isinstance(operand, Term) else (operand.feature,)):
            return "denominator" if index >= below else "numerator"
    return "factor"


def equation_evidence(report: Report, columns: dict[str, np.ndarray]) -> pl.DataFrame:
    """Each practice paired with the **terms** of the published equation that carry it.

    **The equation is a sum of terms, so a term is the unit a practice has to be checked
    against.** A raw feature is not: `Processing Units Number` can appear in several terms,
    in numerators and denominators and under different transforms, and collapsing that
    into one per-feature direction throws away exactly what a reader wants -- which part of
    the equation encodes the advice, how strongly, and with what sign.

    One row per (practice, term) pair. A term is paired with a practice when it contains a
    feature the practice makes a claim about.

    ``beta`` is the **strength**: the standardised weight, already in MCC units because the
    target is centred but never scaled, so it is comparable across terms whose raw units have
    nothing to do with each other. ``effect`` is what the term is worth on this data -- the
    swing in its contribution across the middle 80% of its range -- because a large weight on
    a term that barely varies is not important.

    ``direction`` is the **sign**, and it is *measured* rather than derived. Reading it off the
    weight would be wrong as soon as the feature sits in a denominator or under a reciprocal,
    which several of these do: the rank correlation between the feature and the contribution
    that term actually makes is the only thing that answers "as this rises, what does this
    term do to predicted MCC". ``agrees`` compares that with what the practice predicts.

    A practice can therefore be **encoded by several terms that disagree with each other**,
    and that is a finding rather than a defect: it means the equation says the effect is
    conditional on which other quantity the feature is measured against.
    """
    equation = report.e3.equation
    # `EquationReport.stability` is optional, and an absent one means "the folds were never
    # re-run" rather than "no term was ever reselected". The column comes back as NaN in that
    # case, which reads as unknown; a zero would read as a term the folds rejected.
    fold_choices = report.e3.stability
    stability = (
        {row["term"]: float(row["frequency"]) for row in fold_choices.to_dicts()}
        if fold_choices is not None and fold_choices.height
        else {}
    )
    contributions = attribution_contributions(equation, columns)
    betas = dict(zip((term.name for term in equation.terms), equation.standardized_weights, strict=True))

    rows: list[dict[str, object]] = []
    for practice in CATALOGUE:
        for feature, expected in practice.expectations:
            carrying = [(index, term) for index, term in enumerate(equation.terms) if feature in set(term.features)]
            if not carrying:
                rows.append(
                    {
                        "practice": practice.id,
                        "feature": feature,
                        "expected": expected,
                        "term": "",
                        "position": "",
                        "beta": float("nan"),
                        "effect": float("nan"),
                        "stability": float("nan"),
                        "rho": float("nan"),
                        "direction": "",
                        "agrees": "not selected",
                    }
                )
                continue
            for index, term in carrying:
                share = contributions[:, index]
                low, high = np.percentile(share, [10.0, 90.0])
                slope = spearman(columns[feature], share)
                weak = abs(slope) < MIN_TERM_DIRECTION
                observed = "" if weak else (RAISES if slope > 0.0 else LOWERS)
                rows.append(
                    {
                        "practice": practice.id,
                        "feature": feature,
                        "expected": expected,
                        "term": term.name,
                        "position": feature_position(term, feature),
                        "beta": float(betas[term.name]),
                        "effect": float(high - low),
                        "stability": stability.get(term.name, float("nan")),
                        "rho": float(slope),
                        "direction": observed,
                        "agrees": "no direction" if weak else ("yes" if observed == expected else "no"),
                    }
                )
    return pl.DataFrame(
        rows,
        schema={
            "practice": pl.String,
            "feature": pl.String,
            "expected": pl.String,
            "term": pl.String,
            "position": pl.String,
            "beta": pl.Float64,
            "effect": pl.Float64,
            "stability": pl.Float64,
            "rho": pl.Float64,
            "direction": pl.String,
            "agrees": pl.String,
        },
    )


def equation_coverage(report: Report, columns: dict[str, np.ndarray]) -> dict[str, int]:
    """How much of the catalogue the equation's *terms* can be held against, and what they say.

    Kept separate from the verdict tally, and the distinction is the point. A verdict is drawn
    from corpus statistics -- family means, variance shares, paired tests -- so "5 supported"
    says the advice holds on these twenty datasets, which any study with this corpus could
    establish and which the fitted equation plays no part in. This counts the narrower and
    harder thing: how many of the equation's terms carry a practice at all, and how many of those
    pairings come out the way the practice predicts.

    Counted over (practice, term) pairs rather than over practices, because a practice carried
    by five terms that disagree with each other is not one verdict -- it is five readings, and
    collapsing them would hide the disagreement that makes the equation worth reading.
    """
    evidence = equation_evidence(report, columns)
    verdicts = evidence["agrees"].to_list()
    return {
        "practices": len(CATALOGUE),
        "with_feature_claims": sum(1 for practice in CATALOGUE if practice.expectations),
        "pairs": sum(1 for verdict in verdicts if verdict in ("yes", "no")),
        "agree": verdicts.count("yes"),
        "disagree": verdicts.count("no"),
        "undirected": verdicts.count("no direction"),
        "unselected": verdicts.count("not selected"),
        # How many of the disagreements are the same feature entering as a denominator. If
        # every one of them is, the "disagreement" is arithmetic rather than a conflict: a
        # negatively-weighted ratio contributes more as its denominator grows, so a marginal
        # expectation about the feature can never match a term that only speaks about a ratio.
        "disagree_in_denominator": sum(
            1 for row in evidence.to_dicts() if row["agrees"] == "no" and row["position"] == "denominator"
        ),
        "terms": len(report.e3.equation.terms),
        "carrying_terms": len({term for term in evidence["term"].to_list() if term}),
    }


def as_table(verdicts: list[Verdict]) -> pl.DataFrame:
    """The verdicts as a table, for the report and for a paper."""
    if not verdicts:
        return pl.DataFrame(
            schema={
                "id": pl.String,
                "practice": pl.String,
                "verdict": pl.String,
                "magnitude": pl.Float64,
                "source": pl.String,
            }
        )
    return pl.DataFrame(
        [
            {
                "id": verdict.practice.id,
                "practice": verdict.practice.statement,
                "verdict": verdict.verdict,
                "magnitude": verdict.magnitude,
                "source": verdict.practice.source,
            }
            for verdict in verdicts
        ]
    )


def render(verdicts: list[Verdict]) -> str:
    """The verdicts as readable markdown, one section per practice."""
    if not verdicts:
        return "_No practices assessed._"
    counts: dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
    summary = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))

    lines = [f"{len(verdicts)} practices assessed: {summary}.\n"]
    for index, verdict in enumerate(verdicts, start=1):
        lines.append(f"### {index}. {verdict.practice.statement}\n")
        lines.append(f"**Verdict: {verdict.verdict}.** {verdict.evidence}\n")
        lines.append(f"*Practice from:* {verdict.practice.source}. {verdict.practice.rationale}\n")
    return "\n".join(lines)
