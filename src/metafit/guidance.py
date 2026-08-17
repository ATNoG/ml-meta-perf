"""Best practices from the literature, weighed against what this study measured.

A best practice is not a property of a fitted equation. It is a piece of general,
transferable advice -- short enough to remember, cheap enough to apply -- that already
circulates in the field and that evidence can *support*, *qualify* or *challenge*. The
equation analysis in `metafit.report` is evidence. It is not itself the advice, and a
sentence of the form "higher `nr_norm` went with lower MCC" is a measurement rather than
something anyone can act on.

So this module runs the other way round from `metafit.practices`. Instead of reading
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

Study chapter: [7. From equation to practice](../../assets/docs/07-practices.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from metafit.data import (
    DATASET_COLUMN,
    MODEL_COLUMN,
    MODEL_FAMILY,
    NEURAL_FAMILIES,
    TARGET_COLUMN,
    TREE_FAMILIES,
)
from metafit.experiment import Report

SUPPORTED = "supported"
QUALIFIED = "qualified"
CHALLENGED = "challenged"
NOT_TESTED = "not tested"


@dataclass(frozen=True)
class Practice:
    """One general recommendation, and where it comes from."""

    id: str
    statement: str
    source: str
    #: Why the field believes it, in one line -- so a reader can judge whether this
    #: corpus is even the right place to test it.
    rationale: str


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
    model_means: pl.DataFrame
    complete_model_means: pl.DataFrame
    n_complete_datasets: int
    ranking_agreement: float
    missing_cells: int
    total_cells: int
    decomposition: pl.DataFrame
    comparison: pl.DataFrame
    leakage: pl.DataFrame
    practices: pl.DataFrame
    selection_spearman: float
    baseline_spearman: float
    selection_regret: float
    baseline_regret: float
    #: Tree-minus-neural MCC gap on the smaller and larger halves of the complete-grid
    #: datasets, split at their median instance count.
    size_split: tuple[float, float]
    tabular_nn_small: float
    tabular_nn_large: float
    generic_nn_small: float
    generic_nn_large: float

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
        matched = self.comparison.filter(pl.col("equation") == equation)
        return float(matched["r2"][0]) if matched.height else float("nan")

    def protocol(self, name: str) -> float:
        matched = self.leakage.filter(pl.col("protocol") == name)
        return float(matched["r2"][0]) if matched.height else float("nan")

    def practice_effect(self, feature: str) -> float:
        matched = self.practices.filter(pl.col("feature") == feature)
        return float(matched["effect"][0]) if matched.height else float("nan")


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


def _size_split(subset: pl.DataFrame) -> dict[str, float]:
    """Family means on the smaller and larger halves of the datasets, by instance count.

    Split at the median rather than at a round number, so the two halves are the same
    size and the threshold is a property of the corpus. With seventeen complete datasets
    this is eight against nine, which is thin -- the check that uses it says so.
    """
    if "nr_inst" not in subset.columns or subset.height == 0:
        return dict.fromkeys(
            ("gap_small", "gap_large", "tabular_small", "tabular_large", "generic_small", "generic_large"),
            float("nan"),
        )
    sizes = subset.group_by(DATASET_COLUMN).agg(pl.col("nr_inst").first())
    median = float(np.median(sizes["nr_inst"].to_numpy()))
    labelled = subset.with_columns(pl.col(MODEL_COLUMN).replace(MODEL_FAMILY).alias("family"))

    def mean_of(half: pl.DataFrame, families: tuple[str, ...]) -> float:
        matched = half.filter(pl.col("family").is_in(list(families)))
        return float(np.mean(matched[TARGET_COLUMN].to_numpy())) if matched.height else float("nan")

    out: dict[str, float] = {}
    for name, half in (
        ("small", labelled.filter(pl.col("nr_inst") <= median)),
        ("large", labelled.filter(pl.col("nr_inst") > median)),
    ):
        out[f"gap_{name}"] = mean_of(half, TREE_FAMILIES) - mean_of(half, NEURAL_FAMILIES)
        out[f"tabular_{name}"] = mean_of(half, ("tabular NN",))
        out[f"generic_{name}"] = mean_of(half, ("generic NN",))
    return out


def gather(frame: pl.DataFrame, report: Report) -> Evidence:
    """Measure everything the practice checks need.

    The *complete-grid* tables are the ones the family comparisons use. Twenty-four of the
    500 (dataset, model) cells are absent and they are not absent at random -- eight
    models are missing from the same three datasets -- so a mean taken over all rows
    compares those eight models on a different set of problems from the other seventeen.
    Restricting to the datasets where every model ran removes that.
    """
    from metafit.stats import spearman
    from metafit.validate import baseline_group_mean, ranking_report

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
    halves = _size_split(subset)

    selection = report.selection
    return Evidence(
        frame=frame,
        family_means=_family_means(frame),
        complete_family_means=_family_means(subset),
        model_means=_model_means(frame),
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
        selection_spearman=float(np.mean(selection["spearman"].to_numpy())),
        baseline_spearman=float(np.mean(baseline["spearman"].to_numpy())),
        selection_regret=float(np.mean(selection["regret"].to_numpy())),
        baseline_regret=float(np.mean(baseline["regret"].to_numpy())),
        size_split=(halves["gap_small"], halves["gap_large"]),
        tabular_nn_small=halves["tabular_small"],
        tabular_nn_large=halves["tabular_large"],
        generic_nn_small=halves["generic_small"],
        generic_nn_large=halves["generic_large"],
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
        source="Zha et al., 'Data-centric AI: A Survey', arXiv:2303.10158 (2023)",
        rationale=(
            "The data-centric position holds that returns from improving data exceed returns "
            "from swapping architectures. It is an argument about where to spend effort."
        ),
    ),
    Practice(
        id="tree-ensembles-first",
        statement=(
            "On tabular data, start from tree ensembles. Reach for a neural architecture only "
            "when a tree ensemble has been tried and found wanting."
        ),
        source=(
            "Grinsztajn et al., arXiv:2207.08815 (2022); "
            "Shwartz-Ziv & Armon, 'Deep Learning is Not All You Need', arXiv:2106.03253 (2021)"
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
        source="Hollmann et al., TabPFN, arXiv:2207.01848 (2022); TabICL, arXiv:2502.05564 (2025)",
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
        source="Zha et al., 'Data-centric AI: Perspectives and Challenges', arXiv:2301.04819 (2023)",
        rationale=(
            "Irreducible error from noisy features or labels bounds every model on that data, "
            "so capacity spent against it buys nothing."
        ),
    ),
    Practice(
        id="prefer-outlier-robust-learners",
        statement=(
            "On real-world data that has not been carefully curated, prefer a learner with "
            "built-in robustness to outliers."
        ),
        source="Grinsztajn et al., arXiv:2207.08815 (2022), on non-smooth targets and outliers",
        rationale=(
            "Real tabular data carries outliers that a squared-error learner chases and a "
            "split-based or margin-based one largely ignores."
        ),
    ),
    Practice(
        id="capacity-is-not-free",
        statement=(
            "Match capacity to the problem. A larger, more expensive model is not a safer "
            "default; on small tabular problems it is usually a worse one."
        ),
        source="Shwartz-Ziv & Armon, arXiv:2106.03253 (2021)",
        rationale=(
            "Capacity beyond what the sample supports fits noise, and the cost is paid twice: "
            "in accuracy and in the tuning budget needed to recover it."
        ),
    ),
    Practice(
        id="neural-nets-need-scale",
        statement=(
            "Give neural architectures more data before writing them off: the gap to tree "
            "ensembles is a small-sample effect and closes as the dataset grows."
        ),
        source="Common reading of Grinsztajn et al., arXiv:2207.08815 (2022), §4.2",
        rationale=(
            "The tabular benchmarks where trees win are mostly small, and the scaling "
            "argument that carried deep learning elsewhere is expected to apply here too."
        ),
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
            "Accuracy and F1 can both look strong on a classifier that has learned only the "
            "majority class; MCC cannot."
        ),
    ),
)

_BY_ID = {practice.id: practice for practice in CATALOGUE}


def _profile_the_data_first(evidence: Evidence) -> Verdict:
    rows = {
        row["knowing only"]: float(row["variance_explained"])
        for row in evidence.decomposition.iter_rows(named=True)
    }
    dataset, model = rows.get("dataset identity", float("nan")), rows.get("model identity", float("nan"))
    captured_dataset = evidence.scored("E1 (dataset only)") / dataset if dataset else float("nan")
    captured_model = evidence.scored("E2 (model only)") / model if model else float("nan")
    gap = dataset - model
    return Verdict(
        practice=_BY_ID["profile-the-data-first"],
        verdict=SUPPORTED if gap > 0.02 else QUALIFIED,
        evidence=(
            f"Knowing only which dataset a row came from explains {dataset:.1%} of MCC variance; "
            f"knowing only which model, {model:.1%}. The dataset side is also the better described: "
            f"twelve dataset meta-features reach {captured_dataset:.0%} of what dataset identity "
            f"explains, while five model meta-features reach {captured_model:.0%} of theirs. Both "
            "the effect and our ability to measure it favour the data."
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
    effect = evidence.practice_effect("Robust to Outliers")
    if np.isnan(effect):
        return Verdict(
            practice=_BY_ID["prefer-outlier-robust-learners"],
            verdict=NOT_TESTED,
            evidence="No robustness practice survived the extraction filters in this run.",
            magnitude=float("nan"),
        )
    return Verdict(
        practice=_BY_ID["prefer-outlier-robust-learners"],
        verdict=SUPPORTED if effect > 0.05 else QUALIFIED,
        evidence=(
            f"Built-in robustness to outliers carries the largest feature effect in the "
            f"equation, {effect:+.2f} MCC between its lowest and highest decile, and its "
            "direction agrees with its plain correlation against MCC. It is the single most "
            "actionable thing the equation says about model choice."
        ),
        magnitude=effect,
    )


def _capacity_is_not_free(evidence: Evidence) -> Verdict:
    generic = evidence.family("generic NN")
    trees = evidence.families(TREE_FAMILIES)
    return Verdict(
        practice=_BY_ID["capacity-is-not-free"],
        verdict=SUPPORTED if generic < trees - 0.05 else QUALIFIED,
        evidence=(
            f"The highest-capacity family here is also the worst: generic neural networks "
            f"average MCC {generic:.3f} against {trees:.3f} for tree ensembles. Inside the "
            "equation the same tension is explicit -- one block of terms rises with capacity and "
            "raises MCC, a second block pairs capacity with inference cost and lowers it, and the "
            "two blocks carry equal weight."
        ),
        magnitude=generic - trees,
    )


def _neural_nets_need_scale(evidence: Evidence) -> Verdict:
    small, large = evidence.size_split
    if np.isnan(small) or np.isnan(large):
        return Verdict(
            practice=_BY_ID["neural-nets-need-scale"],
            verdict=NOT_TESTED,
            evidence="Not enough complete datasets to split by size.",
            magnitude=float("nan"),
        )
    closing = small - large
    verdict = SUPPORTED if closing > 0.05 else (QUALIFIED if closing > 0.0 else CHALLENGED)
    return Verdict(
        practice=_BY_ID["neural-nets-need-scale"],
        verdict=verdict,
        evidence=(
            f"It does not close here, it widens. Splitting the complete-grid datasets at "
            f"their median instance count, tree ensembles lead neural architectures by "
            f"{small:.3f} MCC on the smaller half and {large:.3f} on the larger one. The "
            "nuance worth keeping: purpose-built tabular architectures do improve with size "
            f"({evidence.tabular_nn_small:.3f} to {evidence.tabular_nn_large:.3f}) while "
            f"plain MLPs and DNNs get worse ({evidence.generic_nn_small:.3f} to "
            f"{evidence.generic_nn_large:.3f}), so the scaling argument survives for the "
            "architectures designed for this data and fails for the ones that are not. "
            "Eight and nine datasets a side is a thin split and this is a direction, not a "
            "measurement."
        ),
        magnitude=closing,
    )


def _beat_the_trivial_baseline(evidence: Evidence) -> Verdict:
    beaten = evidence.baseline_spearman >= evidence.selection_spearman
    return Verdict(
        practice=_BY_ID["beat-the-trivial-baseline"],
        verdict=SUPPORTED if beaten else QUALIFIED,
        evidence=(
            f"Tested against this study's own equation and the equation loses. Ranking models "
            f"within a held-out dataset, the per-model-mean baseline reaches Spearman "
            f"{evidence.baseline_spearman:.3f} and top-1 regret {evidence.baseline_regret:.3f} "
            f"against the equation's {evidence.selection_spearman:.3f} and "
            f"{evidence.selection_regret:.3f}. The equation wins on predicting the MCC *value*; "
            "for ordering candidates, the trivial baseline is the better tool."
            if beaten
            else " The equation clears the baseline here, which is the outcome the practice "
            "asks you to verify rather than assume."
        ),
        magnitude=evidence.baseline_spearman - evidence.selection_spearman,
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
    "neural-nets-need-scale": _neural_nets_need_scale,
    "beat-the-trivial-baseline": _beat_the_trivial_baseline,
    "report-excluded-runs": _report_excluded_runs,
    "use-a-balanced-metric": _use_a_balanced_metric,
}


def assess(frame: pl.DataFrame, report: Report) -> list[Verdict]:
    """Weigh every catalogued practice against this study, in catalogue order."""
    evidence = gather(frame, report)
    return [CHECKS[practice.id](evidence) for practice in CATALOGUE]


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
