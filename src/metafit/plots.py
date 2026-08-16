"""Figures for the study.

matplotlib is imported through the Agg backend so these run headless, in CI and over
ssh, without a display. Every function takes a destination path, writes one figure, and
returns that path, so the caller decides where output lands and nothing is written as a
side effect of importing.

**No titles and no annotations.** These figures are written for a LaTeX document where
the caption carries the description, and a title rendered into the PNG would duplicate it
in a font the document cannot restyle. What stays is the part a caption cannot replace:
axis labels, tick labels, and legends identifying the series. Anything a reader would
otherwise have to be *told* is instead drawn -- a reference level becomes a line with a
legend entry, not a sentence.

Each figure is one axes with one message, so each gets its own caption. Panels sharing a
figure would need panel titles to be distinguishable, which is the thing being avoided.

The palette is colour-blind safe and every series is distinguished by marker or line
style as well as colour, so the figures survive being printed in greyscale.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

IN_SAMPLE = "#1b6ca8"
LOO_DATASET = "#d1495b"
LOO_MODEL = "#00798c"
COMPARISON = "#8b5cf6"
CEILING = "#6b7280"
POSITIVE = "#00798c"
NEGATIVE = "#d1495b"

FIGURE_DPI = 150


def _finish(figure: Figure, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)
    return path


def term_count_curve(
    curve: pl.DataFrame,
    destination: str | Path,
    *,
    oracle: float | None = None,
    comparison: pl.DataFrame | None = None,
    comparison_label: str = "in-sample (accuracy-leaning)",
) -> Path:
    """Accuracy against equation length: the explainability trade.

    The oracle line is the point of the figure. Without it a reader sees a curve still
    climbing and assumes more terms would keep paying, when the whole approach is bounded
    well below 1. It is drawn as a labelled line rather than described in text.

    ``comparison`` overlays a second configuration's in-sample curve, so the cost of
    tuning for fit rather than transfer is visible in one figure instead of two.
    """
    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    sizes = curve["n_terms"].to_numpy()

    axes.plot(sizes, curve["r2_in_sample"].to_numpy(), "o-", color=IN_SAMPLE, label="in-sample", linewidth=2)
    if "r2_loo_dataset" in curve.columns:
        axes.plot(
            sizes, curve["r2_loo_dataset"].to_numpy(), "s--", color=LOO_DATASET, label="leave-one-dataset-out"
        )
    if "r2_loo_model" in curve.columns:
        axes.plot(
            sizes, curve["r2_loo_model"].to_numpy(), "^:", color=LOO_MODEL, label="leave-one-model-out"
        )
    if comparison is not None:
        axes.plot(
            comparison["n_terms"].to_numpy(),
            comparison["r2_in_sample"].to_numpy(),
            "D-.",
            color=COMPARISON,
            label=comparison_label,
            markersize=4,
        )
    if oracle is not None:
        axes.axhline(
            oracle, color=CEILING, linestyle="-.", linewidth=1.2,
            label=f"additive ceiling ({oracle:.3f})",
        )
        axes.set_ylim(top=oracle + 0.05)

    axes.set_xlabel("number of terms")
    axes.set_ylabel("$R^2$")
    axes.set_xticks(sizes)
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False, loc="lower right", fontsize=9)
    return _finish(figure, destination)


def error_curve(
    curve: pl.DataFrame,
    destination: str | Path,
    *,
    metric: str = "mae",
    marker: int | None = None,
    comparison: pl.DataFrame | None = None,
) -> Path:
    """Error against equation length, in the target's own units.

    R2 answers "how much variance is explained", which is a relative question. MAE answers
    "how far off is a prediction, in MCC", which is the one a practitioner asks. SMAPE is
    included as the scale-free alternative, with the caveat that on a target passing
    through zero it is dominated by the 38 rows at exactly MCC = 0.

    Both configurations appear on every curve: ``curve`` supplies all three protocols and
    ``comparison`` overlays the accuracy-leaning configuration's in-sample line, so fit
    and transfer for both tunings are read from one figure.

    ``marker`` draws a vertical line at a chosen equation length -- the knee, typically --
    as a labelled line rather than an annotation.
    """
    label = {"mae": "mean absolute error (MCC)", "smape": "SMAPE (%)"}.get(metric, metric)
    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    sizes = curve["n_terms"].to_numpy()

    for column, colour, style, name in (
        (f"{metric}_in_sample", IN_SAMPLE, "o-", "in-sample"),
        (f"{metric}_loo_dataset", LOO_DATASET, "s--", "leave-one-dataset-out"),
        (f"{metric}_loo_model", LOO_MODEL, "^:", "leave-one-model-out"),
    ):
        if column in curve.columns:
            axes.plot(sizes, curve[column].to_numpy(), style, color=colour, label=name, linewidth=2)
    if comparison is not None and f"{metric}_in_sample" in comparison.columns:
        axes.plot(
            comparison["n_terms"].to_numpy(),
            comparison[f"{metric}_in_sample"].to_numpy(),
            "D-.",
            color=COMPARISON,
            label="in-sample (accuracy-leaning)",
            markersize=4,
        )
    if marker is not None:
        axes.axvline(marker, color=CEILING, linestyle="-.", linewidth=1.2, label=f"knee ({marker} terms)")

    axes.set_xlabel("number of terms")
    axes.set_ylabel(label)
    axes.set_xticks(sizes)
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False, fontsize=9)
    return _finish(figure, destination)


def scatter_limits(
    truth: np.ndarray,
    predicted: np.ndarray,
    margin: float = 0.05,
    floor: float = 0.0,
) -> tuple[float, float]:
    """Square axis bounds covering both series, padded by ``margin`` and stopping at ``floor``.

    Deliberately not MCC's theoretical [-1, 1]. Nothing here approaches -1 -- the observed
    minimum is -0.29, one row -- and the interval below zero is worse than empty, it is
    uninformative: around MCC = 0 sit 38 degenerate results and one anti-correlated one,
    carrying no linear structure for the diagonal to be read against. Including them only
    compresses the range where the relationship lives.
    """
    lowest = max(float(min(truth.min(), predicted.min())) - margin, floor)
    highest = float(max(truth.max(), predicted.max())) + margin
    return lowest, highest


def count_below_floor(truth: np.ndarray, predicted: np.ndarray, floor: float = 0.0) -> int:
    """How many points ``predicted_versus_actual`` leaves outside its axes.

    Kept separate from the figure so the disclosure goes in the caption rather than being
    rendered into the image, where it could not be restyled or edited.
    """
    return int(np.sum((truth < floor) | (predicted < floor)))


def predicted_versus_actual(
    truth: np.ndarray,
    predicted: np.ndarray,
    destination: str | Path,
    *,
    margin: float = 0.05,
    floor: float = 0.0,
    groups: np.ndarray | None = None,
) -> Path:
    """A scatter against the diagonal.

    17% of the meta-dataset sits at exactly MCC = 1.0 and 8% at exactly 0.0. Both pile-ups
    are the most important thing to see about this target, and a scatter shows them in a
    way no summary statistic does.

    With ``groups`` supplied the marginal distribution of the truth is drawn as a rug
    along the bottom axis, which makes those pile-ups countable rather than merely visible
    as overplotted dots.
    """
    limits = scatter_limits(truth, predicted, margin, floor)

    figure, axes = plt.subplots(figsize=(5.6, 5.4))
    axes.plot(limits, limits, color=CEILING, linewidth=1.0, linestyle="--", label="perfect", zorder=1)
    axes.scatter(truth, predicted, s=18, alpha=0.55, color=IN_SAMPLE, edgecolor="none", zorder=2)
    if groups is not None:
        axes.plot(
            truth, np.full_like(truth, limits[0]), "|", color=IN_SAMPLE, alpha=0.35,
            markersize=6, zorder=1,
        )
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_aspect("equal")
    axes.set_xlabel("actual MCC")
    axes.set_ylabel("predicted MCC")
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False, loc="upper left", fontsize=9)
    return _finish(figure, destination)


def equation_comparison(comparison: pl.DataFrame, destination: str | Path) -> Path:
    """Every equation and every ceiling on one scale.

    The headline result of the study. Equations are drawn solid and the ceilings they are
    bounded by are drawn hatched immediately beside them, so the gap each equation leaves
    against its own limit is read directly off the figure instead of computed by the
    reader from a table.
    """
    labels = comparison["equation"].to_list()
    values = comparison["r2"].to_numpy()
    is_ceiling = [("ceiling" in label) or ("oracle" in label) for label in labels]

    figure, axes = plt.subplots(figsize=(8.6, 4.2))
    positions = np.arange(len(labels))
    axes.bar(
        positions,
        values,
        color=[CEILING if ceiling else IN_SAMPLE for ceiling in is_ceiling],
        hatch=["//" if ceiling else "" for ceiling in is_ceiling],
        alpha=0.85,
        edgecolor="white",
    )
    axes.set_xticks(positions)
    axes.set_xticklabels([_wrap(label) for label in labels], fontsize=8)
    axes.set_ylabel("$R^2$ on all 476 rows")
    axes.grid(axis="y", alpha=0.25, linestyle=":")

    solid = Rectangle((0, 0), 1, 1, facecolor=IN_SAMPLE, alpha=0.85)
    hatched = Rectangle((0, 0), 1, 1, facecolor=CEILING, alpha=0.85, hatch="//")
    axes.legend([solid, hatched], ["fitted equation", "ceiling"], frameon=False, fontsize=9)
    return _finish(figure, destination)


def oracle_ladder(ladder: pl.DataFrame, destination: str | Path, *, achieved: float | None = None) -> Path:
    """How fast the ceiling climbs as interaction components are added.

    Rank 0 is the additive oracle. Each further component is one more pattern of "this
    kind of model suits this kind of dataset", and the steepness of the first step is the
    measure of how much interaction structure the data holds. With ``achieved`` supplied
    the equation's own score is drawn as a line, showing which rung it has reached.
    """
    ranks = ladder["interaction_rank"].to_numpy()
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    axes.plot(ranks, ladder["r2"].to_numpy(), "o-", color=CEILING, linewidth=2, label="oracle ceiling")
    if achieved is not None:
        axes.axhline(
            achieved, color=IN_SAMPLE, linestyle="--", linewidth=1.4,
            label=f"fitted equation ({achieved:.3f})",
        )
    axes.set_xlabel("rank of the interaction added to the additive oracle")
    axes.set_ylabel("$R^2$")
    axes.set_xticks(ranks)
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False, fontsize=9, loc="lower right")
    return _finish(figure, destination)


def term_effects(effects: pl.DataFrame, destination: str | Path, *, top: int = 12) -> Path:
    """Per-term effect sizes in MCC units, signed, strongest at the top."""
    table = effects.head(top).reverse()
    labels = [_shorten(name) for name in table["term"].to_list()]
    values = table["effect"].to_numpy() * np.sign(table["beta"].to_numpy())

    figure, axes = plt.subplots(figsize=(8.2, 0.42 * len(labels) + 1.2))
    axes.barh(
        range(len(labels)),
        values,
        color=[POSITIVE if value > 0 else NEGATIVE for value in values],
        alpha=0.85,
    )
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels, fontsize=8)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("effect on predicted MCC (10th to 90th percentile swing)")
    axes.grid(axis="x", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def practice_effects(practices: pl.DataFrame, destination: str | Path) -> Path:
    """Per-feature effects, the form the written practices are derived from.

    Bars are shaded by the confidence the practice was rated at, so effect size and
    evidential weight are visible together -- a large effect from an unstable term looks
    different from a large effect from a stable one.
    """
    table = practices.reverse()
    labels = table["feature"].to_list()
    values = table["effect"].to_numpy()
    alphas = _confidence_alpha(table)

    figure, axes = plt.subplots(figsize=(7.8, 0.42 * len(labels) + 1.2))
    for index, (value, alpha) in enumerate(zip(values, alphas, strict=True)):
        axes.barh(index, value, color=POSITIVE if value > 0 else NEGATIVE, alpha=alpha)
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels, fontsize=9)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("MCC change from the feature's lowest decile to its highest")
    axes.grid(axis="x", alpha=0.25, linestyle=":")

    if "confidence" in table.columns:
        handles = [
            Rectangle((0, 0), 1, 1, facecolor=CEILING, alpha=alpha)
            for alpha in (0.95, 0.6, 0.3)
        ]
        axes.legend(handles, ["strong", "moderate", "weak"], frameon=False, fontsize=8, loc="lower right")
    return _finish(figure, destination)


def _confidence_alpha(table: pl.DataFrame) -> list[float]:
    scale = {"strong": 0.95, "moderate": 0.6, "weak": 0.3, "unrated": 0.6}
    if "confidence" not in table.columns:
        return [0.85] * table.height
    return [scale.get(str(value), 0.6) for value in table["confidence"].to_list()]


def protocol_comparison(leakage: pl.DataFrame, destination: str | Path) -> Path:
    """The same equation under three splits -- the leakage figure."""
    labels = leakage["protocol"].to_list()
    values = leakage["r2"].to_numpy()

    figure, axes = plt.subplots(figsize=(6.4, 3.8))
    axes.bar(
        range(len(labels)),
        values,
        color=[LOO_DATASET if "random" in label else IN_SAMPLE for label in labels],
        alpha=0.85,
    )
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels([_wrap(label) for label in labels], fontsize=9)
    axes.set_ylabel("$R^2$")
    axes.grid(axis="y", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def contribution_shares(shares: pl.DataFrame, destination: str | Path) -> Path:
    """Which feature groups drive the equation's output variance."""
    groups = shares["group"].to_list()
    figure, axes = plt.subplots(figsize=(5.0, 3.6))
    axes.bar(range(len(groups)), shares["share"].to_numpy(), color=IN_SAMPLE, alpha=0.85)
    axes.set_xticks(range(len(groups)))
    axes.set_xticklabels(groups)
    axes.set_xlabel("features the term uses")
    axes.set_ylabel("share of the equation's output variance")
    axes.axhline(0.0, color="black", linewidth=0.8)
    axes.grid(axis="y", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def identity_ceilings(decomposition: pl.DataFrame, destination: str | Path) -> Path:
    """Variance of MCC explained by dataset identity and by model identity alone."""
    names = decomposition["knowing only"].to_list()
    figure, axes = plt.subplots(figsize=(5.0, 3.6))
    axes.bar(
        range(len(names)), decomposition["variance_explained"].to_numpy(), color=LOO_MODEL, alpha=0.85
    )
    axes.set_xticks(range(len(names)))
    axes.set_xticklabels([_wrap(name) for name in names])
    axes.set_ylabel("variance of MCC explained")
    axes.grid(axis="y", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def term_stability(stability: pl.DataFrame, destination: str | Path, *, top: int = 16) -> Path:
    """How often each term survived the leave-one-dataset-out folds.

    The companion to any published equation. A term chosen in 19 of 20 folds is a finding;
    one chosen in 3 is an artefact of which datasets landed in the training split, and the
    fitted equation alone presents the two identically.
    """
    table = stability.head(top).reverse()
    labels = [_shorten(name) for name in table["term"].to_list()]
    values = table["frequency"].to_numpy()

    figure, axes = plt.subplots(figsize=(8.2, 0.38 * len(labels) + 1.2))
    axes.barh(range(len(labels)), values, color=IN_SAMPLE, alpha=0.85)
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels, fontsize=8)
    axes.set_xlim(0.0, 1.0)
    axes.set_xlabel("fraction of leave-one-dataset-out folds selecting the term")
    axes.grid(axis="x", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def per_group_quality(report: pl.DataFrame, destination: str | Path) -> Path:
    """Rank correlation and top-1 regret for each held-out dataset.

    Averages hide that the equation ranks some datasets almost perfectly and others no
    better than chance. Plotting every fold shows the spread a mean cannot.
    """
    table = report.sort("spearman")
    labels = table["group"].to_list()
    positions = np.arange(len(labels))

    figure, axes = plt.subplots(figsize=(7.4, 0.34 * len(labels) + 1.4))
    axes.barh(positions, table["spearman"].to_numpy(), color=IN_SAMPLE, alpha=0.85, label="Spearman")
    axes.plot(
        table["regret"].to_numpy(), positions, "o", color=NEGATIVE, markersize=5, label="top-1 regret"
    )
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("Spearman correlation / MCC lost by picking the top-ranked model")
    axes.grid(axis="x", alpha=0.25, linestyle=":")
    axes.legend(frameon=False, fontsize=9, loc="lower right")
    return _finish(figure, destination)


def _shorten(name: str, limit: int = 46) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _wrap(label: str, width: int = 18) -> str:
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)
