"""Figures for the study.

matplotlib is imported through the Agg backend so these run headless, in CI and over
ssh, without a display. Every function takes a destination path, writes one figure, and
returns that path, so the caller decides where output lands and nothing is written as a
side effect of importing.

The palette is colour-blind safe and each series is also distinguished by marker or line
style, so the figures survive being printed in greyscale -- which, for a paper, they will
be.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.figure import Figure

from metafit.model import Equation

IN_SAMPLE = "#1b6ca8"
LOO_DATASET = "#d1495b"
LOO_MODEL = "#00798c"
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
    title: str = "Accuracy versus equation length",
) -> Path:
    """The accuracy-versus-explainability trade, with the additive ceiling drawn on.

    The ceiling is the point of the figure: without it a reader sees a curve still
    climbing and assumes more terms would keep paying, when in fact the whole approach
    is bounded well below 1.
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
    if oracle is not None:
        axes.axhline(oracle, color=CEILING, linestyle="-.", linewidth=1.2)
        # Below the line, not above: above it collides with the axis frame once the
        # y-limit is only a little past the ceiling.
        axes.text(
            sizes.max(), oracle - 0.014, f"additive ceiling {oracle:.3f}",
            ha="right", va="top", fontsize=8, color=CEILING,
        )
        axes.set_ylim(top=oracle + 0.05)

    axes.set_xlabel("number of terms")
    axes.set_ylabel("$R^2$")
    axes.set_title(title)
    axes.set_xticks(sizes)
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False, loc="lower right", fontsize=9)
    return _finish(figure, destination)


def scatter_limits(
    truth: np.ndarray,
    predicted: np.ndarray,
    margin: float = 0.05,
    floor: float = 0.0,
) -> tuple[float, float]:
    """Square axis bounds covering both series, padded by ``margin`` and stopping at ``floor``.

    Deliberately not MCC's theoretical [-1, 1]. Two separate reasons.

    Nothing in this meta-dataset approaches -1 -- the observed minimum is -0.29, from a
    single row -- so a [-1, 1] axis would spend its bottom half on an empty region.

    The interval below zero is worse than empty, it is uninformative. Around MCC = 0 sit
    38 degenerate results and one anti-correlated one, and they carry no linear structure
    for the diagonal to be read against; including them compresses the range where the
    relationship actually lives. ``floor`` therefore defaults to 0, and points beneath it
    fall outside the axes -- which the caller is expected to disclose rather than hide.
    """
    lowest = max(float(min(truth.min(), predicted.min())) - margin, floor)
    highest = float(max(truth.max(), predicted.max())) + margin
    return lowest, highest


def count_below_floor(
    truth: np.ndarray,
    predicted: np.ndarray,
    floor: float = 0.0,
) -> int:
    """How many points ``predicted_versus_actual`` leaves outside its axes.

    Kept separate from the figure so the disclosure can go in a LaTeX caption rather than
    being rendered into the image, where it cannot be restyled or translated.
    """
    return int(np.sum((truth < floor) | (predicted < floor)))


def predicted_versus_actual(
    truth: np.ndarray,
    predicted: np.ndarray,
    destination: str | Path,
    *,
    title: str = "Predicted versus actual MCC",
    margin: float = 0.05,
    floor: float = 0.0,
) -> Path:
    """A scatter against the diagonal, with the MCC ceiling made visible.

    17% of the meta-dataset sits at exactly MCC = 1.0. That pile-up is the single most
    important thing to see about this target, and a scatter shows it in a way no summary
    statistic does.

    The axes start at ``floor`` (0 by default) rather than at -1: the sub-zero region
    holds only degenerate results with no linear structure, and including it squashes the
    range where the relationship lives.

    Points below the floor are not drawn, and the figure carries no note saying so --
    that belongs in the caption, not burned into the image. ``count_below_floor`` returns
    the number for whoever writes it.
    """
    limits = scatter_limits(truth, predicted, margin, floor)

    figure, axes = plt.subplots(figsize=(5.4, 5.2))
    axes.plot(limits, limits, color=CEILING, linewidth=1.0, linestyle="--", label="perfect", zorder=1)
    axes.scatter(truth, predicted, s=18, alpha=0.55, color=IN_SAMPLE, edgecolor="none", zorder=2)
    axes.set_xlim(limits)
    axes.set_ylim(limits)
    axes.set_aspect("equal")
    axes.set_xlabel("actual MCC")
    axes.set_ylabel("predicted MCC")
    axes.set_title(title)
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False, loc="upper left", fontsize=9)
    return _finish(figure, destination)


def term_effects(effects: pl.DataFrame, destination: str | Path, *, top: int = 12) -> Path:
    """Per-term effect sizes in MCC units, signed, strongest at the top."""
    table = effects.head(top).reverse()
    labels = [_shorten(name) for name in table["term"].to_list()]
    values = table["effect"].to_numpy() * np.sign(table["beta"].to_numpy())

    figure, axes = plt.subplots(figsize=(8.2, 0.42 * len(labels) + 1.4))
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
    axes.set_title("What each term is worth")
    axes.grid(axis="x", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def practice_effects(practices: pl.DataFrame, destination: str | Path) -> Path:
    """Per-feature effects, the form the written practices are derived from."""
    table = practices.reverse()
    labels = table["feature"].to_list()
    values = table["effect"].to_numpy()

    figure, axes = plt.subplots(figsize=(7.6, 0.42 * len(labels) + 1.4))
    axes.barh(
        range(len(labels)),
        values,
        color=[POSITIVE if value > 0 else NEGATIVE for value in values],
        alpha=0.85,
    )
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels, fontsize=9)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("MCC change from the feature's lowest decile to its highest")
    axes.set_title("Feature effects behind the extracted practices")
    axes.grid(axis="x", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def protocol_comparison(leakage: pl.DataFrame, destination: str | Path) -> Path:
    """The same equation under three splits -- the leakage figure."""
    labels = leakage["protocol"].to_list()
    values = leakage["r2"].to_numpy()
    colors = [LOO_DATASET if "random" in label else IN_SAMPLE for label in labels]

    figure, axes = plt.subplots(figsize=(6.6, 3.6))
    bars = axes.bar(range(len(labels)), values, color=colors, alpha=0.85)
    for bar, value in zip(bars, values, strict=True):
        axes.text(
            bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels([label.replace(" (leaky)", "\n(leaky)") for label in labels], fontsize=9)
    axes.set_ylabel("$R^2$")
    axes.set_title("Same equation, three validation protocols")
    axes.grid(axis="y", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def contribution_sources(shares: pl.DataFrame, decomposition: pl.DataFrame, destination: str | Path) -> Path:
    """Where the signal comes from: the equation's terms, and the target's own structure."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.8))

    groups = shares["group"].to_list()
    left.bar(range(len(groups)), shares["share"].to_numpy(), color=IN_SAMPLE, alpha=0.85)
    left.set_xticks(range(len(groups)))
    left.set_xticklabels(groups)
    left.set_ylabel("share of the equation's output variance")
    left.set_title("Which terms drive the equation")
    left.axhline(0.0, color="black", linewidth=0.8)
    left.grid(axis="y", alpha=0.25, linestyle=":")

    names = decomposition["knowing only"].to_list()
    right.bar(range(len(names)), decomposition["variance_explained"].to_numpy(), color=LOO_MODEL, alpha=0.85)
    right.set_xticks(range(len(names)))
    right.set_xticklabels([name.replace(" ", "\n") for name in names])
    right.set_ylabel("variance of MCC explained")
    right.set_title("Ceiling from identity alone")
    right.grid(axis="y", alpha=0.25, linestyle=":")

    return _finish(figure, destination)


def _shorten(name: str, limit: int = 46) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def equation_summary(equation: Equation, destination: str | Path) -> Path:
    """The equation itself, rendered as a figure for slides."""
    lines = [f"MCC = {equation.intercept:+.4g}"]
    for term, weight, _ in equation.ranked_terms():
        lines.append(f"   {weight:+.4g} · {_shorten(term.name, 54)}")

    figure, axes = plt.subplots(figsize=(9.0, 0.32 * len(lines) + 0.9))
    axes.axis("off")
    axes.text(
        0.01, 0.98, "\n".join(lines), va="top", ha="left",
        family="monospace", fontsize=9.5, transform=axes.transAxes,
    )
    return _finish(figure, destination)
