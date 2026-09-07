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
CEILING = "#6b7280"
POSITIVE = "#00798c"
NEGATIVE = "#d1495b"

FIGURE_DPI = 150


#: Written alongside the raster for every figure. A vector copy is what a paper's
#: typesetting actually wants, and generating it here costs one extra ``savefig``.
VECTOR_SUFFIX = ".pdf"


def _finish(figure: Figure, destination: str | Path) -> Path:
    """Write the figure as PNG and PDF, both on a transparent background.

    Transparent rather than white so a figure sits on whatever the page behind it is,
    without a rectangle of the wrong shade around it. Text and ticks keep matplotlib's
    near-black default, which suits the white or near-white page these are written for;
    on a dark background they would need restyling, and transparency alone would not be
    enough.

    The returned path is the PNG: it is what the markdown chapters embed. The PDF sits
    beside it under the same stem, since a paper's typesetting wants the vector copy.
    """
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True)
    figure.savefig(path.with_suffix(VECTOR_SUFFIX), bbox_inches="tight", transparent=True)
    plt.close(figure)
    return path


def _length_ticks(sizes: np.ndarray, limit: int = 16) -> np.ndarray:
    """Tick positions for a length axis, thinned so the labels stay readable.

    The curve is reported at every length now, which is right for the detector and wrong for
    the axis: 32 labels collide into a grey band. Every nth length is labelled instead, with
    the last one always kept so the axis states its own range.
    """
    if sizes.shape[0] <= limit:
        return sizes
    step = int(np.ceil(sizes.shape[0] / limit))
    kept = list(sizes[::step])
    # Append the final length so the axis states its own range -- unless it would sit on top
    # of the tick before it, which is what produced a "3132" smudge at the right-hand end.
    if sizes[-1] not in kept:
        if kept and sizes[-1] - kept[-1] < step:
            kept[-1] = sizes[-1]
        else:
            kept.append(sizes[-1])
    return np.asarray(kept)


def term_count_curve(
    curve: pl.DataFrame,
    destination: str | Path,
    *,
    oracle: float | None = None,
) -> Path:
    """Accuracy against equation length: the explainability trade.

    The oracle line is the point of the figure. Without it a reader sees a curve still
    climbing and assumes more terms would keep paying, when the whole approach is bounded
    well below 1. It is drawn as a labelled line rather than described in text.

    **It is labelled "additive oracle", not "additive ceiling", and E3 is expected to cross
    it.** `validate.additive_oracle` is the best score reachable by a model that is additive
    in *dataset effect plus model effect* -- perfect group means and nothing else. E3 carries
    mixed terms, each multiplying a dataset feature by a model one, so it represents
    interactions the oracle cannot and rises above it at 16 terms. Calling that line a
    ceiling while a curve sits above it invited the reader to find an error where the
    headline result is: the crossing is what the mixed terms buy.
    """
    figure, axes = plt.subplots(figsize=(7.0, 4.4))
    sizes = curve["n_terms"].to_numpy()

    axes.plot(sizes, curve["r2_in_sample"].to_numpy(), "o-", color=IN_SAMPLE, label="in-sample", linewidth=2)
    if "r2_loo_dataset" in curve.columns:
        axes.plot(sizes, curve["r2_loo_dataset"].to_numpy(), "s--", color=LOO_DATASET, label="leave-one-dataset-out")
    if "r2_loo_model" in curve.columns:
        axes.plot(sizes, curve["r2_loo_model"].to_numpy(), "^:", color=LOO_MODEL, label="leave-one-model-out")
    if oracle is not None:
        axes.axhline(
            oracle,
            color=CEILING,
            linestyle="-.",
            linewidth=1.2,
            label=f"additive oracle ({oracle:.3f})",
        )
        # Headroom above whichever is higher. Pinning the top to the oracle cropped the
        # in-sample curve the moment it crossed -- which is exactly when it matters most.
        highest = max(float(curve["r2_in_sample"].to_numpy().max()), oracle)
        axes.set_ylim(top=highest + 0.05)

    axes.set_xlabel("number of terms")
    axes.set_ylabel("$R^2$")
    axes.set_xticks(_length_ticks(sizes))
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False, loc="lower right", fontsize=9)
    return _finish(figure, destination)


def error_curve(
    curve: pl.DataFrame,
    destination: str | Path,
    *,
    metric: str = "mae",
    marker: int | None = None,
    marker_label: str | None = None,
) -> Path:
    """Error against equation length, in the target's own units.

    R2 answers "how much variance is explained", which is a relative question. MAE answers
    "how far off is a prediction, in MCC", which is the one a practitioner asks. SMAPE is
    included as the scale-free alternative, with the caveat that on a target passing
    through zero it is dominated by the 15 rows at exactly MCC = 0.

    ``curve`` supplies all three protocols, so fit and transfer are read from one figure.

    ``marker`` draws a vertical line at a chosen equation length, and ``marker_label`` says
    what that length *is*. The label used to be hardcoded to "knee", which is how this figure
    came to assert a knee of 4 terms while the study published an equation of 16 -- two
    different quantities, one of them not what the caller was passing. Whoever draws the line
    now has to name it.
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
    if marker is not None:
        name = marker_label or "marked length"
        axes.axvline(marker, color=CEILING, linestyle="-.", linewidth=1.2, label=f"{name} ({marker} terms)")

    axes.set_xlabel("number of terms")
    axes.set_ylabel(label)
    axes.set_xticks(_length_ticks(sizes))
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
    uninformative: around MCC = 0 sit 15 degenerate results and one anti-correlated one,
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
        # Drawn just *below* the axis floor with clipping off, in the neutral grey, so it
        # reads as a marginal distribution against the frame rather than as a row of
        # predictions sitting at MCC = 0. In the data colour and inside the axis it was
        # indistinguishable from the scatter, and this target genuinely has 15 rows at
        # exactly zero -- so a reader had no way to tell the rug from real points.
        span = limits[1] - limits[0]
        # `clip_on=False` frees the rug to sit under the frame, and would equally free the one
        # row at MCC = -0.29 to render as a stray tick out in the left margin. The axis floor
        # is a deliberate choice (`scatter_limits`) and the caption discloses the excluded
        # point, so the rug is restricted to the range it annotates.
        inside = truth[(truth >= limits[0]) & (truth <= limits[1])]
        axes.plot(
            inside,
            np.full_like(inside, limits[0] - 0.012 * span),
            "|",
            color=CEILING,
            alpha=0.5,
            markersize=5,
            clip_on=False,
            zorder=1,
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

    The headline result of the study. Equations are drawn solid and the reference levels
    they are read against are drawn hatched immediately beside them, so the gap each
    equation leaves against its own limit is read directly off the figure instead of
    computed by the reader from a table.

    **"Reference level", not "ceiling", because one of the three is not a ceiling for the
    bar next to it.** The E1 and E2 references are genuine ceilings -- true per-group means
    are the most a predictor constant within that group can achieve. The additive oracle is
    a ceiling only for an equation additive in dataset *and* model effects, and E3 is not
    one: its mixed terms carry interactions, and it scores above the oracle. A legend
    calling every hatched bar a ceiling made the study's headline look like a bug.
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
    axes.legend([solid, hatched], ["fitted equation", "reference level"], frameon=False, fontsize=9)
    return _finish(figure, destination)


def term_effects(effects: pl.DataFrame, destination: str | Path, *, top: int | None = None) -> Path:
    """Per-term effect sizes in MCC units, signed, strongest at the top.

    ``top`` defaults to every term in the equation. It used to default to 12, which silently
    dropped four of the published sixteen -- and the dropped ones were the small-effect terms
    a reader most needs to see, since a term that survived selection while moving the
    prediction barely at all is exactly what the brevity argument is about.

    Term names are wrapped at their operators rather than truncated. Truncation cut names
    mid-feature ("[log(gravity)] / [log(Processing Units Numbe...") which leaves the reader
    unable to tell two terms over the same numerator apart.
    """
    table = (effects if top is None else effects.head(top)).reverse()
    labels = [_wrap_term(name) for name in table["term"].to_list()]
    values = table["effect"].to_numpy() * np.sign(table["beta"].to_numpy())

    figure, axes = plt.subplots(figsize=(8.6, 0.52 * len(labels) + 1.2))
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

    # Only the levels actually present, and drawn in the palette the bars use. The previous
    # legend advertised strong/moderate/weak in grey against red and blue bars, which read as
    # a third series; worse, `unrated` and `moderate` shared an alpha, so on a table of four
    # moderate and two unrated rows every bar rendered identically and the legend described a
    # distinction that was not on the plot.
    present = _confidence_levels(table)
    if present:
        handles = [Rectangle((0, 0), 1, 1, facecolor=POSITIVE, alpha=CONFIDENCE_ALPHA[name]) for name in present]
        axes.legend(handles, present, frameon=False, fontsize=8, loc="lower right", title="confidence")
    return _finish(figure, destination)


#: Bar opacity per rated confidence. ``unrated`` is deliberately the faintest rather than
#: sharing ``moderate``'s value: an unrated practice is one whose term never stabilised, which
#: is weaker evidence than a moderate rating, not equal to it.
CONFIDENCE_ALPHA: dict[str, float] = {"strong": 0.95, "moderate": 0.65, "weak": 0.4, "unrated": 0.25}


def _confidence_levels(table: pl.DataFrame) -> list[str]:
    """The rated levels present in ``table``, strongest first."""
    if "confidence" not in table.columns:
        return []
    seen = {str(value) for value in table["confidence"].to_list()}
    return [name for name in CONFIDENCE_ALPHA if name in seen]


def _confidence_alpha(table: pl.DataFrame) -> list[float]:
    if "confidence" not in table.columns:
        return [0.85] * table.height
    return [CONFIDENCE_ALPHA.get(str(value), 0.25) for value in table["confidence"].to_list()]


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


def per_group_quality(report: pl.DataFrame, destination: str | Path) -> Path:
    """What following the equation's top pick costs, per held-out dataset, in MCC.

    Regret is the practitioner's question stated in the target's own units: take the model
    the equation ranks first, and this is how much MCC that gives up against the dataset's
    genuinely best model. Zero means the pick was right.

    **Spearman is deliberately not drawn here.** It was, and it should not have been, for
    two reasons. It sits between 0.63 and 0.73 on this data for every predictor *and* every
    baseline including a constant, so it separates nothing; and plotting it on the same
    axis as regret put a unitless correlation and an MCC difference on one shared scale,
    where neither could be read. Average precision, reciprocal rank and hit@1 are the
    ranking metrics that do discriminate here, and `ranking_quality` draws those.
    """
    table = report.sort("regret", descending=True)
    labels = table["group"].to_list()
    positions = np.arange(len(labels))
    values = table["regret"].to_numpy()

    figure, axes = plt.subplots(figsize=(7.0, 0.32 * len(labels) + 1.4))
    axes.barh(positions, values, color=[CEILING if v <= 0.0 else NEGATIVE for v in values], alpha=0.85)
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8)
    axes.invert_yaxis()
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("MCC given up by taking the equation's top-ranked model")
    axes.grid(axis="x", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def _shorten(name: str, limit: int = 46) -> str:
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _wrap_term(name: str, width: int = 34) -> str:
    """Break a term name at its operators, so no feature name is ever cut in half.

    Term names are `[a] * [b]`, `[a] / [b]` or `([a] + [b]) / [c]`, so the operators outside
    the brackets are the only safe break points. Splitting on them keeps every bracketed
    feature intact on one line.
    """
    if len(name) <= width:
        return name
    parts = name.replace(" * ", "\x00* ").replace(" / ", "\x00/ ").replace(" + ", "\x00+ ").split("\x00")
    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current} {part}".strip() if current else part
        if len(candidate) > width and current:
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


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


def decision_quality(decision: pl.DataFrame, destination: str | Path) -> Path:
    """Accuracy and F1 of the go/no-go decision against threshold, over the majority baseline.

    Two lines rather than one because the classes are unbalanced at the outer thresholds and
    accuracy alone hides it: always answering with the larger class reaches 0.51 accuracy at a
    threshold of 0.9 and an F1 of exactly zero. The baseline is drawn rather than described,
    so the gap the equation buys is visible instead of asserted.
    """
    figure, axes = plt.subplots(figsize=(6.0, 4.0))
    table = decision.sort("threshold")
    axes.plot(table["threshold"], table["accuracy"], color=IN_SAMPLE, marker="o", label="accuracy")
    axes.plot(
        table["threshold"], table["f1"], color=LOO_MODEL, marker="^", linestyle="--", markerfacecolor="none", label="F1"
    )
    axes.plot(
        table["threshold"], table["majority"], color=CEILING, marker="x", linestyle=":", label="majority-class baseline"
    )
    axes.set_xlabel("MCC threshold")
    axes.set_ylabel("score")
    axes.set_ylim(0.0, 1.0)
    axes.set_xticks(list(table["threshold"]))
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(fontsize=8, loc="lower left", framealpha=0.0)
    return _finish(figure, destination)


def ranking_quality(selection: pl.DataFrame, destination: str | Path) -> Path:
    """Per-dataset head-of-list ranking quality, one bar group per dataset.

    ``hit@1`` is 0 or 1 per dataset, so it is drawn as the marker rather than a bar: what it
    adds is *which* datasets the first pick was right on, which the averages cannot show.
    Sorted by average precision so the hard datasets group at one end.
    """
    table = selection.sort("ap")
    positions = np.arange(table.height)
    figure, axes = plt.subplots(figsize=(7.0, 0.34 * table.height + 1.4))
    axes.barh(positions - 0.19, table["ap"], height=0.36, color=IN_SAMPLE, label="average precision")
    axes.barh(positions + 0.19, table["mrr"], height=0.36, color=LOO_DATASET, label="reciprocal rank")
    # Inside the axis, against the tick label, rather than floating past x = 1 in the margin:
    # a marker drawn beyond the data limit reads as a value on the scale it sits next to.
    hits = [index for index, value in enumerate(table["hit_at_1"]) if value >= 1.0]
    if hits:
        axes.scatter([0.02] * len(hits), hits, marker="*", s=60, color=POSITIVE, label="best model ranked first")
    axes.set_yticks(positions)
    axes.set_yticklabels([_shorten(name, 24) for name in table["group"]], fontsize=7)
    axes.set_xlabel("score")
    axes.set_xlim(0.0, 1.0)
    axes.grid(alpha=0.25, linewidth=0.6, axis="x")
    # Below the axes rather than inside it: the bars are sorted, so whichever corner the
    # legend takes it covers either the best or the worst datasets.
    axes.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, framealpha=0.0)
    return _finish(figure, destination)
