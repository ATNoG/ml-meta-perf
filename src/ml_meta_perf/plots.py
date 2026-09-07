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


#: Model features abbreviated in rendered term labels. The full names are five syllables long
#: and a figure of fifteen terms cannot carry them; the glossary in chapter 1 expands them.
TERM_ABBREVIATIONS: dict[str, str] = {
    "Processing Units Number": "PUN",
    "Model Capability": "MC",
    "Input Distribution Modelling": "IDM",
    "Fitting Regime": "FR",
    "Solution Stochasticity": "SS",
    "Loss Margin Behaviour": "LMB",
}


def _math_atom(text: str) -> str:
    """One operand of a term, as mathtext."""
    text = text.strip()
    for pattern, wrap in ((r"log(", r"\log\,{}"), (r"sqrt(", r"\sqrt{{{}}}")):
        if text.startswith(pattern) and text.endswith(")"):
            return wrap.format(_math_atom(text[len(pattern) : -1]))
    if text.endswith("^2"):
        return _math_atom(text[:-2]) + "^{2}"
    if text.startswith("1/"):
        return r"\frac{1}{" + _math_atom(text[2:]) + "}"
    if all(character.isdigit() or character == "." for character in text):
        return text
    name = TERM_ABBREVIATIONS.get(text, text)
    return r"\mathrm{" + name.replace("_", r"\_").replace(" ", r"\ ") + "}"


def _math_operand(text: str) -> str:
    text = text.strip()
    return _math_atom(text[1:-1] if text.startswith("[") and text.endswith("]") else text)


def _split_top(text: str, operator: str) -> tuple[str, str] | None:
    """Split on ``operator`` at bracket depth zero, so operators inside `[...]` are ignored."""
    depth = 0
    for index, character in enumerate(text):
        if character in "[(":
            depth += 1
        elif character in "])":
            depth -= 1
        elif depth == 0 and text.startswith(operator, index):
            return text[:index], text[index + len(operator) :]
    return None


def term_to_math(name: str) -> str:
    """A term name as a mathtext expression: ratios as fractions, products as centre dots.

    Term names are written for a CSV -- ``[log(gravity)] / [log(Processing Units Number)]`` --
    and a figure of fifteen of those is a wall of brackets. Rendered as mathematics the same
    term is one fraction, which is how it would appear in the paper the equation is for.
    """
    text = name.strip()
    ratio = _split_top(text, " / ")
    if ratio is not None:
        numerator, denominator = ratio
        inner = numerator.strip()
        if inner.startswith("(") and inner.endswith(")"):
            summed = _split_top(inner[1:-1], " + ")
            if summed is not None:
                left, right = summed
                numerator_math = _math_operand(left) + " + " + _math_operand(right)
                return r"$\frac{" + numerator_math + "}{" + _math_operand(denominator) + "}$"
        return r"$\frac{" + _math_operand(numerator) + "}{" + _math_operand(denominator) + "}$"
    product = _split_top(text, " * ")
    if product is not None:
        left, right = product
        return "$" + _math_operand(left) + r" \cdot " + _math_operand(right) + "$"
    return "$" + _math_operand(text) + "$"


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
    marker: int | None = None,
    marker_label: str | None = None,
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

    if marker is not None:
        # This figure is *about* choosing a length, so the chosen one belongs on it. Named by
        # the caller for the same reason `error_curve`'s is: a line whose label is fixed in
        # the plotting code cannot be kept in step with what is actually being passed.
        name = marker_label or "marked length"
        axes.axvline(marker, color=CEILING, linestyle=":", linewidth=1.4, label=f"{name} ({marker} terms)")

    axes.set_xlabel("number of terms")
    axes.set_ylabel("$R^2$")
    axes.set_xticks(_length_ticks(sizes))
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

    ``groups`` is accepted and unused. A rug of the target's marginal distribution was drawn
    along the bottom axis and removed: on an axis that starts at zero it reads as a row of
    predictions at MCC = 0, which is exactly the region this target genuinely occupies, and
    no caption reliably undoes that.
    """
    limits = scatter_limits(truth, predicted, margin, floor)

    figure, axes = plt.subplots(figsize=(5.6, 5.4))
    axes.plot(limits, limits, color=CEILING, linewidth=1.0, linestyle="--", label="perfect", zorder=1)
    axes.scatter(truth, predicted, s=18, alpha=0.55, color=IN_SAMPLE, edgecolor="none", zorder=2)
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

    **The capability equation is deliberately absent.** It appears in one table in chapter 4
    and nowhere else, because a bar chart of headline equations is exactly the place a reader
    would take it for a second recommendation. Its purpose is to answer one question about how
    far the form reaches, and a figure cannot carry that qualification.

    **"Reference level", not "ceiling", because one of the three is not a ceiling for the
    bar next to it.** The E1 and E2 references are genuine ceilings -- true per-group means
    are the most a predictor constant within that group can achieve. The additive oracle is
    a ceiling only for an equation additive in dataset *and* model effects, and E3 is not
    one: its mixed terms carry interactions, and it scores above the oracle. A legend
    calling every hatched bar a ceiling made the study's headline look like a bug.
    """
    table = comparison.filter(~pl.col("equation").str.contains("capability"))
    labels = table["equation"].to_list()
    values = table["r2"].to_numpy()
    is_ceiling = [("reference" in label) or ("oracle" in label) for label in labels]

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

    Terms are rendered as mathematics rather than as their CSV names -- a ratio becomes a
    fraction, a product a centre dot -- because a figure of fifteen bracketed strings is a
    wall of punctuation, and the equation is written for a paper. `term_to_math` does it and
    `TERM_ABBREVIATIONS` shortens the five-syllable model features; chapter 1 expands them.
    """
    table = (effects if top is None else effects.head(top)).reverse()
    labels = [term_to_math(name) for name in table["term"].to_list()]
    values = table["effect"].to_numpy() * np.sign(table["beta"].to_numpy())

    figure, axes = plt.subplots(figsize=(8.0, 0.46 * len(labels) + 1.2))
    axes.barh(
        range(len(labels)),
        values,
        color=[POSITIVE if value > 0 else NEGATIVE for value in values],
        alpha=0.85,
    )
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels, fontsize=11)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("effect on predicted MCC (10th to 90th percentile swing)")
    axes.grid(axis="x", alpha=0.25, linestyle=":")
    return _finish(figure, destination)


def practice_effects(practices: pl.DataFrame, destination: str | Path) -> Path:
    """What the fitted equation says each raw feature does to MCC, and how stable that is.

    One bar per feature the equation uses. Its length is the change in **predicted MCC**
    between that feature's lowest and highest decile, holding the rest of the equation --
    so a bar reaching -0.6 means the equation predicts 0.6 less MCC at the top of that
    feature's range than at the bottom. Sign is measured on the data rather than read off a
    weight, because a feature can appear in several terms and inside denominators, and then
    it has no single weight to read.

    Opacity is the confidence its written practice was rated at, which is driven by how often
    the feature's terms survived reselection across folds. A long bar at low opacity is a
    large effect the folds disagreed about; both facts are needed and neither is the other.

    This is the evidence layer of chapter 6, and it is not itself advice: a statement about a
    meta-feature column becomes a practice only when it supports or contradicts something a
    practitioner could already have been told.
    """
    table = practices.reverse()
    labels = [TERM_ABBREVIATIONS.get(str(name), str(name)) for name in table["feature"].to_list()]
    values = table["effect"].to_numpy()
    alphas = _confidence_alpha(table)

    figure, axes = plt.subplots(figsize=(7.8, 0.44 * len(labels) + 1.4))
    for index, (value, alpha) in enumerate(zip(values, alphas, strict=True)):
        axes.barh(index, value, color=POSITIVE if value > 0 else NEGATIVE, alpha=alpha)
    axes.set_yticks(range(len(labels)))
    axes.set_yticklabels(labels, fontsize=9)
    axes.axvline(0.0, color="black", linewidth=0.8)
    axes.set_xlabel("change in predicted MCC, lowest decile of the feature to its highest")
    axes.grid(axis="x", alpha=0.25, linestyle=":")

    # Only the levels actually present, and drawn in the palette the bars use. A legend in a
    # third colour reads as a third series, and one advertising levels the table does not
    # contain describes a distinction that is not on the plot.
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


def decision_quality(decision: pl.DataFrame, destination: str | Path) -> Path:
    """The go/no-go decision against threshold, as **F1**, one line per protocol.

    One metric, four lines, and what differs between them is only how much the equation was
    allowed to see. Previously this drew the equation's accuracy, the equation's F1 and a
    majority-class baseline on one axis -- two metrics and a predictor, so no reader could
    tell which metric the baseline was being scored on.

    F1 rather than accuracy: the classes are unbalanced at the outer thresholds, where always
    answering with the larger one reaches 0.51 accuracy at an F1 of exactly zero. Accuracy
    would draw four flattering lines and hide that. The trivial predictors stay in the table
    rather than on the plot, because they cannot be computed under the strictest protocol at
    all -- a model held out of every fold has no rows to average.

    The four lines are the point: the gap between the top and the bottom one is the whole cost
    of generalisation on this task, and on this corpus it is small.
    """
    figure, axes = plt.subplots(figsize=(6.8, 4.2))
    table = decision.sort("threshold")

    order = ["in-sample", "loo-dataset", "loo-model", "loo-cell"]
    styles = {
        "in-sample": (IN_SAMPLE, "o-", 1.8),
        "loo-dataset": (LOO_DATASET, "s--", 1.8),
        "loo-model": (LOO_MODEL, "^:", 1.8),
        "loo-cell": ("#2f2f2f", "D-", 2.2),
    }

    if "predictor" in table.columns:
        names = [str(name) for name in table["predictor"].unique(maintain_order=True)]
        equations = [name for name in names if "equation" in name.lower()]
        equations.sort(key=lambda name: next((i for i, k in enumerate(order) if k in name), len(order)))
        for name in equations:
            rows = table.filter(pl.col("predictor") == name).sort("threshold")
            key = next((k for k in order if k in name), "in-sample")
            colour, marker, width = styles[key]
            label = key + (" (both held out)" if key == "loo-cell" else "")
            axes.plot(rows["threshold"], rows["f1"], marker, color=colour, label=label, linewidth=width)
    else:
        axes.plot(table["threshold"], table["f1"], "o-", color=IN_SAMPLE, label="equation", linewidth=2)

    axes.set_xlabel("MCC threshold for the go/no-go decision")
    axes.set_ylabel("F1 of the decision")
    axes.set_xticks(sorted({float(value) for value in table["threshold"]}))
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(fontsize=9, loc="lower left", framealpha=0.0, title="what the equation was shown")
    return _finish(figure, destination)


def ranking_quality(selection: pl.DataFrame, destination: str | Path) -> Path:
    """Per-dataset ranking quality beside what a bad ranking actually *costs*, in MCC.

    Ranking metrics alone raise a question they cannot answer. Four datasets here score
    average precision below 0.85 and two below 0.5, which reads as serious failure -- until
    one asks what following the bad ranking costs, and the answer is at most 0.13 MCC and on
    most of them nothing at all. The two belong together: a dataset whose models all score
    within a hair of each other is *unrankable*, and ranking it badly is not a cost.

    Two panels sharing the dataset axis rather than two scales on one axis. Regret is in MCC
    and the ranking metrics are unitless, so overlaying them would repeat the mistake this
    figure exists to avoid -- a reader cannot tell which axis a marker belongs to. Sorted by
    average precision, so the hard datasets group at the bottom and the cost of each sits
    directly beside it.
    """
    table = selection.sort("ap", descending=True)
    positions = np.arange(table.height)

    figure, (axes, cost) = plt.subplots(
        1, 2, figsize=(9.0, 0.34 * table.height + 1.8), sharey=True,
        gridspec_kw={"width_ratios": [2.1, 1.0], "wspace": 0.10},
    )

    axes.barh(positions - 0.19, table["ap"], height=0.36, color=IN_SAMPLE, label="average precision")
    axes.barh(positions + 0.19, table["mrr"], height=0.36, color=LOO_MODEL, label="reciprocal rank")
    hits = [index for index, value in enumerate(table["hit_at_1"]) if value >= 1.0]
    if hits:
        axes.scatter([0.03] * len(hits), hits, marker="*", s=70, color="white",
                     edgecolor="#333333", linewidth=0.7, zorder=5, label="best model ranked first")
    axes.set_yticks(positions)
    axes.set_yticklabels([_shorten(name, 24) for name in table["group"]], fontsize=8)
    axes.set_xlim(0.0, 1.0)
    axes.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    axes.set_xlabel("average precision / reciprocal rank")
    axes.invert_yaxis()
    axes.grid(alpha=0.25, linewidth=0.6, axis="x")
    axes.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, framealpha=0.0)

    regret = table["regret"].to_numpy()
    cost.barh(positions, regret, height=0.55, color=[CEILING if v <= 1e-9 else NEGATIVE for v in regret])
    cost.set_xlabel("MCC given up by taking\nthe top-ranked model")
    cost.grid(alpha=0.25, linewidth=0.6, axis="x")
    cost.tick_params(labelleft=False)
    return _finish(figure, destination)
