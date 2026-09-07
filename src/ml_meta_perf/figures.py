"""Generate the study's figure set from a finished report.

Figures carry no titles: each is written for a LaTeX ``figure`` environment where the
caption supplies the description. ``CAPTIONS`` holds a suggested caption per file,
including the disclosures that would otherwise have to be drawn into the image.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ml_meta_perf.data import DATASET_FEATURES, MODEL_FEATURES, columns_as_arrays, load, target
from ml_meta_perf.experiment import Report
from ml_meta_perf.plots import (
    count_below_floor,
    decision_quality,
    equation_comparison,
    practice_effects,
    predicted_versus_actual,
    ranking_quality,
    term_count_curve,
    term_effects,
)

ORACLE_ROW = "reference: additive oracle"

#: The figure set in the order the study presents it. **The order is the identity**: files
#: are written as ``01_equation_comparison.png`` and so on, so a figure can be named by its
#: number in a review comment or a caption without anyone having to agree on which
#: "comparison" plot is meant. Position in this tuple is the only place a number is written
#: down -- `figure_name` derives it -- so re-ordering the set renumbers the files and the
#: captions together and cannot leave the two disagreeing.
FIGURE_ORDER: tuple[str, ...] = (
    "equation_comparison",
    "term_count_curve",
    "predicted_vs_actual",
    "term_effects",
    "practice_effects",
    "ranking_quality",
    "decision_quality",
)


def figure_name(stem: str) -> str:
    """The published filename for a figure, ``NN_stem.png``, numbered from `FIGURE_ORDER`."""
    return f"{FIGURE_ORDER.index(stem) + 1:02d}_{stem}.png"


def _oracle(report: Report) -> float | None:
    """The additive ceiling as this run computed it.

    Read from the report rather than hardcoded: the ceiling is a property of the data,
    so a literal here would silently drift out of step with the comparison table the
    moment the meta-dataset changed.
    """
    matched = report.comparison.filter(pl.col("equation") == ORACLE_ROW)
    return float(matched["r2"][0]) if matched.height else None


def _published_length(report: Report) -> int | None:
    """How many terms the equation being reported actually has.

    This figure used to mark `term_choice`'s in-sample knee instead, which on the current
    configuration is 4 while the published equation has 16 -- so the line labelled "knee"
    stood four fifths of the way from the equation it was drawn beside. Marking the length
    the rest of the report is about cannot go out of step with it.

    The knee itself is still reported, in `term_choice`, and it is under review: it is
    detected on in-sample R2 alone and on a non-uniform grid that happens to skip every
    length where the transfer curve craters. See `TODO.md`, item 2.
    """
    return len(report.e3.equation.terms) or None


def generate(report: Report, destination: str | Path, data: str | Path | None = None) -> list[Path]:
    """Write every figure and return the paths, in the order they appear in the README.

    ``data`` must be the meta-dataset the report was fitted on. Defaulting it to the
    packaged one is only safe because that is the usual case; a caller who fitted on
    another file and does not pass it here would get figures drawn from the wrong rows.
    """
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    frame = load(data)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    predicted = report.e3.equation.predict(columns)

    written = [
        equation_comparison(report.comparison, folder / figure_name("equation_comparison")),
        term_count_curve(
            report.e3.curve,
            folder / figure_name("term_count_curve"),
            oracle=_oracle(report),
            marker=_published_length(report),
            marker_label="selected term count",
        ),
        predicted_versus_actual(truth, predicted, folder / figure_name("predicted_vs_actual")),
        term_effects(report.effects, folder / figure_name("term_effects")),
        practice_effects(report.practices, folder / figure_name("practice_effects")),
        ranking_quality(report.selection, folder / figure_name("ranking_quality")),
        decision_quality(report.decision_baselines, folder / figure_name("decision_quality")),
    ]
    # The list is what the README and the chapters index against, so it has to come back in
    # `FIGURE_ORDER`. Asserting it here means a call added out of order fails the run rather
    # than shipping a figure numbered one thing and referenced as another.
    expected = [folder / figure_name(stem) for stem in FIGURE_ORDER]
    if written != expected:
        raise AssertionError(f"figures written out of FIGURE_ORDER: {[path.name for path in written]}")
    return written


def captions(report: Report, data: str | Path | None = None) -> dict[str, str]:
    """Suggested LaTeX captions, carrying the disclosures kept out of the images."""
    frame = load(data)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    hidden = count_below_floor(truth, report.e3.equation.predict(columns))

    return {
        figure_name("equation_comparison"): (
            "Each fitted equation against the level it is read against, all scored on the "
            f"same {frame.height} rows. Dataset-only and model-only equations are bounded by what "
            "their group identity can explain. The additive oracle bounds only an equation "
            "additive in dataset and model effects, which E3 is not: its mixed terms carry "
            "interactions, and it scores above the oracle. All bars are in-sample."
        ),
        figure_name("term_count_curve"): (
            "Accuracy against equation length for E3, in-sample and under both "
            "cross-validation protocols. The vertical line is the length chosen by "
            "`selection.best_length`, the argmax of the consensus over the three protocols. "
            "The additive oracle is the best score reachable by an equation additive in "
            "dataset and model effects; an equation passes it only by representing the "
            "dataset-by-model interaction the oracle cannot."
        ),
        figure_name("predicted_vs_actual"): (
            "Predicted against actual MCC for E3. Axes start at 0; "
            f"{hidden} point below that is not shown. The equation compresses toward the "
            "middle of the range, as a shrunk linear fit will."
        ),
        figure_name("term_effects"): (
            "Per-term effect on predicted MCC, measured as the swing between the term's "
            "10th and 90th percentile. Sign follows the fitted weight."
        ),
        figure_name("practice_effects"): (
            "Per-feature effect on predicted MCC between the feature's lowest and highest "
            "decile, shaded by the confidence its practice was rated at. Only the confidence "
            "levels present in the table appear in the legend."
        ),
        figure_name("decision_quality"): (
            "F1 of the above-or-below-threshold decision against the threshold, one line per "
            "protocol. The four differ only in what the equation was allowed to see, so the "
            "spread between them is the cost of generalisation on this task. F1 rather than "
            "accuracy because the classes are unbalanced at the outer thresholds, where "
            "always answering with the larger class reaches 0.51 accuracy at an F1 of zero."
        ),
        figure_name("ranking_quality"): (
            "Head-of-list ranking quality per held-out dataset, beside what a bad ranking "
            "costs. Left: average precision and reciprocal rank, with a star where the "
            "equation ranked the best model first; relevance is being within 0.01 MCC of the "
            "dataset's best. Right: the MCC given up by taking the top-ranked model. Both are "
            "scored with the dataset and the model of every cell held out of the fit."
        ),
    }
