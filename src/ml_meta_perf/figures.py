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
        equation_comparison(report.comparison, folder / "equation_comparison.png"),
        term_count_curve(
            report.e3.curve,
            folder / "term_count_curve.png",
            oracle=_oracle(report),
            marker=_published_length(report),
            marker_label="selected term count",
        ),
        predicted_versus_actual(truth, predicted, folder / "predicted_vs_actual.png"),
        term_effects(report.effects, folder / "term_effects.png"),
        practice_effects(report.practices, folder / "practice_effects.png"),
        ranking_quality(report.selection, folder / "ranking_quality.png"),
        decision_quality(report.decision_baselines, folder / "decision_quality.png"),
    ]
    return written


def captions(report: Report, data: str | Path | None = None) -> dict[str, str]:
    """Suggested LaTeX captions, carrying the disclosures kept out of the images."""
    frame = load(data)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    hidden = count_below_floor(truth, report.e3.equation.predict(columns))

    return {
        "equation_comparison.png": (
            "Each fitted equation against the level it is read against, all scored on the "
            "same 476 rows. Dataset-only and model-only equations are bounded by what "
            "their group identity can explain. The additive oracle bounds only an equation "
            "additive in dataset and model effects, which E3 is not: its mixed terms carry "
            "interactions, and it scores above the oracle. All bars are in-sample."
        ),
        "term_count_curve.png": (
            "Accuracy against equation length for E3, in-sample and under both "
            "cross-validation protocols. The vertical line is the length chosen by "
            "`selection.best_length`, the argmax of the consensus over the three protocols. "
            "The additive oracle is the best score reachable by an equation additive in "
            "dataset and model effects; an equation passes it only by representing the "
            "dataset-by-model interaction the oracle cannot."
        ),
        "predicted_vs_actual.png": (
            "Predicted against actual MCC for E3, with the rug showing the marginal "
            f"distribution of the target. Axes start at 0; {hidden} point below that is "
            "not shown. Predictions never fall below 0.17 while 15 rows sit at exactly 0."
        ),
        "error_curve_mae.png": (
            "Mean absolute error in MCC against equation length, under both protocols. "
            "The vertical line marks the length of the published equation."
        ),
        "term_effects.png": (
            "Per-term effect on predicted MCC, measured as the swing between the term's "
            "10th and 90th percentile. Sign follows the fitted weight."
        ),
        "practice_effects.png": (
            "Per-feature effect on predicted MCC between the feature's lowest and highest "
            "decile, shaded by the confidence its practice was rated at. Only the confidence "
            "levels present in the table appear in the legend."
        ),
        "contribution_shares.png": (
            "Share of E3's output variance driven by terms using dataset features only, "
            "model features only, and both. Shares are covariance-based and sum to 1."
        ),
        "per_group_quality.png": (
            "MCC given up on each held-out dataset by taking the model the equation ranks "
            "first, under leave-one-dataset-out validation. Zero means the top pick was the "
            "dataset's best model, to within the 0.01 MCC relevance tolerance."
        ),
        "decision_quality.png": (
            "Accuracy and F1 of the above-or-below-threshold decision, against the threshold, "
            "with the majority-class baseline any such rule has to clear. Both curves are "
            "shown because the classes are unbalanced at the outer thresholds, where the "
            "baseline reaches high accuracy at an F1 of zero."
        ),
        "ranking_quality.png": (
            "Head-of-list ranking quality for each held-out dataset: average precision and "
            "mean reciprocal rank over the models, with a star where the equation ranked the "
            "best model first. Relevance is being within 0.01 MCC of the dataset's best."
        ),
    }
