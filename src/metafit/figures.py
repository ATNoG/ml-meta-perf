"""Generate the study's figure set from a finished report.

Figures carry no titles: each is written for a LaTeX ``figure`` environment where the
caption supplies the description. ``CAPTIONS`` holds a suggested caption per file,
including the disclosures that would otherwise have to be drawn into the image.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from metafit.data import DATASET_FEATURES, MODEL_FEATURES, columns_as_arrays, load, target
from metafit.experiment import Report
from metafit.plots import (
    contribution_shares,
    count_below_floor,
    equation_comparison,
    error_curve,
    oracle_ladder,
    per_group_quality,
    practice_effects,
    predicted_versus_actual,
    term_count_curve,
    term_effects,
    term_stability,
)

ORACLE_ROW = "additive oracle (ceiling)"


def _oracle(report: Report) -> float | None:
    """The additive ceiling as this run computed it.

    Read from the report rather than hardcoded: the ceiling is a property of the data,
    so a literal here would silently drift out of step with the comparison table the
    moment the meta-dataset changed.
    """
    matched = report.comparison.filter(pl.col("equation") == ORACLE_ROW)
    return float(matched["r2"][0]) if matched.height else None


def _knee(report: Report) -> int | None:
    """The knee of the in-sample curve, as the run's own selection table reports it."""
    matched = report.term_choice.filter(pl.col("rule") == "knee (in-sample)")
    return int(matched["n_terms"][0]) if matched.height else None


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
        ),
        predicted_versus_actual(truth, predicted, folder / "predicted_vs_actual.png", groups=truth),
        error_curve(
            report.e3.curve,
            folder / "error_curve_mae.png",
            metric="mae",
            marker=_knee(report),
        ),
        term_effects(report.effects, folder / "term_effects.png"),
        practice_effects(report.practices, folder / "practice_effects.png"),
        contribution_shares(report.shares, folder / "contribution_shares.png"),
        oracle_ladder(
            report.oracles,
            folder / "oracle_ladder.png",
            achieved=float(report.e3.in_sample["r2"]),
        ),
        per_group_quality(report.selection, folder / "per_group_quality.png"),
    ]
    if report.e3.stability is not None:
        written.append(term_stability(report.e3.stability, folder / "term_stability.png"))
    return written


def captions(report: Report, data: str | Path | None = None) -> dict[str, str]:
    """Suggested LaTeX captions, carrying the disclosures kept out of the images."""
    frame = load(data)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    truth = target(frame)
    hidden = count_below_floor(truth, report.e3.equation.predict(columns))

    return {
        "equation_comparison.png": (
            "Each fitted equation against the ceiling that bounds it, all scored on the "
            "same 476 rows. Dataset-only and model-only equations are bounded by what "
            "their group identity can explain; any additive equation is bounded by the "
            "oracle."
        ),
        "term_count_curve.png": (
            "Accuracy against equation length for E3, in-sample and under both "
            "cross-validation protocols. The additive ceiling bounds every curve shown."
        ),
        "predicted_vs_actual.png": (
            "Predicted against actual MCC for E3, with the rug showing the marginal "
            f"distribution of the target. Axes start at 0; {hidden} point below that is "
            "not shown. Predictions never fall below 0.17 while 38 rows sit at exactly 0."
        ),
        "error_curve_mae.png": (
            "Mean absolute error in MCC against equation length, under both protocols. "
            "The vertical line marks the knee of the in-sample curve."
        ),
        "term_effects.png": (
            "Per-term effect on predicted MCC, measured as the swing between the term's "
            "10th and 90th percentile. Sign follows the fitted weight."
        ),
        "practice_effects.png": (
            "Per-feature effect on predicted MCC between the feature's lowest and highest "
            "decile, shaded by the confidence its practice was rated at."
        ),
        "contribution_shares.png": (
            "Share of E3's output variance driven by terms using dataset features only, "
            "model features only, and both. Shares are covariance-based and sum to 1."
        ),
        "oracle_ladder.png": (
            "Ceiling as interaction components are added to the additive oracle (AMMI). "
            "Rank 0 is the additive oracle; the first component alone is worth +0.12 R2, "
            "which the fitted equation does not reach."
        ),
        "per_group_quality.png": (
            "Rank correlation and top-1 regret for each held-out dataset under "
            "leave-one-dataset-out validation."
        ),
        "term_stability.png": (
            "Fraction of the 20 leave-one-dataset-out folds that selected each term. "
            "Terms selected in few folds are artefacts of the training split."
        ),
    }
