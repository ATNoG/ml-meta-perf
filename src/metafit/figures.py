"""Generate the study's figure set from a finished report."""

from __future__ import annotations

from pathlib import Path

from metafit.data import DATASET_FEATURES, MODEL_FEATURES, columns_as_arrays, load, target
from metafit.experiment import Report
from metafit.plots import (
    contribution_sources,
    equation_summary,
    practice_effects,
    predicted_versus_actual,
    protocol_comparison,
    term_count_curve,
    term_effects,
)

ADDITIVE_ORACLE = 0.661


def generate(report: Report, destination: str | Path) -> list[Path]:
    """Write every figure and return the paths, in the order they appear in the README."""
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    frame = load()
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)

    return [
        term_count_curve(
            report.e2.curve,
            folder / "term_count_curve.png",
            oracle=ADDITIVE_ORACLE,
            title="E2: accuracy versus equation length",
        ),
        term_count_curve(
            report.e1.curve,
            folder / "term_count_curve_e1.png",
            title="E1: accuracy versus equation length (20 dataset means)",
        ),
        predicted_versus_actual(
            target(frame),
            report.e2.equation.predict(columns),
            folder / "predicted_vs_actual.png",
            title="E2: predicted versus actual MCC",
        ),
        term_effects(report.effects, folder / "term_effects.png"),
        practice_effects(report.practices, folder / "practice_effects.png"),
        protocol_comparison(report.leakage, folder / "protocol_comparison.png"),
        contribution_sources(report.shares, report.decomposition, folder / "contribution_sources.png"),
        equation_summary(report.e2.equation, folder / "equation.png"),
    ]
