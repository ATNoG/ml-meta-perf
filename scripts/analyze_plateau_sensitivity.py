"""Explore E3-Valid plateau parameters over a completed configuration search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.patches import Rectangle

from ml_meta_perf.configuration_search import _select_plateau_valid
from ml_meta_perf.plots import FIGURE_DPI
from ml_meta_perf.selection import PLATEAU_TOLERANCE, PLATEAU_WINDOW

DEFAULT_WINDOWS = (1, 2, 3, 4, 5, 6)
DEFAULT_TOLERANCES = (0.0, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01)


def _load_shared_candidates(search_directory: Path) -> tuple[pl.DataFrame, dict[str, Any]]:
    selected_path = search_directory / "selected_configurations.json"
    finalists_path = search_directory / "finalists.csv"
    for path in (selected_path, finalists_path):
        if not path.is_file():
            raise FileNotFoundError(f"required configuration-search output is missing: {path}")

    selected = cast(dict[str, Any], json.loads(selected_path.read_text(encoding="utf-8")))
    maximum = cast(dict[str, Any], selected["e3_max"])
    finalists = pl.read_csv(finalists_path)
    shared = finalists.filter(
        (pl.col("features") == maximum["features"])
        & (pl.col("penalty") == float(maximum["penalty"]))
        & (pl.col("max_abs_zscore") == float(maximum["max_abs_zscore"]))
    )
    if shared.is_empty():
        raise ValueError("the finalist table does not contain the selected E3-MAX base")
    return shared, maximum


def evaluate(
    search_directory: Path,
    *,
    windows: tuple[int, ...],
    tolerances: tuple[float, ...],
) -> pl.DataFrame:
    """Evaluate every plateau-parameter combination without changing the saved selection."""
    shared, _ = _load_shared_candidates(search_directory)
    rows: list[dict[str, object]] = []
    for tolerance in tolerances:
        for window in windows:
            candidate, diagnostics = _select_plateau_valid(shared, tolerance=tolerance, window=window)
            rows.append(
                {
                    "plateau_tolerance": tolerance,
                    "plateau_window": window,
                    "is_current_setting": tolerance == PLATEAU_TOLERANCE and window == PLATEAU_WINDOW,
                    "plateau_detected": bool(diagnostics["plateau_detected"]),
                    "plateau_starts_at_terms": diagnostics.get("plateau_starts_at_terms"),
                    "window_gain": diagnostics.get("window_gain"),
                    "max_arity": int(candidate["max_arity"]),
                    "n_terms": int(candidate["n_terms"]),
                    "complexity": int(candidate["complexity"]),
                    "objective": float(candidate["objective"]),
                    "in_sample_r2": float(candidate["in_sample_r2"]),
                    "loo_dataset_r2": float(candidate["loo_dataset_r2"]),
                    "loo_model_r2": float(candidate["loo_model_r2"]),
                    "loo_cell_r2": float(candidate["loo_cell_r2"]),
                    "combined_r2": float(candidate["combined_r2"]),
                    "four_protocol_floor": float(candidate["four_protocol_floor"]),
                    "stability": float(candidate["stability"]),
                }
            )
    return pl.DataFrame(rows).sort("plateau_tolerance", "plateau_window")


def _format_tolerance(value: float) -> str:
    return "0" if value == 0.0 else f"{value:g}"


def _matrix(table: pl.DataFrame, windows: tuple[int, ...], tolerances: tuple[float, ...]) -> str:
    rows = ["| Plateau tolerance | " + " | ".join(f"Window {window}" for window in windows) + " |"]
    rows.append("|---:" + "|---:" * len(windows) + "|")
    indexed = {
        (float(row["plateau_tolerance"]), int(row["plateau_window"])): row
        for row in table.iter_rows(named=True)
    }
    for tolerance in tolerances:
        cells = []
        for window in windows:
            row = indexed[(tolerance, window)]
            label = f"arity {row['max_arity']}, {row['n_terms']} terms"
            if bool(row["is_current_setting"]):
                label = f"**{label}**"
            cells.append(label)
        rows.append(f"| {_format_tolerance(tolerance)} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def _unique_outcomes(table: pl.DataFrame) -> pl.DataFrame:
    metrics = (
        "max_arity",
        "n_terms",
        "complexity",
        "objective",
        "in_sample_r2",
        "loo_dataset_r2",
        "loo_model_r2",
        "loo_cell_r2",
        "combined_r2",
        "four_protocol_floor",
        "stability",
    )
    counts = table.group_by("max_arity", "n_terms").len().rename({"len": "parameter_combinations"})
    return table.select(*metrics).unique(subset=["max_arity", "n_terms"]).join(
        counts,
        on=["max_arity", "n_terms"],
    ).sort("n_terms", "max_arity")


def _outcome_table(table: pl.DataFrame) -> str:
    header = (
        "| Arity | Terms | Parameter combinations | Combined R² | LODO R² | LOMO R² | "
        "Pair R² | Objective J | Stability |"
    )
    rows = [header, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in _unique_outcomes(table).iter_rows(named=True):
        rows.append(
            f"| {row['max_arity']} | {row['n_terms']} | {row['parameter_combinations']} | "
            f"{row['combined_r2']:.4f} | {row['loo_dataset_r2']:.4f} | "
            f"{row['loo_model_r2']:.4f} | {row['loo_cell_r2']:.4f} | "
            f"{row['objective']:.4f} | {row['stability']:.4f} |"
        )
    return "\n".join(rows)


def _plot(
    table: pl.DataFrame,
    windows: tuple[int, ...],
    tolerances: tuple[float, ...],
    destination: Path,
) -> tuple[Path, Path]:
    values = []
    labels = []
    for tolerance in tolerances:
        row_values = []
        row_labels = []
        for window in windows:
            row = table.filter(
                (pl.col("plateau_tolerance") == tolerance) & (pl.col("plateau_window") == window)
            ).row(0, named=True)
            row_values.append(int(row["n_terms"]))
            row_labels.append(f"a{row['max_arity']}\n{row['n_terms']} terms")
        values.append(row_values)
        labels.append(row_labels)

    figure, axes = plt.subplots(figsize=(8.2, 6.3))
    image = axes.imshow(values, cmap="viridis", aspect="auto")
    for row_index, row_labels in enumerate(labels):
        for column_index, label in enumerate(row_labels):
            colour = "white" if values[row_index][column_index] >= 14 else "black"
            axes.text(column_index, row_index, label, ha="center", va="center", fontsize=7, color=colour)

    if PLATEAU_WINDOW in windows and PLATEAU_TOLERANCE in tolerances:
        column = windows.index(PLATEAU_WINDOW)
        row = tolerances.index(PLATEAU_TOLERANCE)
        axes.add_patch(
            Rectangle(
                (column - 0.48, row - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor="#ef4444",
                linewidth=2.5,
            )
        )

    axes.set_xticks(range(len(windows)), [str(value) for value in windows])
    axes.set_yticks(range(len(tolerances)), [_format_tolerance(value) for value in tolerances])
    axes.set_xlabel("plateau window")
    axes.set_ylabel("plateau tolerance")
    axes.set_title("E3-Valid plateau-parameter sensitivity (1-25 terms)")
    colour_bar = figure.colorbar(image, ax=axes, pad=0.02)
    colour_bar.set_label("selected number of terms")
    figure.tight_layout()

    png_path = destination / "plateau_sensitivity.png"
    pdf_path = destination / "plateau_sensitivity.pdf"
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(figure)
    return png_path, pdf_path


def _report(
    table: pl.DataFrame,
    maximum: dict[str, Any],
    windows: tuple[int, ...],
    tolerances: tuple[float, ...],
    destination: Path,
) -> Path:
    current = table.filter(pl.col("is_current_setting")).row(0, named=True)
    outcomes = _unique_outcomes(table)
    dominant = outcomes.sort("parameter_combinations", descending=True).row(0, named=True)
    current_matches = table.filter(
        (pl.col("max_arity") == current["max_arity"]) & (pl.col("n_terms") == current["n_terms"])
    )
    same_window = current_matches.filter(pl.col("plateau_window") == PLATEAU_WINDOW)
    tolerance_low = cast(float, same_window["plateau_tolerance"].min())
    tolerance_high = cast(float, same_window["plateau_tolerance"].max())
    shorter = (
        outcomes.filter(pl.col("n_terms") < current["n_terms"])
        .sort("n_terms", descending=True)
        .row(0, named=True)
    )
    longer = (
        outcomes.filter(pl.col("n_terms") > current["n_terms"])
        .sort("n_terms", descending=True)
        .row(0, named=True)
    )
    features = ", ".join(f"`{name}`" for name in json.loads(str(maximum["features"])))
    report = f"""# E3-Valid Plateau-Parameter Sensitivity: 1-25 Terms

## Scope

This analysis changes only the plateau window and tolerance. It reuses the completed global
1-25-term search and fixes the E3-MAX base configuration: ridge penalty
`{maximum['penalty']}`, maximum absolute z-score `{maximum['max_abs_zscore']}`, and model
descriptors {features}. It does not rerun symbolic regression and does not change
the saved E3-Valid selection.

The current setting is `plateau_window = {PLATEAU_WINDOW}` and
`plateau_tolerance = {PLATEAU_TOLERANCE}`. It selects arity {current['max_arity']} with
{current['n_terms']} terms and Combined R² {current['combined_r2']:.4f}.

## Selected equation across the parameter grid

{_matrix(table, windows, tolerances)}

The bold cell is the current setting.

![Plateau-parameter sensitivity](plateau_sensitivity.png)

## Unique selected outcomes

{_outcome_table(table)}

## Recommended retained setting

Retain `plateau_window = {PLATEAU_WINDOW}` and
`plateau_tolerance = {PLATEAU_TOLERANCE}`. The tolerance has the direct interpretation that
an increase of no more than 0.001 in best-so-far Combined R² across the next three equation
lengths constitutes a plateau. It is also the strictest tested tolerance at window 3 that
selects the 18-term equation.

A two-point window stops at {shorter['n_terms']} terms before the later improvement at 18.
Adding the three terms raises Combined R² by
{current['combined_r2'] - shorter['combined_r2']:.4f}, objective J by
{current['objective'] - shorter['objective']:.4f}, and stability by
{current['stability'] - shorter['stability']:.4f}. Conversely, windows of four or more select
the {longer['n_terms']}-term boundary solution. That equation raises Combined R² by
{longer['combined_r2'] - current['combined_r2']:.4f} but increases grammar complexity from
{current['complexity']} to {longer['complexity']}, while objective J falls by
{current['objective'] - longer['objective']:.4f} and stability falls by
{current['stability'] - longer['stability']:.4f}.

Among the distinct outcomes in this grid, the retained 18-term equation has the highest
objective J and term stability. It therefore provides the clearest observed balance between
predictive performance, equation size, grammar complexity, and repeatability.

## Interpretation

The grid contains {table.height} parameter combinations and produces {outcomes.height} unique
equations. The most frequent result is arity {dominant['max_arity']} with
{dominant['n_terms']} terms, selected by {dominant['parameter_combinations']} combinations.
The current 18-term outcome appears in {current_matches.height} combinations. At the current
window of {PLATEAU_WINDOW}, it is retained for tested tolerances from
{_format_tolerance(tolerance_low)} through {_format_tolerance(tolerance_high)}.

This is a sensitivity analysis rather than an unbiased hyperparameter optimisation. The same
validation curves define and assess each selection, so choosing the numerically strongest cell
would reuse the evidence and would favour longer equations. The useful result is the size and
location of stable parameter regions, together with their performance and complexity.
"""
    path = destination / "plateau_sensitivity.md"
    path.write_text(report, encoding="utf-8", newline="\n")
    return path


def generate(
    search_directory: Path,
    *,
    windows: tuple[int, ...],
    tolerances: tuple[float, ...],
) -> tuple[Path, Path, Path, Path]:
    """Write the sensitivity table, figure and report beside the completed search."""
    if not windows or any(value < 1 for value in windows):
        raise ValueError("plateau windows must be positive")
    if not tolerances or any(value < 0.0 for value in tolerances):
        raise ValueError("plateau tolerances must be non-negative")
    windows = tuple(sorted({*windows, PLATEAU_WINDOW}))
    tolerances = tuple(sorted({*tolerances, PLATEAU_TOLERANCE}))
    table = evaluate(search_directory, windows=windows, tolerances=tolerances)
    csv_path = search_directory / "plateau_sensitivity.csv"
    table.write_csv(csv_path)
    _, maximum = _load_shared_candidates(search_directory)
    png_path, pdf_path = _plot(table, windows, tolerances, search_directory)
    report_path = _report(table, maximum, windows, tolerances, search_directory)
    return csv_path, png_path, pdf_path, report_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search_directory", nargs="?", type=Path, default=Path("results/configuration_search"))
    parser.add_argument("--window", type=int, action="append", help="plateau window; repeat to define the grid")
    parser.add_argument("--tolerance", type=float, action="append", help="plateau tolerance; repeat to define the grid")
    arguments = parser.parse_args()
    windows = tuple(arguments.window) if arguments.window else DEFAULT_WINDOWS
    tolerances = tuple(arguments.tolerance) if arguments.tolerance else DEFAULT_TOLERANCES
    for path in generate(arguments.search_directory.resolve(), windows=windows, tolerances=tolerances):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
