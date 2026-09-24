"""Plot the retained E3-Valid and E3-MAX selections over the term-count curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from ml_meta_perf.plots import CEILING, FIGURE_DPI, IN_SAMPLE, LOO_DATASET, LOO_MODEL

DHO_COLOR = "#7f3c8d"


def _candidate_rank(row: dict[str, Any]) -> tuple[float, int, int, int]:
    return (
        -float(row["combined_r2"]),
        int(row["complexity"]),
        int(row["max_arity"]),
        int(row["requested_terms"]),
    )


def _best_per_term_count(table: pl.DataFrame) -> pl.DataFrame:
    selected: dict[int, dict[str, Any]] = {}
    for row in table.to_dicts():
        length = int(row["n_terms"])
        incumbent = selected.get(length)
        if incumbent is None or _candidate_rank(row) < _candidate_rank(incumbent):
            selected[length] = row
    return pl.DataFrame([selected[length] for length in sorted(selected)])


def _additive_reference_value(search_directory: Path) -> float | None:
    baselines = search_directory.parent / "baselines.csv"
    if not baselines.is_file():
        return None
    table = pl.read_csv(baselines)
    label_column = "baseline" if "baseline" in table.columns else table.columns[0]
    # Accept the legacy result label so archived searches remain reproducible after the
    # public figure terminology changed to "additive mean-based reference".
    rows = table.filter(pl.col(label_column).cast(pl.String).str.contains("additive mean-based reference"))
    if rows.is_empty() or "r2" not in rows.columns:
        return None
    return float(rows["r2"][0])


def generate(search_directory: Path) -> tuple[Path, Path, Path]:
    selected_path = search_directory / "selected_configurations.json"
    finalists_path = search_directory / "finalists.csv"
    for path in (selected_path, finalists_path):
        if not path.is_file():
            raise FileNotFoundError(f"required configuration-search output is missing: {path}")

    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    finalists = pl.read_csv(finalists_path)
    maximum = payload["e3_max"]
    shared = finalists.filter(
        (pl.col("features") == maximum["features"])
        & (pl.col("penalty") == float(maximum["penalty"]))
        & (pl.col("max_abs_zscore") == float(maximum["max_abs_zscore"]))
    )
    curve = _best_per_term_count(shared)
    curve_path = search_directory / "e3_valid_term_count_curve.csv"
    curve.select(
        "n_terms",
        "max_arity",
        "in_sample_r2",
        "loo_dataset_r2",
        "loo_model_r2",
        "loo_cell_r2",
        "combined_r2",
        "four_protocol_floor",
    ).write_csv(curve_path)

    lengths = curve["n_terms"].to_numpy()
    figure, axes = plt.subplots(figsize=(8.4, 5.3))
    axes.plot(
        lengths,
        curve["in_sample_r2"].to_numpy(),
        "o-",
        color=IN_SAMPLE,
        label="IS",
        linewidth=2,
        markersize=3.5,
    )
    axes.plot(
        lengths,
        curve["loo_dataset_r2"].to_numpy(),
        "s--",
        color=LOO_DATASET,
        label="LODO",
        markersize=3.5,
    )
    axes.plot(
        lengths,
        curve["loo_model_r2"].to_numpy(),
        "^:",
        color=LOO_MODEL,
        label="LOMO",
        markersize=3.5,
    )
    axes.plot(
        lengths,
        curve["loo_cell_r2"].to_numpy(),
        "v-.",
        color=DHO_COLOR,
        label="DHO",
        linewidth=1.6,
        markersize=3.5,
    )
    axes.plot(
        lengths,
        curve["combined_r2"].to_numpy(),
        "D-",
        color="#4b5563",
        label="combined $R^2$",
        linewidth=1.25,
        markersize=3,
        alpha=0.8,
    )

    reference = _additive_reference_value(search_directory)
    if reference is not None:
        axes.axhline(
            reference,
            color=CEILING,
            linestyle="-.",
            linewidth=1.2,
            label=f"additive mean-based reference ({reference:.3f})",
        )

    markers = (("E3-Valid", payload["e3_valid"], "#e17c05"), ("E3-MAX", maximum, "#111827"))
    for label, candidate, colour in markers:
        terms = int(candidate["n_terms"])
        score = float(candidate["combined_r2"])
        axes.axvline(
            terms,
            color=colour,
            linestyle="--",
            linewidth=1.25,
            alpha=0.9,
            label=f"{label}\n({terms} terms; Combined $R^2$ = {score:.3f})",
        )
        axes.annotate(
            f"{score:.3f}",
            xy=(terms, score),
            xytext=(0, 8),
            textcoords="offset points",
            color=colour,
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": colour, "alpha": 0.85},
            zorder=6,
        )

    axes.set_xlabel("number of terms")
    axes.set_ylabel("$R^2$")
    if len(lengths) <= 25:
        axes.set_xticks(lengths)
    else:
        axes.set_xticks(range(10, int(lengths.max()) + 1, 10))
    axes.set_xlim(float(lengths.min()) - 0.5, float(lengths.max()) + 0.5)
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    figure.tight_layout()

    png_path = search_directory / "e3_valid_term_count_curve.png"
    pdf_path = png_path.with_suffix(".pdf")
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True)
    figure.savefig(pdf_path, bbox_inches="tight", transparent=True, metadata={"CreationDate": None})
    plt.close(figure)
    return png_path, pdf_path, curve_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search_directory", type=Path)
    arguments = parser.parse_args()
    for path in generate(arguments.search_directory.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
