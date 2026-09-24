"""Plot the three E3 R2 validation curves from a configuration search."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl

from ml_meta_perf.plots import FIGURE_DPI, IN_SAMPLE, LOO_DATASET, LOO_MODEL

DHO_COLOR = "#7f3c8d"

CURVE_COLUMNS = (
    "n_terms",
    "in_sample_r2",
    "loo_dataset_r2",
    "loo_model_r2",
    "loo_cell_r2",
)


def generate(search_directory: Path) -> tuple[Path, Path, Path]:
    """Generate the IS, LODO, LOMO, and doubly held-out (DHO) term-count curves."""
    source_path = search_directory / "e3_valid_term_count_curve.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"required term-count curve is missing: {source_path}")

    curve = pl.read_csv(source_path)
    missing = set(CURVE_COLUMNS) - set(curve.columns)
    if missing:
        raise ValueError(f"term-count curve is missing required columns: {sorted(missing)}")
    curve = curve.select(CURVE_COLUMNS).sort("n_terms")

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

    axes.set_xlabel("number of terms")
    axes.set_ylabel("$R^2$")
    if len(lengths) <= 25:
        axes.set_xticks(lengths)
    else:
        axes.set_xticks(range(10, int(lengths.max()) + 1, 10))
    axes.set_xlim(float(lengths.min()) - 0.5, float(lengths.max()) + 0.5)
    axes.grid(alpha=0.25, linestyle=":")
    axes.legend(
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=4,
    )
    figure.tight_layout()

    csv_path = search_directory / "e3_validation_term_count_curve.csv"
    png_path = search_directory / "e3_validation_term_count_curve.png"
    pdf_path = png_path.with_suffix(".pdf")
    curve.write_csv(csv_path)
    figure.savefig(png_path, dpi=FIGURE_DPI, bbox_inches="tight", transparent=True)
    figure.savefig(pdf_path, bbox_inches="tight", transparent=True, metadata={"CreationDate": None})
    plt.close(figure)
    return png_path, pdf_path, csv_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("search_directory", type=Path)
    arguments = parser.parse_args()
    for path in generate(arguments.search_directory.resolve()):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
