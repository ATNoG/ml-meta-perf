"""Command line entry point: ``python -m metafit``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from metafit.experiment import Report, run


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _show(frame: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(frame)


def render(report: Report) -> None:
    """Print the full study to stdout."""
    _section("Correlation screening -- linear vs monotone association with MCC")
    _show(report.correlations)

    _section("E1 -- dataset features only (fitted on 20 dataset means)")
    print(report.e1.equation)
    print(f"\nin-sample: {report.e1.in_sample}")
    for label, scores in report.e1.cross_validated.items():
        print(f"{label}: {scores}")
    print("\naccuracy vs number of terms:")
    _show(report.e1.curve)

    _section("E2 -- dataset + model features (fitted on all rows)")
    print(report.e2.equation)
    print(f"\nin-sample: {report.e2.in_sample}")
    for label, scores in report.e2.cross_validated.items():
        print(f"{label}: {scores}")
    print("\naccuracy vs number of terms:")
    _show(report.e2.curve)
    if report.e2.stability is not None:
        print("\nterm stability across leave-one-dataset-out folds:")
        _show(report.e2.stability.head(20))

    _section("E1 vs E2 on the common scale (all rows)")
    _show(report.comparison)

    _section("Baselines")
    _show(report.baselines)

    _section("Validation protocol matters (same equation, different splits)")
    _show(report.leakage)

    _section("Model selection on held-out datasets")
    _show(report.selection)
    print(
        f"\nmean spearman = {report.selection['spearman'].mean():.3f}   "
        f"mean top-1 regret = {report.selection['regret'].mean():.3f}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metafit", description=__doc__)
    parser.add_argument("--data", default=None, help="path to meta_dataset.csv")
    parser.add_argument("--save", default=None, help="directory to write the fitted equations to")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run a reduced configuration to check the pipeline; not the reported study",
    )
    arguments = parser.parse_args(argv)

    report = run(arguments.data, quick=arguments.quick)
    render(report)

    if arguments.save:
        destination = Path(arguments.save)
        destination.mkdir(parents=True, exist_ok=True)
        report.e1.equation.save(destination / "e1.json")
        report.e2.equation.save(destination / "e2.json")
        print(f"\nequations written to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
