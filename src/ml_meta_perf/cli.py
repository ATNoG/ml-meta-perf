"""The one entry point that runs every phase of the study: ``python -m ml_meta_perf``.

Given a meta-dataset it screens the term library, fits E1, E2, E3-Valid and E3-MAX,
cross-validates them under the retained grouped protocols, extracts the
practices, writes the figures and generates the written report -- in one command, from
one set of parameters, so a result can be reproduced by repeating the command line rather
than by rerunning a notebook in the right order.

Every hyperparameter comes from ``config/study.json`` -- the command line holds no tuned
default of its own -- so a bare ``python -m ml_meta_perf`` reproduces the reported run. The
file is read with `jsonargparse`: ``--config other.json`` replaces it, any field can be
overridden as a dotted flag (``--search.penalty 3``, ``--selection.delta 0.02``), and
``--print_config`` shows the effective configuration.

Study chapter: [4. The equation][study-chapter] -- the rationale, in
prose, with the figures.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/04-equation.md
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace  # pyright: ignore[reportPrivateImportUsage]

from ml_meta_perf.config import (
    DEFAULT_CONFIG_PATH,
    Configuration,
    OpaqueConfig,
    SelectionConfig,
    SweepConfig,
)
from ml_meta_perf.data import DATASET_FEATURES, DEFAULT_PATH, MODEL_FEATURES, columns_as_arrays, load, target
from ml_meta_perf.experiment import Report, run
from ml_meta_perf.practices import render as render_practices
from ml_meta_perf.report import term_importance, write_into_chapters

#: What ``--phase`` accepts. ``all`` is the default and is what the study runs.
PHASES = ("screen", "equations", "validation", "practices", "figures", "report")
PROTOCOL_LABELS = {"loo_dataset": "LODO", "loo_model": "LOMO", "loo_cell": "DHO"}


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _show(frame: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(frame)


def _configure_windows_output() -> None:
    """Keep Polars' Unicode table borders printable under a legacy Windows code page."""
    if sys.platform != "win32":
        return
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


def render(
    report: Report,
    phases: frozenset[str] = frozenset(PHASES),
    columns: dict[str, np.ndarray] | None = None,
) -> None:
    """Print the full study to stdout, restricted to the requested phases.

    ``columns`` is the feature matrix the report was fitted on. It is passed in rather
    than reloaded because reloading would use the default meta-dataset, which is the
    wrong one whenever ``--data`` was given.
    """
    if columns is None:
        columns = columns_as_arrays(load(), DATASET_FEATURES + MODEL_FEATURES)

    if "screen" in phases:
        _section("Correlation screening -- linear vs monotone association with MCC")
        _show(report.correlations)

    if "equations" in phases:
        print(
            "Evaluation protocols: in-sample (IS), leave-one-dataset-out (LODO), "
            "leave-one-model-out (LOMO), and doubly held out (DHO)."
        )
        _section("E1 -- dataset features only (fitted on all rows)")
        print(report.e1.equation)
        print(f"\nIS: {report.e1.in_sample}")
        for label, scores in report.e1.cross_validated.items():
            print(f"{PROTOCOL_LABELS.get(label, label)}: {scores}")
        print("\naccuracy vs number of terms:")
        _show(report.e1.curve)

        _section("E2 -- model features only (the control for 'model choice dominates')")
        print(report.e2.equation)
        print(f"\nIS: {report.e2.in_sample}")
        for label, scores in report.e2.cross_validated.items():
            print(f"{PROTOCOL_LABELS.get(label, label)}: {scores}")

        _section("E3 -- dataset + model features (fitted on all rows)")
        print(report.e3.equation)
        print(f"\nIS: {report.e3.in_sample}")
        for label, scores in report.e3.cross_validated.items():
            print(f"{PROTOCOL_LABELS.get(label, label)}: {scores}")
        print("\naccuracy vs number of terms:")
        _show(report.e3.curve)
        if report.e3.stability is not None:
            print("\nterm stability across LODO folds:")
            _show(report.e3.stability.head(20))

    if "practices" in phases:
        _section("Where the signal lives")
        print("variance of MCC explained by identity alone (no equation involved):")
        _show(report.decomposition)
        print("\nshare of E3's output variance, by which features its terms use:")
        _show(report.shares)

        _section("Term importance -- ranked by standardised weight, major terms flagged")
        _show(term_importance(report.e3.equation, columns, DATASET_FEATURES, MODEL_FEATURES, report.e3.stability))

        _section("Extracted practices")
        print(render_practices(report.practices))
        print("\nevidence:")
        _show(report.practices.select("feature", "n_terms", "direction", "effect", "stability", "confidence"))

        _section("What each term is worth (MCC units)")
        _show(report.effects)

    if "validation" in phases:
        _section("E3-Valid and E3-MAX selection")
        _show(report.grammars)

        _section("E1 vs E2 vs E3 on the common scale (all rows)")
        _show(report.comparison)

        _section("Oracle ladder -- what each interaction component would be worth")
        _show(report.oracles)
        _section("How much of the leading interaction the equation reaches")
        _show(report.interaction)

        _section("Baselines")
        _show(report.baselines)

        _section("Validation protocol matters (same equation, different splits)")
        _show(report.leakage)

        _section("Go/no-go decision quality on held-out datasets")
        _show(report.decision)

        _section("Model selection on held-out datasets")
        _show(report.selection)
        print(
            f"\nmean spearman = {report.selection['spearman'].mean():.3f}   "
            f"mean top-1 regret = {report.selection['regret'].mean():.3f}"
        )


def _save_tables(report: Report, folder: Path) -> list[Path]:
    """Every table the study produced, as CSV, for a paper's tables and plots."""
    tables: dict[str, pl.DataFrame] = {
        "correlations": report.correlations,
        "reach": report.reach,
        "ceiling": report.ceiling,
        "curve_e1": report.e1.curve,
        "curve_e2": report.e2.curve,
        "curve_e3": report.e3.curve,
        "comparison": report.comparison,
        "baselines": report.baselines,
        "ranking_baselines": report.ranking_baselines,
        "decision_baselines": report.decision_baselines,
        "leakage": report.leakage,
        "oracles": report.oracles,
        "interaction": report.interaction,
        "decision": report.decision,
        "selection": report.selection,
        "decomposition": report.decomposition,
        "shares": report.shares,
        "term_effects": report.effects,
        "feature_practices": report.feature_practices,
        "practices": report.practices,
        "length_choice": report.length_choice,
        "grammars": report.grammars,
        "optimism": report.optimism,
    }
    if report.e3.stability is not None:
        tables["stability"] = report.e3.stability
    written: list[Path] = []
    for name, frame in tables.items():
        path = folder / f"{name}.csv"
        frame.write_csv(path)
        written.append(path)
    return written


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ml-meta-perf",
        description=(
            "Fit interpretable equations predicting MCC from dataset and model meta-features, "
            "validate them under leave-one-group-out, and write the figures and the report."
        ),
        default_config_files=[str(DEFAULT_CONFIG_PATH)],
    )
    parser.add_argument(
        "--config",
        action=ActionConfigFile,
        help=f"study configuration JSON (default: {DEFAULT_CONFIG_PATH.name} in config/)",
    )
    # The hyperparameters. Their values live only in the configuration file; each field can be
    # overridden as a dotted flag, e.g. `--search.penalty 3`.
    parser.add_argument("--search", type=Configuration, help="search hyperparameters shared by E1, E2 and E3")
    parser.add_argument("--selection", type=SelectionConfig, help="the equation-length rule")
    parser.add_argument("--opaque", type=OpaqueConfig, help="the opaque regressors' sizes")
    parser.add_argument("--sweep", type=SweepConfig, help="the grid `ml-meta-perf-search` explores (unused here)")

    parser.add_argument(
        "--data", type=str | None, default=None, help="meta-dataset CSV (default: dataset/meta_dataset.csv)"
    )
    parser.add_argument("--output", type=str, default="results", help="directory for the equations and CSV tables")
    parser.add_argument("--figures", type=str, default="assets/figures", help="directory for the generated figures")
    # The generated results go *into* the chapters that discuss them, between markers, rather
    # than into a report of their own. Everything outside the markers is hand-written and never
    # touched; everything inside is rewritten on every run, so a chapter cannot carry a stale table.
    parser.add_argument(
        "--docs", type=str, default="assets/docs", help="chapters whose generated sections are rewritten"
    )
    parser.add_argument("--readme", type=str, default="README.md", help="README whose generated headline is rewritten")
    parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
    parser.add_argument("--no-report", action="store_true", help="skip rewriting the generated chapter sections")
    parser.add_argument("--no-tables", action="store_true", help="skip the CSV tables")
    parser.add_argument("--quiet", action="store_true", help="write files without printing the study")
    parser.add_argument(
        "--phase",
        type=str | None,
        default=None,
        help=f"comma-separated phases to run (default: all): {', '.join(PHASES)}",
    )
    return parser


def phases_of(arguments: Namespace) -> frozenset[str]:
    """The phases to run; every phase when none, or ``all``, was asked for."""
    requested = [name.strip() for name in (arguments.phase or "").split(",") if name.strip()]
    if not requested or "all" in requested:
        return frozenset(PHASES)
    unknown = sorted(set(requested) - set(PHASES))
    if unknown:
        raise SystemExit(f"ml-meta-perf: unknown phase(s): {', '.join(unknown)}")
    return frozenset(requested)


def main(argv: list[str] | None = None) -> int:
    _configure_windows_output()
    parser = build_parser()
    arguments = parser.instantiate(parser.parse_args(argv))
    phases = phases_of(arguments)

    config: Configuration = arguments.search
    started = time.perf_counter()
    report = run(arguments.data, config=config, selection=arguments.selection, opaque=arguments.opaque)
    elapsed = time.perf_counter() - started

    frame = load(arguments.data)
    columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
    source = arguments.data or str(DEFAULT_PATH)

    if not arguments.quiet:
        render(report, phases, columns)

    if arguments.output:
        destination = Path(arguments.output)
        destination.mkdir(parents=True, exist_ok=True)
        report.e1.equation.save(destination / "e1.json")
        report.e3.equation.save(destination / "e3.json")
        report.e2.equation.save(destination / "e2.json")
        print(f"\nequations written to {destination}")

        if not arguments.no_tables:
            written = _save_tables(report, destination)
            print(f"{len(written)} tables written to {destination}")

    if arguments.docs and not arguments.no_report and "report" in phases:
        pages = write_into_chapters(
            report,
            columns,
            target(frame),
            DATASET_FEATURES,
            MODEL_FEATURES,
            arguments.docs,
            frame=frame,
            config=config,
            selection=arguments.selection,
            source=source,
            readme=arguments.readme,
        )
        print(f"{len(pages)} pages regenerated ({arguments.docs}, {arguments.readme})")

    if arguments.figures and not arguments.no_figures and "figures" in phases:
        from ml_meta_perf.figures import generate

        written = generate(report, arguments.figures, arguments.data)
        print(f"{len(written)} figures written to {arguments.figures}")

    print(f"\ncompleted in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
