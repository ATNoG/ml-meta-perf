"""The one entry point that runs every phase of the study: ``python -m ml_meta_perf``.

Given a meta-dataset it screens the term library, fits E1, E3 and the two controls,
cross-validates all of them under both leave-one-group-out protocols, extracts the
practices, writes the figures and generates the written report -- in one command, from
one set of parameters, so a result can be reproduced by repeating the command line rather
than by rerunning a notebook in the right order.

Every search knob is exposed as a flag. Defaults retain the sweep's settings, so a bare
``python -m ml_meta_perf`` reproduces the reported corrected-data run.

Study chapter: [4. The equation](../../assets/docs/04-equation.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf.data import DATASET_FEATURES, DEFAULT_PATH, MODEL_FEATURES, columns_as_arrays, load, target
from ml_meta_perf.experiment import (
    ARITIES,
    BEAM_WIDTH,
    DEFAULT,
    MAX_ABS_ZSCORE,
    MAX_TERMS,
    PENALTY,
    POOL_SIZE,
    Configuration,
    Report,
    run,
)
from ml_meta_perf.practices import render as render_practices
from ml_meta_perf.report import term_importance, write_into_chapters

#: What ``--phase`` accepts. ``all`` is the default and is what the study runs.
PHASES = ("screen", "equations", "validation", "practices", "figures", "report")


def _section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _show(frame: pl.DataFrame) -> None:
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(frame)


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
        _section("E1 -- dataset features only (fitted on all rows)")
        print(report.e1.equation)
        print(f"\nin-sample: {report.e1.in_sample}")
        for label, scores in report.e1.cross_validated.items():
            print(f"{label}: {scores}")
        print("\naccuracy vs number of terms:")
        _show(report.e1.curve)

        _section("E2 -- model features only (the control for 'model choice dominates')")
        print(report.e2.equation)
        print(f"\nin-sample: {report.e2.in_sample}")
        for label, scores in report.e2.cross_validated.items():
            print(f"{label}: {scores}")

        _section("E3 -- dataset + model features (fitted on all rows)")
        print(report.e3.equation)
        print(f"\nin-sample: {report.e3.in_sample}")
        for label, scores in report.e3.cross_validated.items():
            print(f"{label}: {scores}")
        print("\naccuracy vs number of terms:")
        _show(report.e3.curve)
        if report.e3.stability is not None:
            print("\nterm stability across leave-one-dataset-out folds:")
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
        _section("How many terms? (knee detection and Pareto fronts)")
        _show(report.term_choice)
        print("\nper-length Pareto membership, in-sample and cross-validated:")
        _show(report.pareto)

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


def configuration(arguments: argparse.Namespace) -> Configuration:
    """The configuration the run fits every equation under.

    **One, not three.** Until 2026-09-09 this returned a pair and E2 got a third object no
    flag reached, so `--penalty 3` moved two of the three equations and the comparison between
    them stopped being like-for-like. `experiment.DEFAULT` is now the single retained
    configuration and the argparse defaults *are* the constants behind it, which is also what
    makes `--help` state the real values rather than `None`.

    `max_arity` is not read from here. E3's is chosen by `experiment.search_grammars` over
    ``--arity``, and E1 and E2 -- which are fitted once rather than searched -- take the most
    parsimonious grammar in that set.
    """
    return dataclasses.replace(
        DEFAULT,
        max_abs_zscore=arguments.zscore,
        penalty=arguments.penalty,
        pool_size=arguments.pool,
        beam_width=arguments.beam,
        max_terms=arguments.max_terms,
        max_arity=min(arities(arguments)),
    )


def arities(arguments: argparse.Namespace) -> tuple[int, ...]:
    """The grammars to search, de-duplicated in the order the flags gave them."""
    return tuple(dict.fromkeys(arguments.arity)) if arguments.arity else ARITIES


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
        "practices": report.practices,
        "term_choice": report.term_choice,
        "length_choice": report.length_choice,
        "pareto": report.pareto,
        "grammars": report.grammars,
    }
    if report.e3.stability is not None:
        tables["stability"] = report.e3.stability
    written: list[Path] = []
    for name, frame in tables.items():
        path = folder / f"{name}.csv"
        frame.write_csv(path)
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-meta-perf",
        description=(
            "Fit interpretable equations predicting MCC from dataset and model meta-features, "
            "validate them under leave-one-group-out, and write the figures and the report."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data and output")
    data.add_argument("--data", default=None, help="meta-dataset CSV (defaults to the shipped corpus)")
    data.add_argument(
        "--output",
        default="results",
        help="directory for the fitted equations, the CSV tables and the report",
    )
    data.add_argument(
        "--figures",
        default="assets/figures",
        help="directory for the generated figures",
    )
    # The generated results go *into* the chapters that discuss them, between markers, rather
    # than into a report of their own: the study has six chapters and a reader should not have
    # to hold a chapter and a separate report at once. Everything outside the markers is
    # hand-written and never touched; everything inside is rewritten on every run, so a
    # chapter cannot carry a stale table. `--output` holds the equations and the CSV tables,
    # which are regenerated every run and are not tracked.
    data.add_argument(
        "--docs",
        default="assets/docs",
        help="chapter directory whose generated sections are rewritten",
    )
    data.add_argument("--no-figures", action="store_true", help="skip figure generation")
    data.add_argument("--no-report", action="store_true", help="skip rewriting the generated chapter sections")
    data.add_argument("--no-tables", action="store_true", help="skip the CSV tables")
    data.add_argument("--quiet", action="store_true", help="write files without printing the study")

    search = parser.add_argument_group("equation and search")
    # `--terms` was removed on 2026-09-09 with `Configuration.headline_terms`. It set the
    # published length by hand, and the published length is now derived from the equation's own
    # curve by `selection.floor_argmax`. `--max-terms` is a different thing and stays: the
    # search *horizon*, which is a cost control and the range the reported curve covers.
    search.add_argument("--max-terms", type=int, default=MAX_TERMS, help="longest equation the search explores")
    search.add_argument("--penalty", type=float, default=PENALTY, help="ridge penalty on standardised terms")
    # Repeatable, because the arity is searched rather than fixed: `--arity 2 --arity 3` is the
    # default set and `--arity 4` narrows the search to the grammar the negatives were measured
    # under. One value is a search over one grammar, which is what fixing the arity now means.
    search.add_argument(
        "--arity",
        type=int,
        choices=(1, 2, 3, 4),
        action="append",
        default=None,
        help=f"raw features per term; repeat to search several grammars; unset searches {ARITIES}",
    )
    search.add_argument("--pool", type=int, default=POOL_SIZE, help="terms surviving screening into the beam")
    search.add_argument("--beam", type=int, default=BEAM_WIDTH, help="beam width")
    search.add_argument(
        "--zscore",
        type=float,
        default=MAX_ABS_ZSCORE,
        help="largest standard score a term may reach before it is rejected as a spike",
    )

    parser.add_argument(
        "--phase",
        choices=(*PHASES, "all"),
        action="append",
        default=None,
        help="run a subset of the phases; repeatable (default: all)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    phases = frozenset(PHASES) if not arguments.phase or "all" in arguments.phase else frozenset(arguments.phase)

    config = configuration(arguments)
    started = time.perf_counter()
    report = run(
        arguments.data,
        config_e1=config,
        config_e2=config,
        config_e3=config,
        arities=arities(arguments),
    )
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
            source=source,
        )
        print(f"{len(pages)} chapters regenerated in {arguments.docs}")

    if arguments.figures and not arguments.no_figures and "figures" in phases:
        from ml_meta_perf.figures import generate

        written = generate(report, arguments.figures, arguments.data)
        print(f"{len(written)} figures written to {arguments.figures}")

    print(f"\ncompleted in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
