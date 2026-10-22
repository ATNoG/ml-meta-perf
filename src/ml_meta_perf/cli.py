"""The command line: ``ml-meta-perf fit | sweep | init``.

* ``fit`` reads a study file, fits the equations it lists on its corpus, scores them under the
  protocols it names and writes the result folder (`ml_meta_perf.output`).
* ``sweep`` searches the shared hyperparameters and proposes a study file (`ml_meta_perf.sweep`).
* ``init`` writes a study file for a CSV (`ml_meta_perf.skeleton`).

Every hyperparameter comes from the study file -- the command line holds no tuned default of its
own -- so a bare ``ml-meta-perf fit`` reproduces ``config/mcc.json``. The file is read with
`jsonargparse`: ``--config other.json`` replaces it, any field can be overridden as a dotted flag
(``--search.penalty 3``) and ``--print_config`` shows the effective study.

Study chapter: [7. Other targets][study-chapter] -- what a study is made of.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/07-generic-targets.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import polars as pl
from jsonargparse import ArgumentParser, Namespace  # pyright: ignore[reportPrivateImportUsage]

from ml_meta_perf import __version__, sweep
from ml_meta_perf.arguments import study_parser
from ml_meta_perf.config import ConfigError, study_of
from ml_meta_perf.data import SchemaError, resolve_path
from ml_meta_perf.output import OverwriteError, study_directory, write_study
from ml_meta_perf.skeleton import skeleton
from ml_meta_perf.study import StudyReport, run_study


def _section(title: str) -> None:
    """Print a visual banner heading for ``title`` on stdout."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _show(frame: pl.DataFrame) -> None:
    """Print ``frame`` formatted to three decimal places without truncating study rows."""
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, float_precision=3):
        print(frame)


def _configure_windows_output() -> None:
    """Keep Polars' Unicode table borders printable under a legacy Windows code page."""
    if sys.platform != "win32":
        return
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")


def fit_parser() -> ArgumentParser:
    """Build the argument parser for ``ml-meta-perf fit``."""
    parser = study_parser("ml-meta-perf fit", "Fit the equations a study file lists and write the result folder.")
    parser.add_argument(
        "--output", type=str, default="out", help="root of the result folders (<root>/<csv stem>/<target>)"
    )
    parser.add_argument("--force", action="store_true", help="replace a result folder made from a different corpus")
    parser.add_argument("--quiet", action="store_true", help="write files without printing the study")
    return parser


def init_parser() -> ArgumentParser:
    """Build the argument parser for ``ml-meta-perf init``."""
    parser = ArgumentParser(prog="ml-meta-perf init", description="Write a study file for a corpus CSV.")
    parser.add_argument("corpus", type=str, help="the corpus CSV")
    parser.add_argument("--dataset", type=str, required=True, help="column naming the dataset (first grouping factor)")
    parser.add_argument("--response", type=str, required=True, help="the target column")
    parser.add_argument("--model", type=str | None, default=None, help="column naming the model, when there is one")
    parser.add_argument(
        "--bounds", type=str | None, default=None, help='the range the target can take, e.g. "[0, null]"'
    )
    parser.add_argument("--transform", type=str, default="none", help="the scale to fit on: none, log1p or log")
    parser.add_argument("--output", type=str, default="study.json", help="the study file to write")
    parser.add_argument("--force", action="store_true", help="overwrite an existing study file")
    return parser


def build_parser() -> ArgumentParser:
    """Build the top-level ``ml-meta-perf`` parser with ``fit``, ``sweep`` and ``init`` subcommands."""
    parser = ArgumentParser(
        prog="ml-meta-perf",
        description=(
            "Interpretable additive equations for a performance measure, fitted from dataset and model "
            "meta-features and validated by leave-one-group-out."
        ),
    )
    parser.add_argument("--version", action="version", version=f"ml-meta-perf {__version__}")
    commands = parser.add_subcommands(required=True)
    commands.add_subcommand("fit", fit_parser(), help="fit and validate the equations of a study")
    commands.add_subcommand("sweep", sweep.build_parser(), help="search the shared hyperparameters")
    commands.add_subcommand("init", init_parser(), help="write a study file for a corpus CSV")
    return parser


def _print_report(report: StudyReport) -> None:
    """Print the equations, target-scale scores and E3-Valid group shares of ``report`` to stdout."""
    schema = report.schema
    for name, equation in report.equations.items():
        _section(f"{name}: {schema.label}  ({equation.n_terms} terms, arity {equation.arity})")
        print(equation.equation)
    scale = "original" if schema.log_target else "fit"
    _section(f"scores ({'on ' + schema.target_column + ' itself' if schema.log_target else 'on the target'})")
    _show(report.scores.filter(pl.col("scale") == scale).drop("scale"))
    if not report.shares.is_empty():
        _section("which features carry E3-Valid")
        _show(report.shares)


def run_fit(arguments: Namespace) -> int:
    """Execute ``ml-meta-perf fit``: run the study and write its result folder."""
    study = study_of(arguments)
    corpus = Path(arguments.data) if arguments.data else resolve_path(study.corpus.path)
    started = time.perf_counter()
    report = run_study(study, corpus)
    elapsed = time.perf_counter() - started
    if not arguments.quiet:
        _print_report(report)
    destination = study_directory(arguments.output, corpus, report)
    written = write_study(report, study, corpus, destination, force=arguments.force)
    print(f"\n{len(written)} files written to {destination}")
    print(f"completed in {elapsed:.1f}s")
    return 0


def run_init(arguments: Namespace) -> int:
    """Execute ``ml-meta-perf init``: infer a study skeleton from a corpus CSV and write it to disk."""
    output = Path(arguments.output)
    if output.exists() and not arguments.force:
        raise ConfigError(f"{output} exists; pass --force to overwrite it")
    bounds = None if arguments.bounds is None else json.loads(arguments.bounds)
    if bounds is not None and not (isinstance(bounds, list) and len(bounds) == 2):
        raise ConfigError(f"--bounds must be a [low, high] pair, got {arguments.bounds}")
    study, left_out = skeleton(
        Path(arguments.corpus),
        dataset_column=arguments.dataset,
        response=arguments.response,
        model_column=arguments.model,
        bounds=None if bounds is None else (bounds[0], bounds[1]),
        transform=arguments.transform,
    )
    output.write_text(json.dumps(study, indent=2) + "\n", encoding="utf-8")
    features: dict[str, list[str]] = study["features"]  # pyright: ignore[reportAssignmentType]
    print(f"wrote {output}: {len(features['dataset'])} dataset features, {len(features['model'])} model features")
    for reason, names in left_out.items():
        if names:
            print(f"left out ({reason}): {', '.join(names)}")
    print("edit the features, the bounds and the protocols, then run: ml-meta-perf fit --config", output)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for ``ml-meta-perf``; returns the process exit code."""
    _configure_windows_output()
    parser = build_parser()
    parsed: Namespace = parser.instantiate_classes(parser.parse_args(argv))  # pyright: ignore[reportAssignmentType]
    command = str(parsed["subcommand"])
    arguments = parsed[command]
    try:
        if command == "sweep":
            return sweep.run(arguments, parser)
        if command == "init":
            return run_init(arguments)
        return run_fit(arguments)
    except (ConfigError, SchemaError, OverwriteError) as error:
        parser.error(str(error))  # exits with status 2


if __name__ == "__main__":
    sys.exit(main())
