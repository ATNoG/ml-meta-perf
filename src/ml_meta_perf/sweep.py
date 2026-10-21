"""Choose the shared configuration: ``ml-meta-perf-search``.

The study fits E1, E2 and E3 under **one** hyperparameter configuration (``config/study.json``),
chosen by E3-Valid's own criterion. This sweep makes that choice reproducible:

1. every point of the grid in the file's ``sweep`` section -- stability cap x ridge penalty x
   arity, with the pool, beam and horizon of its ``search`` section -- is fitted once with
   `experiment.fit_path` (E3, every feature) and cross-validated at every length;
2. each point's length is chosen by the study's own rule, `selection.plateau_start`;
3. points at or below ``readable_terms`` whose smoothed worst-protocol R2 is within ``band`` of
   the best such point are **indistinguishable on R2**, because adjacent configurations differ
   by less than adjacent lengths do;
4. among those, the point whose doubly-held-out (DHO) predictions rank the models best --
   mean average precision over the twenty datasets -- is proposed; ties go to fewer terms,
   lower arity and then the heavier penalty.

The DHO task metrics break the tie because they are what the equation is *for* -- ranking
models for a dataset nobody has run -- and they separate configurations that R2 cannot: at the
same floor, ranking AP ranges from 0.70 to 0.85.

A FlexFL target (``--target``, with ``--data``) runs the same grid on that corpus's E3. Its
targets are per-run costs and scores with no go/no-go threshold, so the DHO task metrics are
left empty and the band is broken by the remaining ties alone: fewer terms, lower arity, then
the heavier penalty.

Nothing is changed in place. The run writes every curve, one row per configuration, and a
``proposed_study.json``; adopting it is a reviewed edit to ``config/study.json``.

The search is small enough for a workstation: one fit per configuration, no re-selection
inside folds, one BLAS thread per worker.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from jsonargparse import ArgumentParser, Namespace  # pyright: ignore[reportPrivateImportUsage]

from ml_meta_perf.arguments import study_parser
from ml_meta_perf.config import (
    ConfigError,
    Configuration,
    SelectionConfig,
    StudyConfig,
    SweepConfig,
    default_schema,
    study_of,
)
from ml_meta_perf.data import (
    Schema,
    drop_constant_features,
    groups,
    load,
    resolve_path,
    target,
)
from ml_meta_perf.selection import JUDGED_PROTOCOLS, floor_curve, plateau_start, smoothed
from ml_meta_perf.validate import decision_report, ranking_report

#: The go/no-go threshold the decision metrics are reported at, as in chapter 5.
DECISION_THRESHOLD = 0.7
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def grid(study: StudyConfig) -> list[Configuration]:
    """Every configuration the sweep fits: the ``sweep`` grid over the ``search`` base."""
    return [
        dataclasses.replace(study.search, max_abs_zscore=zscore, penalty=penalty, max_arity=arity)
        for zscore, penalty, arity in itertools.product(study.sweep.zscores, study.sweep.penalties, study.sweep.arities)
    ]


def has_task_metrics(schema: Schema) -> bool:
    """Whether the DHO ranking and go/no-go metrics apply: a score in [-1, 1] scored under DHO.

    They rank the models within each dataset and call a model good above ``DECISION_THRESHOLD``,
    which only means something for a bounded performance score such as MCC. A cost has no threshold.
    """
    return "DHO" in schema.protocols and schema.bounds == (-1.0, 1.0)


TASK_METRICS = ("dho_ap", "dho_mrr", "dho_hit_at_1", "dho_regret", "dho_accuracy", "dho_mcc")


def _task_metrics(truth: np.ndarray, predictions: np.ndarray, datasets: np.ndarray) -> dict[str, float]:
    """The DHO ranking and go/no-go metrics of one MCC configuration."""
    ranking = ranking_report(truth, predictions, datasets)
    decision = decision_report(truth, predictions, datasets, thresholds=(DECISION_THRESHOLD,)).row(0, named=True)
    return {
        "dho_ap": float(np.mean(ranking["ap"].to_numpy())),
        "dho_mrr": float(np.mean(ranking["mrr"].to_numpy())),
        "dho_hit_at_1": float(np.mean(ranking["hit_at_1"].to_numpy())),
        "dho_regret": float(np.mean(ranking["regret"].to_numpy())),
        "dho_accuracy": float(decision["accuracy"]),
        "dho_mcc": float(decision["mcc"]),
    }


def evaluate(
    config: Configuration,
    selection: SelectionConfig,
    data: str | None,
    schema: Schema | None = None,
    with_tasks: bool | None = None,
) -> tuple[dict[str, Any], pl.DataFrame]:
    """One configuration: its curve, its chosen length, and that length's DHO task metrics.

    ``with_tasks`` says whether the target has go/no-go task metrics to score (a performance score in
    [-1, 1] does, a cost does not); by default it is `has_task_metrics` of the schema.
    """
    from ml_meta_perf.experiment import fit_path

    schema = schema or default_schema()
    is_mcc = has_task_metrics(schema) if with_tasks is None else with_tasks
    frame = load(data, schema)
    if schema.drop_constant_features:
        schema = drop_constant_features(frame, schema)
    truth, datasets = target(frame, schema), groups(frame, schema.dataset_column)
    fitted = fit_path(frame, schema.dataset_features, schema.model_features, config, schema.tag("E3"), schema=schema)
    curve = fitted.curve
    size, how = plateau_start(curve, delta=selection.delta, window=selection.window, smoothing=selection.smoothing)
    position = list(curve["n_terms"]).index(size)
    at = curve.row(position, named=True)
    if is_mcc and "loo_cell" in fitted.paths:
        tasks = _task_metrics(truth, fitted.paths["loo_cell"][size].predictions, datasets)
    else:
        tasks = {name: float("nan") for name in TASK_METRICS}
    row = {
        "max_abs_zscore": config.max_abs_zscore,
        "penalty": config.penalty,
        "max_arity": config.max_arity,
        "n_terms": size,
        "reached_by": how,
        "smoothed_floor": float(smoothed(floor_curve(curve), selection.smoothing)[position]),
        "floor": float(floor_curve(curve)[position]),
        **{name: float(at[name]) if name in at else float("nan") for name in JUDGED_PROTOCOLS},
        **tasks,
        "terms": "; ".join(term.name for term in fitted.equations[size].terms),
    }
    labelled = curve.with_columns(
        pl.lit(config.max_abs_zscore).alias("max_abs_zscore"),
        pl.lit(config.penalty).alias("penalty"),
        pl.lit(config.max_arity).alias("max_arity"),
    )
    return row, labelled


def choose(candidates: pl.DataFrame, sweep: SweepConfig) -> dict[str, Any]:
    """The proposed configuration: best DHO ranking AP within the readable R2 band.

    Without task metrics (a FlexFL target) the band is broken by the remaining ties alone.
    """
    readable = candidates.filter(pl.col("n_terms") <= sweep.readable_terms)
    if readable.height == 0:
        raise ValueError(f"no configuration chose an equation of at most {sweep.readable_terms} terms")
    best = float(readable["smoothed_floor"].max())  # pyright: ignore[reportArgumentType]
    band = readable.filter(pl.col("smoothed_floor") >= best - sweep.band)
    keys, descending = ["n_terms", "max_arity", "penalty"], [False, False, True]
    if not band["dho_ap"].is_nan().all():
        keys, descending = ["dho_ap", *keys], [True, *descending]
    return band.sort(keys, descending=descending).row(0, named=True)


def _worker_initialiser() -> None:
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"


def _task_summary(chosen: dict[str, Any]) -> str:
    """The DHO ranking AP of the proposal, or nothing for a target that has no task metrics."""
    value = float(chosen["dho_ap"])
    return "" if np.isnan(value) else f", DHO AP {value:.4f}"


def sweep_directory(root: str | Path, corpus: Path, schema: Schema) -> Path:
    """``<root>/<csv stem>/<target>[-tags]/sweep``: beside the result folder of the same study."""
    return Path(root) / corpus.stem / schema.slug / "sweep"


def build_parser() -> ArgumentParser:
    parser = study_parser("ml-meta-perf sweep", "Sweep the shared hyperparameters and propose a study file.")
    parser.add_argument(
        "--output", type=str, default="out", help="output directory; the sweep goes under <csv stem>/<target>/sweep"
    )
    parser.add_argument("--jobs", type=int, default=0, help="parallel workers; 0 uses every CPU")
    return parser


def run(arguments: Namespace, parser: ArgumentParser) -> int:
    try:
        study = study_of(arguments)
    except ConfigError as error:
        parser.error(str(error))
        return 2
    schema = study.schema()
    own = has_task_metrics(schema)
    points = grid(study)
    corpus = Path(arguments.data) if arguments.data else resolve_path(study.corpus.path)
    output = sweep_directory(arguments.output, corpus, schema)
    output.mkdir(parents=True, exist_ok=True)
    jobs = arguments.jobs or os.cpu_count() or 1
    print(f"{len(points)} configurations on {jobs} workers", flush=True)
    started = time.perf_counter()
    arguments_per_point = (
        points,
        itertools.repeat(study.selection),
        itertools.repeat(str(corpus)),
        itertools.repeat(schema),
        itertools.repeat(own),
    )
    # One job runs in this process: forking one whose BLAS or Polars thread pools are already
    # running (a caller that fitted before, as the test suite does) can deadlock the workers.
    if jobs == 1:
        results = list(map(evaluate, *arguments_per_point))
    else:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_initialiser) as pool:
            results = list(pool.map(evaluate, *arguments_per_point, chunksize=1))
    candidates = pl.DataFrame([row for row, _ in results]).sort(["max_abs_zscore", "penalty", "max_arity"])
    pl.concat([curve for _, curve in results]).write_csv(output / "curves.csv")
    candidates.write_csv(output / "candidates.csv")
    chosen = choose(candidates, study.sweep)
    proposed = dataclasses.replace(
        study,
        search=dataclasses.replace(
            study.search,
            max_abs_zscore=float(chosen["max_abs_zscore"]),
            penalty=float(chosen["penalty"]),
            max_arity=int(chosen["max_arity"]),
        ),
    )
    (output / "proposed_study.json").write_text(json.dumps(dataclasses.asdict(proposed), indent=2) + "\n")
    elapsed = time.perf_counter() - started
    with pl.Config(tbl_rows=20, tbl_cols=12, float_precision=3):
        leading = "dho_ap" if own else "smoothed_floor"
        print(candidates.sort(leading, descending=True).drop("terms").head(10))
    print(
        f"\nproposed for {schema.label}: z-cap {chosen['max_abs_zscore']}, penalty {chosen['penalty']}, "
        f"arity {chosen['max_arity']} -> {chosen['n_terms']} terms ({chosen['reached_by']}), "
        f"smoothed floor {chosen['smoothed_floor']:.4f}{_task_summary(chosen)}\n"
        f"written to {output} in {elapsed:.0f} s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """``ml-meta-perf-search``: the sweep as a command of its own (``ml-meta-perf sweep`` is the same)."""
    parser = build_parser()
    arguments: Namespace = parser.instantiate_classes(parser.parse_args(argv))  # pyright: ignore[reportAssignmentType]
    return run(arguments, parser)


if __name__ == "__main__":
    sys.exit(main())
