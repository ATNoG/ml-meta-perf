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
from jsonargparse import ActionConfigFile, ArgumentParser  # pyright: ignore[reportPrivateImportUsage]

from ml_meta_perf.config import (
    DEFAULT_CONFIG_PATH,
    Configuration,
    OpaqueConfig,
    SelectionConfig,
    StudyConfig,
    SweepConfig,
)
from ml_meta_perf.data import DATASET_COLUMN, DATASET_FEATURES, MODEL_FEATURES, groups, load, target
from ml_meta_perf.selection import floor_curve, plateau_start, smoothed
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


def evaluate(
    config: Configuration, selection: SelectionConfig, data: str | None
) -> tuple[dict[str, Any], pl.DataFrame]:
    """One configuration: its curve, its chosen length, and that length's DHO task metrics."""
    from ml_meta_perf.experiment import fit_path

    frame = load(data)
    truth, datasets = target(frame), groups(frame, DATASET_COLUMN)
    fitted = fit_path(frame, DATASET_FEATURES, MODEL_FEATURES, config, "E3")
    curve = fitted.curve
    size, how = plateau_start(curve, delta=selection.delta, window=selection.window, smoothing=selection.smoothing)
    position = list(curve["n_terms"]).index(size)
    at = curve.row(position, named=True)
    predictions = fitted.paths["loo_cell"][size].predictions
    ranking = ranking_report(truth, predictions, datasets)
    decision = decision_report(truth, predictions, datasets, thresholds=(DECISION_THRESHOLD,)).row(0, named=True)
    row = {
        "max_abs_zscore": config.max_abs_zscore,
        "penalty": config.penalty,
        "max_arity": config.max_arity,
        "n_terms": size,
        "reached_by": how,
        "smoothed_floor": float(smoothed(floor_curve(curve), selection.smoothing)[position]),
        "floor": float(floor_curve(curve)[position]),
        **{name: float(at[name]) for name in ("r2_in_sample", "r2_loo_dataset", "r2_loo_model", "r2_loo_cell")},
        "dho_ap": float(np.mean(ranking["ap"].to_numpy())),
        "dho_mrr": float(np.mean(ranking["mrr"].to_numpy())),
        "dho_hit_at_1": float(np.mean(ranking["hit_at_1"].to_numpy())),
        "dho_regret": float(np.mean(ranking["regret"].to_numpy())),
        "dho_accuracy": float(decision["accuracy"]),
        "dho_mcc": float(decision["mcc"]),
        "terms": "; ".join(term.name for term in fitted.equations[size].terms),
    }
    labelled = curve.with_columns(
        pl.lit(config.max_abs_zscore).alias("max_abs_zscore"),
        pl.lit(config.penalty).alias("penalty"),
        pl.lit(config.max_arity).alias("max_arity"),
    )
    return row, labelled


def choose(candidates: pl.DataFrame, sweep: SweepConfig) -> dict[str, Any]:
    """The proposed configuration: best DHO ranking AP within the readable R2 band."""
    readable = candidates.filter(pl.col("n_terms") <= sweep.readable_terms)
    if readable.height == 0:
        raise ValueError(f"no configuration chose an equation of at most {sweep.readable_terms} terms")
    best = float(readable["smoothed_floor"].max())  # pyright: ignore[reportArgumentType]
    band = readable.filter(pl.col("smoothed_floor") >= best - sweep.band)
    ranked = band.sort(["dho_ap", "n_terms", "max_arity", "penalty"], descending=[True, False, False, True])
    return ranked.row(0, named=True)


def _worker_initialiser() -> None:
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ml-meta-perf-search",
        description="Sweep the shared hyperparameters and propose config/study.json.",
        default_config_files=[str(DEFAULT_CONFIG_PATH)],
    )
    parser.add_argument("--config", action=ActionConfigFile, help="study configuration JSON")
    parser.add_argument("--search", type=Configuration, help="the base configuration the grid varies")
    parser.add_argument("--selection", type=SelectionConfig, help="the length rule applied to every point")
    parser.add_argument("--opaque", type=OpaqueConfig, help="carried into the proposal unchanged")
    parser.add_argument("--sweep", type=SweepConfig, help="the grid and the choice among close candidates")
    parser.add_argument("--data", type=str | None, default=None, help="meta-dataset CSV")
    parser.add_argument("--output", type=str, default="results/sweep", help="directory for the sweep outputs")
    parser.add_argument("--jobs", type=int, default=0, help="parallel workers; 0 uses every CPU")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.instantiate(parser.parse_args(argv))
    study = StudyConfig(
        search=arguments.search, selection=arguments.selection, opaque=arguments.opaque, sweep=arguments.sweep
    )
    points = grid(study)
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    jobs = arguments.jobs or os.cpu_count() or 1
    print(f"{len(points)} configurations on {jobs} workers", flush=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_worker_initialiser) as pool:
        results = list(
            pool.map(evaluate, points, itertools.repeat(study.selection), itertools.repeat(arguments.data), chunksize=1)
        )
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
        print(candidates.sort("dho_ap", descending=True).drop("terms").head(10))
    print(
        f"\nproposed: z-cap {chosen['max_abs_zscore']}, penalty {chosen['penalty']}, arity {chosen['max_arity']} "
        f"-> {chosen['n_terms']} terms ({chosen['reached_by']}), smoothed floor {chosen['smoothed_floor']:.4f}, "
        f"DHO AP {chosen['dho_ap']:.4f}\nwritten to {output} in {elapsed:.0f} s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
