"""One Slurm job: search for the smallest equation that holds up on every test.

Nothing here is imported by the study pipeline; `python -m ml_meta_perf` does not touch it.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
from pathlib import Path

import polars as pl
from joblib import Parallel, delayed

from ml_meta_perf.data import MODEL_FEATURES, load
from ml_meta_perf.equation_search import SearchPoint, evaluate, grid

#: The two columns every subset keeps. The ablation settled them: removing `Model Capability`
#: costs 0.077 leave-one-dataset-out and removing `Processing Units Number` 0.065, while every
#: other model feature is neutral or harmful on its own. Searching without them would spend the
#: budget re-deriving a known answer.
CORE: tuple[str, ...] = ("Model Capability", "Processing Units Number")

#: The rest of `data.MODEL_FEATURES`, searched as subsets: no equation is expected to use every
#: feature, and the shortest one that holds up is the one wanted.
OPTIONAL: tuple[str, ...] = tuple(name for name in MODEL_FEATURES if name not in CORE)


def feature_subsets(minimum: int = 2, maximum: int = 6) -> list[tuple[str, ...]]:
    """Every subset that keeps the two load-bearing columns.

    `MC` and `PUN` are in every subset because the ablation settled it: removing either costs
    0.077 and 0.065 leave-one-dataset-out, while every other column is neutral or harmful on
    its own. Searching without them would spend the budget re-deriving a known answer.
    """
    out: list[tuple[str, ...]] = []
    for size in range(0, len(OPTIONAL) + 1):
        for extra in itertools.combinations(OPTIONAL, size):
            subset = tuple(sorted(CORE + extra))
            if minimum <= len(subset) <= maximum:
                out.append(subset)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, default=None, help="defaults to the packaged corpus")
    parser.add_argument("--output", type=Path, default=Path("results/cluster"))
    parser.add_argument("--name", default="equation_search")
    parser.add_argument("--jobs", type=int, default=0, help="0 = every core the job holds")
    parser.add_argument("--min-terms", type=int, default=6)
    parser.add_argument("--max-terms", type=int, default=28)
    parser.add_argument("--term-step", type=int, default=1)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    frame = load(args.frame)
    points = grid(
        features=feature_subsets(),
        penalties=(3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0),
        term_counts=tuple(range(args.min_terms, args.max_terms + 1, args.term_step)),
        zscores=(3.0, 3.5, 4.0, 4.25, 4.5, 5.0),
        arities=(2, 3),
    )
    logging.info("%d points over %d feature subsets", len(points), len(feature_subsets()))

    started = time.time()
    workers = args.jobs or -1

    def run(point: SearchPoint) -> dict[str, object] | None:
        score = evaluate(point, frame)
        return score.as_row() if score else None

    rows = Parallel(n_jobs=workers, verbose=5)(delayed(run)(point) for point in points)
    table = pl.DataFrame([row for row in rows if row]).sort("objective", descending=True)
    args.output.mkdir(parents=True, exist_ok=True)
    table.write_csv(args.output / f"{args.name}.csv")
    logging.info("%d scored points in %.0fs", table.height, time.time() - started)

    columns = [
        "n_features",
        "n_terms",
        "penalty",
        "max_abs_zscore",
        "max_arity",
        "objective",
        "in_sample_r2",
        "loo_dataset_r2",
        "loo_model_r2",
        "binary",
        "ranking",
        "stability",
    ]
    print(table.select(columns).head(15))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
