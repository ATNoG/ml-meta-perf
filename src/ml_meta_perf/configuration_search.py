"""Recalibrate the E3 configuration on a Slurm node.

The published study retained several search settings from a sweep that predates the
correction of ``Processing Units Number``.  This module repeats that experiment against the
current packaged corpus without making the expensive doubly-held-out protocol part of every
grid point.

The run has two stages.  ``sweep`` evaluates the historical composite objective over
in-sample, leave-one-dataset-out and leave-one-model-out predictions.  One beam traversal is
reused for every requested equation length.  ``validate`` then computes leave-one-cell-out
only for a defensible shortlist.  ``select`` reports a shared-configuration E3-Valid/E3-MAX
pair: E3-MAX is the best four-protocol floor found, while E3-Valid is the simplest grammar at
the same feature subset, ridge penalty and z-score whose loss is no larger than its paired
bootstrap spread.

Every intermediate result is a shard written by the parent process.  Re-running the command
skips complete shards, so a Slurm timeout does not discard completed work.  A manifest records
the data hash, grid, software versions and Git revision before the first fit starts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from joblib import Parallel, delayed

from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    DEFAULT_PATH,
    MODEL_COLUMN,
    MODEL_FEATURES,
    columns_as_arrays,
    groups,
    load,
    target,
)
from ml_meta_perf.fit import Standardizer
from ml_meta_perf.model import Equation
from ml_meta_perf.search import Selector, guided_screen, prune, search
from ml_meta_perf.selection import complexity, grammar_margin
from ml_meta_perf.stats import r2_score
from ml_meta_perf.terms import Library, build_library
from ml_meta_perf.validate import cross_validate_doubly_held_out, cross_validate_fixed_form, leave_one_group_out

SCHEMA_VERSION = 1
DEFAULT_PENALTIES = (0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0)
DEFAULT_ZSCORES = (3.0, 3.5, 4.0, 4.25, 4.5, 5.0)
DEFAULT_ARITIES = (2, 3)
DEFAULT_MIN_FEATURES = 2
DEFAULT_MAX_FEATURES = len(MODEL_FEATURES)
DEFAULT_MIN_TERMS = 6
DEFAULT_MAX_TERMS = 25
DEFAULT_POOL_SIZE = 600
DEFAULT_BEAM_WIDTH = 6
DEFAULT_SHORTLIST_TOP = 25

# The historical objective.  These values are intentionally constants rather than CLI flags:
# changing them changes the experiment rather than merely changing its execution.
OBJECTIVE_WEIGHTS: dict[str, float] = {
    "in_sample_r2": 0.10,
    "loo_dataset_r2": 0.20,
    "loo_model_r2": 0.20,
    "binary": 0.15,
    "ranking": 0.15,
    "stability": 0.15,
    "brevity": 0.05,
}
OBJECTIVE_MIN_TERMS = 6
OBJECTIVE_MAX_TERMS = 30
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
RELEVANCE_TOLERANCE = 0.01


@dataclass(frozen=True)
class SearchSettings:
    """The scientific and execution-neutral inputs that identify one sweep."""

    penalties: tuple[float, ...] = DEFAULT_PENALTIES
    zscores: tuple[float, ...] = DEFAULT_ZSCORES
    arities: tuple[int, ...] = DEFAULT_ARITIES
    minimum_features: int = DEFAULT_MIN_FEATURES
    maximum_features: int = DEFAULT_MAX_FEATURES
    minimum_terms: int = DEFAULT_MIN_TERMS
    maximum_terms: int = DEFAULT_MAX_TERMS
    pool_size: int = DEFAULT_POOL_SIZE
    beam_width: int = DEFAULT_BEAM_WIDTH
    shortlist_top: int = DEFAULT_SHORTLIST_TOP
    explicit_feature_sets: tuple[tuple[str, ...], ...] = ()

    def validate(self) -> None:
        if not self.penalties or any(value < 0.0 for value in self.penalties):
            raise ValueError("penalties must contain non-negative values")
        if not self.zscores or any(value <= 0.0 for value in self.zscores):
            raise ValueError("zscores must contain positive values")
        if not self.arities or any(value < 1 or value > 4 for value in self.arities):
            raise ValueError("arities must be between 1 and 4")
        if not 1 <= self.minimum_features <= self.maximum_features <= len(MODEL_FEATURES):
            raise ValueError("feature-count bounds must lie inside the model-feature schema")
        if not 1 <= self.minimum_terms <= self.maximum_terms:
            raise ValueError("term-count bounds must be positive and ordered")
        if self.pool_size < self.maximum_terms:
            raise ValueError("pool size must be at least the maximum equation length")
        if self.beam_width < 1 or self.shortlist_top < 1:
            raise ValueError("beam width and shortlist size must be positive")
        allowed = set(MODEL_FEATURES)
        for feature_set in self.explicit_feature_sets:
            if len(feature_set) != len(set(feature_set)):
                raise ValueError(f"feature set contains duplicates: {feature_set}")
            unknown = set(feature_set) - allowed
            if unknown:
                raise ValueError(f"unknown model features: {sorted(unknown)}")


@dataclass(frozen=True)
class LibraryPoint:
    """One term library; every ridge penalty is evaluated while it is resident in memory."""

    point_id: int
    features: tuple[str, ...]
    max_abs_zscore: float
    max_arity: int


@dataclass(frozen=True)
class BasePoint:
    """Settings shared by the two grammars in the final comparison."""

    base_id: int
    features: tuple[str, ...]
    penalty: float
    max_abs_zscore: float


@dataclass
class WorkerResult:
    """A picklable worker response; the parent alone writes files."""

    identifier: int
    rows: list[dict[str, object]]
    fold_errors: list[dict[str, object]]
    equations: dict[str, dict[str, object]]
    error: str | None = None


def feature_subsets(settings: SearchSettings) -> list[tuple[str, ...]]:
    """Every requested model-feature subset in a deterministic order."""
    settings.validate()
    if settings.explicit_feature_sets:
        return sorted({_canonical_features(value) for value in settings.explicit_feature_sets})
    subsets = [
        tuple(sorted(combination))
        for size in range(settings.minimum_features, settings.maximum_features + 1)
        for combination in itertools.combinations(MODEL_FEATURES, size)
    ]
    return sorted(subsets)


def library_points(settings: SearchSettings) -> list[LibraryPoint]:
    """Libraries crossed with z-score caps and arities; penalties reuse each library."""
    return [
        LibraryPoint(index, features, zscore, arity)
        for index, (features, zscore, arity) in enumerate(
            itertools.product(feature_subsets(settings), settings.zscores, settings.arities)
        )
    ]


def grid_table(settings: SearchSettings) -> pl.DataFrame:
    """One row per fitted configuration, before the reusable length path is expanded."""
    rows: list[dict[str, object]] = []
    configuration_id = 0
    for point in library_points(settings):
        for penalty in settings.penalties:
            rows.append(
                {
                    "configuration_id": configuration_id,
                    "library_point_id": point.point_id,
                    "features": _encode_features(point.features),
                    "n_features": len(point.features),
                    "penalty": penalty,
                    "max_abs_zscore": point.max_abs_zscore,
                    "max_arity": point.max_arity,
                }
            )
            configuration_id += 1
    return pl.DataFrame(rows)


def historical_objective(row: dict[str, float | int]) -> float:
    """The exact scalar used by the deleted 2026-09-07 configuration search."""
    parts = {
        "in_sample_r2": max(float(row["in_sample_r2"]), 0.0),
        "loo_dataset_r2": max(float(row["loo_dataset_r2"]), 0.0),
        "loo_model_r2": max(float(row["loo_model_r2"]), 0.0),
        "binary": float(row["binary"]),
        "ranking": float(row["ranking"]),
        "stability": float(row["stability"]),
        "brevity": float(row["brevity"]),
    }
    return float(sum(OBJECTIVE_WEIGHTS[name] * value for name, value in parts.items()))


def sweep(
    frame: pl.DataFrame,
    settings: SearchSettings,
    output: Path,
    *,
    jobs: int,
) -> None:
    """Evaluate every cheap grid point, resuming from one shard per library point."""
    points = library_points(settings)
    shard_directory = output / "shards" / "sweep"
    shard_directory.mkdir(parents=True, exist_ok=True)
    pending = [point for point in points if not _sweep_shard(shard_directory, point.point_id).is_file()]
    print(f"sweep: {len(points) - len(pending)} complete, {len(pending)} pending library points", flush=True)
    if not pending:
        return

    failures: list[str] = []
    parallel = Parallel(n_jobs=_normalise_jobs(jobs), verbose=10, return_as="generator_unordered")
    results = parallel(delayed(_safe_evaluate_library_point)(point, settings, frame) for point in pending)
    for item in results:
        result = cast(WorkerResult, item)
        if result.error is not None:
            failures.append(f"library point {result.identifier}: {result.error}")
            _write_json_atomic(output / "errors" / f"sweep-{result.identifier:04d}.json", asdict(result))
            continue
        _write_csv_atomic(_sweep_shard(shard_directory, result.identifier), pl.DataFrame(result.rows))
        print(f"sweep: wrote library point {result.identifier}", flush=True)
    if failures:
        raise RuntimeError("configuration sweep failures:\n" + "\n".join(failures))


def merge_sweep(settings: SearchSettings, output: Path) -> pl.DataFrame:
    """Require and merge every sweep shard."""
    shard_directory = output / "shards" / "sweep"
    paths = [_sweep_shard(shard_directory, point.point_id) for point in library_points(settings)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot merge sweep: {len(missing)} of {len(paths)} shards are missing")
    table = pl.concat([pl.read_csv(path) for path in paths], how="vertical_relaxed").sort(
        ["objective", "three_protocol_floor", "complexity"], descending=[True, True, False]
    )
    _write_csv_atomic(output / "equation_search.csv", table)
    return table


def make_shortlist(table: pl.DataFrame, settings: SearchSettings, output: Path) -> pl.DataFrame:
    """Choose base settings for the expensive protocol without trusting one scalar alone."""
    required = {
        "features",
        "penalty",
        "max_abs_zscore",
        "max_arity",
        "objective",
        "three_protocol_floor",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"sweep table is missing shortlist columns: {sorted(missing)}")

    rows = table.to_dicts()
    selected: dict[tuple[str, float, float], set[str]] = {}

    def add(row: dict[str, Any], reason: str) -> None:
        key = _base_key(row)
        selected.setdefault(key, set()).add(reason)

    for arity in settings.arities:
        matching = [row for row in rows if int(row["max_arity"]) == arity]
        for metric in ("objective", "three_protocol_floor"):
            ranked = sorted(
                matching,
                key=lambda row: (float(row[metric]), -int(row["complexity"]), -int(row["n_features"])),
                reverse=True,
            )
            for row in ranked[: settings.shortlist_top]:
                add(row, f"top-{metric}-arity-{arity}")

    for encoded in sorted({str(row["features"]) for row in rows}):
        matching = [row for row in rows if str(row["features"]) == encoded]
        for metric in ("objective", "three_protocol_floor"):
            add(max(matching, key=lambda row: float(row[metric])), f"best-{metric}-for-feature-set")

    current = tuple(
        sorted(
            (
                "Model Capability",
                "Processing Units Number",
                "Fitting Regime",
                "Loss Margin Behaviour",
            )
        )
    )
    for row in rows:
        if (
            _decode_features(str(row["features"])) == current
            and float(row["penalty"]) == 1.0
            and float(row["max_abs_zscore"]) == 5.0
        ):
            add(row, "current-published-base")
            break

    shortlist_rows = [
        {
            "base_id": index,
            "features": key[0],
            "n_features": len(_decode_features(key[0])),
            "penalty": key[1],
            "max_abs_zscore": key[2],
            "reasons": " | ".join(sorted(reasons)),
        }
        for index, (key, reasons) in enumerate(sorted(selected.items()))
    ]
    shortlist = pl.DataFrame(shortlist_rows).sort("base_id")
    _write_csv_atomic(output / "shortlist.csv", shortlist)
    return shortlist


def validate_finalists(
    frame: pl.DataFrame,
    shortlist: pl.DataFrame,
    settings: SearchSettings,
    output: Path,
    *,
    jobs: int,
) -> None:
    """Compute the doubly-held-out curve for every shortlisted shared base setting."""
    points = [
        BasePoint(
            base_id=int(row["base_id"]),
            features=_decode_features(str(row["features"])),
            penalty=float(row["penalty"]),
            max_abs_zscore=float(row["max_abs_zscore"]),
        )
        for row in shortlist.iter_rows(named=True)
    ]
    shard_directory = output / "shards" / "finalists"
    shard_directory.mkdir(parents=True, exist_ok=True)
    pending = [point for point in points if not _finalist_shard(shard_directory, point.base_id).is_file()]
    print(f"validate: {len(points) - len(pending)} complete, {len(pending)} pending base settings", flush=True)
    if not pending:
        return

    failures: list[str] = []
    parallel = Parallel(n_jobs=_normalise_jobs(jobs), verbose=10, return_as="generator_unordered")
    results = parallel(delayed(_safe_evaluate_finalist)(point, settings, frame) for point in pending)
    for item in results:
        result = cast(WorkerResult, item)
        if result.error is not None:
            failures.append(f"base setting {result.identifier}: {result.error}")
            _write_json_atomic(output / "errors" / f"finalist-{result.identifier:04d}.json", asdict(result))
            continue
        _write_json_atomic(_finalist_shard(shard_directory, result.identifier), asdict(result))
        print(f"validate: wrote base setting {result.identifier}", flush=True)
    if failures:
        raise RuntimeError("finalist validation failures:\n" + "\n".join(failures))


def merge_finalists(
    sweep_table: pl.DataFrame,
    shortlist: pl.DataFrame,
    output: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Join expensive metrics onto their cheap rows and write long-form fold errors."""
    directory = output / "shards" / "finalists"
    expected = [int(value) for value in shortlist["base_id"].to_list()]
    paths = [_finalist_shard(directory, identifier) for identifier in expected]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"cannot merge finalists: {len(missing)} of {len(paths)} shards are missing")

    cell_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cell_rows.extend(payload["rows"])
        fold_rows.extend(payload["fold_errors"])
    cell = pl.DataFrame(cell_rows)
    keys = ["features", "penalty", "max_abs_zscore", "max_arity", "requested_terms"]
    finalists = sweep_table.join(cell, on=keys, how="inner").with_columns(
        pl.min_horizontal("in_sample_r2", "loo_dataset_r2", "loo_model_r2", "loo_cell_r2").alias(
            "four_protocol_floor"
        )
    )
    finalists = finalists.sort(
        ["four_protocol_floor", "objective", "complexity"], descending=[True, True, False]
    )
    errors = pl.DataFrame(fold_rows).sort(["base_id", "max_arity", "requested_terms", "dataset"])
    _write_csv_atomic(output / "finalists.csv", finalists)
    _write_csv_atomic(output / "finalist_fold_errors.csv", errors)
    return finalists, errors


def select_equations(
    finalists: pl.DataFrame,
    fold_errors: pl.DataFrame,
    output: Path,
) -> dict[str, object]:
    """Select E3-MAX globally, then E3-Valid under exactly its shared base settings."""
    if finalists.is_empty():
        raise ValueError("no finalists are available for selection")
    ranked = finalists.sort(
        ["four_protocol_floor", "complexity", "n_features", "requested_terms"],
        descending=[True, False, False, False],
    )
    maximum = ranked.row(0, named=True)
    shared = finalists.filter(
        (pl.col("features") == maximum["features"])
        & (pl.col("penalty") == maximum["penalty"])
        & (pl.col("max_abs_zscore") == maximum["max_abs_zscore"])
    )
    candidates: list[dict[str, Any]] = []
    for arity in sorted(set(int(value) for value in shared["max_arity"].to_list())):
        per_arity = shared.filter(pl.col("max_arity") == arity).sort(
            ["four_protocol_floor", "complexity", "requested_terms"], descending=[True, False, False]
        )
        candidates.append(per_arity.row(0, named=True))

    reference_errors = _candidate_errors(fold_errors, maximum)
    valid: dict[str, Any] | None = None
    margin_rows: list[dict[str, object]] = []
    for candidate in sorted(
        candidates,
        key=lambda row: (complexity(int(row["max_arity"]), int(row["n_terms"])), int(row["n_terms"])),
    ):
        gain, scale, ratio = grammar_margin(_candidate_errors(fold_errors, candidate), reference_errors)
        margin_rows.append(
            {
                "max_arity": int(candidate["max_arity"]),
                "requested_terms": int(candidate["requested_terms"]),
                "n_terms": int(candidate["n_terms"]),
                "gain": gain,
                "bootstrap_scale": scale,
                "gain_to_scale": ratio,
                "eligible_as_valid": ratio <= 1.0,
            }
        )
        if ratio <= 1.0 and valid is None:
            valid = candidate
    if valid is None:
        valid = maximum

    independent_valid = _simplest_eligible(finalists, fold_errors, maximum)
    payload: dict[str, object] = {
        "selection": (
            "E3-MAX maximises the four-protocol R2 floor; E3-Valid shares its base settings "
            "and uses gain/paired bootstrap spread <= 1."
        ),
        "e3_valid": _public_candidate(valid),
        "e3_max": _public_candidate(maximum),
        "independent_valid_diagnostic": _public_candidate(independent_valid),
        "shared_grammar_margins": margin_rows,
    }
    _write_json_atomic(output / "selected_configurations.json", payload)
    _write_csv_atomic(output / "shared_grammar_margins.csv", pl.DataFrame(margin_rows))

    shards = output / "shards" / "finalists"
    for label, row in (("e3_valid", valid), ("e3_max", maximum)):
        equation = _load_candidate_equation(shards, row)
        equation.save(output / f"{label}.json")
        (output / f"{label}.txt").write_text(str(equation) + "\n", encoding="utf-8")
    return payload


def initialise(output: Path, data_path: Path, settings: SearchSettings) -> dict[str, object]:
    """Write or verify the immutable manifest and the expanded configuration grid."""
    settings.validate()
    output.mkdir(parents=True, exist_ok=True)
    digest = _file_hash(data_path)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "data_sha256": digest,
        "source_sha256": _source_hash(),
        "settings": _settings_payload(settings),
        "objective_weights": OBJECTIVE_WEIGHTS,
        "objective_brevity_range": [OBJECTIVE_MIN_TERMS, OBJECTIVE_MAX_TERMS],
    }
    search_id = hashlib.sha256(_stable_json(identity).encode()).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("search_id") != search_id:
            raise RuntimeError(
                "output directory belongs to a different search; choose another directory or remove it explicitly"
            )
        return existing

    git_status = _git_value("status", "--porcelain")
    manifest: dict[str, object] = identity | {
        "search_id": search_id,
        "data_path": str(data_path),
        "git_revision": _git_value("rev-parse", "HEAD"),
        "git_dirty": None if git_status == "unavailable" else bool(git_status),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in ("ml-meta-perf", "numpy", "polars", "joblib")},
        "created_unix": time.time(),
    }
    _write_json_atomic(manifest_path, manifest)
    _write_csv_atomic(output / "grid.csv", grid_table(settings))
    return manifest


def settings_from_arguments(arguments: argparse.Namespace) -> SearchSettings:
    """Turn repeatable CLI arguments into one canonical settings object."""
    explicit = tuple(_parse_feature_set(value) for value in arguments.feature_set)
    return SearchSettings(
        penalties=_unique(arguments.penalty or DEFAULT_PENALTIES),
        zscores=_unique(arguments.zscore or DEFAULT_ZSCORES),
        arities=_unique(arguments.arity or DEFAULT_ARITIES),
        minimum_features=arguments.min_features,
        maximum_features=arguments.max_features,
        minimum_terms=arguments.min_terms,
        maximum_terms=arguments.max_terms,
        pool_size=arguments.pool,
        beam_width=arguments.beam,
        shortlist_top=arguments.shortlist_top,
        explicit_feature_sets=explicit,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-meta-perf-search",
        description="Resumable two-stage Slurm search for E3-Valid and E3-MAX.",
    )
    parser.add_argument("stage", choices=("plan", "sweep", "merge", "shortlist", "validate", "select", "all"))
    parser.add_argument("--data", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output", type=Path, default=Path("results/configuration_search"))
    parser.add_argument("--jobs", type=int, default=0, help="parallel workers; 0 uses every available CPU")
    parser.add_argument(
        "--penalty", type=float, action="append", help="ridge penalty; repeat to replace the default grid"
    )
    parser.add_argument(
        "--zscore", type=float, action="append", help="term z-score cap; repeat to replace the default grid"
    )
    parser.add_argument(
        "--arity", type=int, action="append", help="maximum term arity; repeat to replace the default grid"
    )
    parser.add_argument(
        "--feature-set",
        action="append",
        default=[],
        help="comma-separated model features; repeat to replace the combinatorial subset grid",
    )
    parser.add_argument("--min-features", type=int, default=DEFAULT_MIN_FEATURES)
    parser.add_argument("--max-features", type=int, default=DEFAULT_MAX_FEATURES)
    parser.add_argument("--min-terms", type=int, default=DEFAULT_MIN_TERMS)
    parser.add_argument("--max-terms", type=int, default=DEFAULT_MAX_TERMS)
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--beam", type=int, default=DEFAULT_BEAM_WIDTH)
    parser.add_argument("--shortlist-top", type=int, default=DEFAULT_SHORTLIST_TOP)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    started_unix = time.time()
    started = time.perf_counter()
    arguments = build_parser().parse_args(argv)
    settings = settings_from_arguments(arguments)
    data_path = Path(arguments.data).resolve()
    output = Path(arguments.output).resolve()
    manifest = initialise(output, data_path, settings)
    if arguments.stage == "plan":
        print(f"planned {grid_table(settings).height} configurations in {output}")
        _write_run_summary(output, arguments.stage, manifest, started_unix, started)
        return 0
    frame: pl.DataFrame | None = None
    sweep_table: pl.DataFrame | None = None
    shortlist: pl.DataFrame | None = None
    if arguments.stage in {"sweep", "validate", "all"}:
        frame = load(data_path)
    if arguments.stage in {"sweep", "all"}:
        assert frame is not None
        sweep(frame, settings, output, jobs=arguments.jobs)
    if arguments.stage in {"merge", "shortlist", "validate", "select", "all"}:
        sweep_table = merge_sweep(settings, output)
    if arguments.stage in {"shortlist", "validate", "select", "all"}:
        assert sweep_table is not None
        shortlist = make_shortlist(sweep_table, settings, output)
    if arguments.stage in {"validate", "all"}:
        assert frame is not None and shortlist is not None
        validate_finalists(frame, shortlist, settings, output, jobs=arguments.jobs)
    if arguments.stage in {"select", "all"}:
        assert sweep_table is not None and shortlist is not None
        finalists, fold_errors = merge_finalists(sweep_table, shortlist, output)
        selected = select_equations(finalists, fold_errors, output)
        print(json.dumps(selected, indent=2), flush=True)
    _write_run_summary(output, arguments.stage, manifest, started_unix, started)
    return 0


def _safe_evaluate_library_point(
    point: LibraryPoint,
    settings: SearchSettings,
    frame: pl.DataFrame,
) -> WorkerResult:
    try:
        return WorkerResult(point.point_id, _evaluate_library_point(point, settings, frame), [], {})
    except Exception as error:
        return WorkerResult(point.point_id, [], [], {}, f"{type(error).__name__}: {error}")


def _evaluate_library_point(
    point: LibraryPoint,
    settings: SearchSettings,
    frame: pl.DataFrame,
) -> list[dict[str, object]]:
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)
    columns = columns_as_arrays(frame, DATASET_FEATURES + point.features)
    library = build_library(
        DATASET_FEATURES,
        point.features,
        columns,
        max_abs_zscore=point.max_abs_zscore,
        max_arity=point.max_arity,
    )
    rows: list[dict[str, object]] = []
    for penalty in settings.penalties:
        outcome = search(
            library,
            truth,
            max_terms=settings.maximum_terms,
            penalty=penalty,
            pool_size=settings.pool_size,
            beam_width=settings.beam_width,
            name="E3",
        )
        equations = _pruned_path(outcome.equations, columns, truth, penalty, settings.minimum_terms)
        validation_library = _equation_library(equations, columns)
        lodo = cross_validate_fixed_form(validation_library, columns, truth, datasets, equations, penalty=penalty)
        lomo = cross_validate_fixed_form(validation_library, columns, truth, models, equations, penalty=penalty)
        selections = _fold_selection_paths(
            library,
            truth,
            datasets,
            penalty=penalty,
            pool_size=settings.pool_size,
            beam_width=settings.beam_width,
            maximum_terms=settings.maximum_terms,
        )
        for requested_terms, equation in equations.items():
            if requested_terms not in lodo or requested_terms not in lomo:
                continue
            prediction = equation.predict(columns)
            dataset_prediction = lodo[requested_terms].predictions
            binary_accuracy, binary_f1, binary_map = _binary_scores(truth, dataset_prediction, datasets)
            ranking_map, ranking_mrr, ranking_hit1, ranking_regret = _ranking_scores(
                truth, dataset_prediction, datasets
            )
            binary = (binary_accuracy + binary_f1 + binary_map) / 3.0
            ranking = (ranking_map + ranking_mrr + ranking_hit1 + (1.0 - min(ranking_regret, 1.0))) / 4.0
            stability = _stability(equation, selections.get(requested_terms, []))
            brevity = float(
                np.clip(
                    (OBJECTIVE_MAX_TERMS - equation.n_terms) / (OBJECTIVE_MAX_TERMS - OBJECTIVE_MIN_TERMS),
                    0.0,
                    1.0,
                )
            )
            metrics: dict[str, float | int] = {
                "in_sample_r2": r2_score(truth, prediction),
                "loo_dataset_r2": lodo[requested_terms].scores(truth).r2,
                "loo_model_r2": lomo[requested_terms].scores(truth).r2,
                "binary": binary,
                "ranking": ranking,
                "stability": stability,
                "brevity": brevity,
            }
            dispersion = lodo[requested_terms].dispersion()
            rows.append(
                {
                    "library_point_id": point.point_id,
                    "features": _encode_features(point.features),
                    "n_features": len(point.features),
                    "penalty": penalty,
                    "max_abs_zscore": point.max_abs_zscore,
                    "max_arity": point.max_arity,
                    "requested_terms": requested_terms,
                    "n_terms": equation.n_terms,
                    "complexity": complexity(point.max_arity, equation.n_terms),
                    **metrics,
                    "binary_accuracy": binary_accuracy,
                    "binary_f1": binary_f1,
                    "binary_map": binary_map,
                    "ranking_map": ranking_map,
                    "ranking_mrr": ranking_mrr,
                    "ranking_hit1": ranking_hit1,
                    "ranking_regret1": ranking_regret,
                    "three_protocol_floor": min(
                        float(metrics["in_sample_r2"]),
                        float(metrics["loo_dataset_r2"]),
                        float(metrics["loo_model_r2"]),
                    ),
                    "objective": historical_objective(metrics),
                    "median_fold_r2": dispersion["median_fold_r2"],
                    "worst_fold_r2": dispersion["worst_fold_r2"],
                    "median_fold_mae": dispersion["median_fold_mae"],
                    "worst_fold_mae": dispersion["worst_fold_mae"],
                    "terms": json.dumps([term.name for term in equation.terms], ensure_ascii=False),
                }
            )
    return rows


def _safe_evaluate_finalist(
    point: BasePoint,
    settings: SearchSettings,
    frame: pl.DataFrame,
) -> WorkerResult:
    try:
        return _evaluate_finalist(point, settings, frame)
    except Exception as error:
        return WorkerResult(point.base_id, [], [], {}, f"{type(error).__name__}: {error}")


def _evaluate_finalist(point: BasePoint, settings: SearchSettings, frame: pl.DataFrame) -> WorkerResult:
    truth = target(frame)
    datasets = groups(frame, DATASET_COLUMN)
    models = groups(frame, MODEL_COLUMN)
    columns = columns_as_arrays(frame, DATASET_FEATURES + point.features)
    rows: list[dict[str, object]] = []
    fold_errors: list[dict[str, object]] = []
    equations_payload: dict[str, dict[str, object]] = {}
    for arity in settings.arities:
        library = build_library(
            DATASET_FEATURES,
            point.features,
            columns,
            max_abs_zscore=point.max_abs_zscore,
            max_arity=arity,
        )
        outcome = search(
            library,
            truth,
            max_terms=settings.maximum_terms,
            penalty=point.penalty,
            pool_size=settings.pool_size,
            beam_width=settings.beam_width,
            name="E3",
        )
        equations = _pruned_path(outcome.equations, columns, truth, point.penalty, settings.minimum_terms)
        validation_library = _equation_library(equations, columns)
        cell = cross_validate_doubly_held_out(
            validation_library,
            columns,
            truth,
            datasets,
            models,
            equations,
            penalty=point.penalty,
        )
        for requested_terms, equation in equations.items():
            if requested_terms not in cell:
                continue
            result = cell[requested_terms]
            rows.append(
                {
                    "base_id": point.base_id,
                    "features": _encode_features(point.features),
                    "penalty": point.penalty,
                    "max_abs_zscore": point.max_abs_zscore,
                    "max_arity": arity,
                    "requested_terms": requested_terms,
                    "loo_cell_r2": result.scores(truth).r2,
                    "loo_cell_mae": result.scores(truth).mae,
                    "loo_cell_clipped": result.clipped,
                }
            )
            key = _equation_key(arity, requested_terms)
            equations_payload[key] = equation.to_dict()
            for dataset, scores in sorted(result.per_fold.items()):
                fold_errors.append(
                    {
                        "base_id": point.base_id,
                        "features": _encode_features(point.features),
                        "penalty": point.penalty,
                        "max_abs_zscore": point.max_abs_zscore,
                        "max_arity": arity,
                        "requested_terms": requested_terms,
                        "dataset": dataset,
                        "mae": scores.mae,
                    }
                )
    return WorkerResult(point.base_id, rows, fold_errors, equations_payload)


def _pruned_path(
    equations: dict[int, Equation],
    columns: dict[str, np.ndarray],
    truth: np.ndarray,
    penalty: float,
    minimum_terms: int,
) -> dict[int, Equation]:
    path = {
        size: prune(equation, columns, truth, penalty=penalty)
        for size, equation in equations.items()
        if size >= minimum_terms
    }
    return {size: equation for size, equation in path.items() if equation.terms}


def _equation_library(equations: dict[int, Equation], columns: dict[str, np.ndarray]) -> Library:
    terms = []
    seen: set[str] = set()
    for equation in equations.values():
        for term in equation.terms:
            if term.name not in seen:
                seen.add(term.name)
                terms.append(term)
    clone = object.__new__(Library)
    clone.terms = terms
    with np.errstate(all="ignore"):
        clone.matrix = np.column_stack([term.evaluate(columns) for term in terms])
    return clone


def _fold_selection_paths(
    library: Library,
    truth: np.ndarray,
    labels: np.ndarray,
    *,
    penalty: float,
    pool_size: int,
    beam_width: int,
    maximum_terms: int,
) -> dict[int, list[set[str]]]:
    paths: dict[int, list[set[str]]] = {}
    for _, train, _ in leave_one_group_out(labels):
        view = _library_view(library, train)
        standardizer = Standardizer.fit(view.matrix)
        selector = Selector(
            standardizer.apply(view.matrix),
            truth[train],
            penalty,
            library.feature_groups,
        )
        pool = guided_screen(view, truth[train], keep=pool_size)
        subsets = selector.search(pool, maximum_terms, beam_width=beam_width)
        for size, subset in subsets.items():
            paths.setdefault(size, []).append({library.terms[index].name for index in subset.indices})
    return paths


def _library_view(library: Library, mask: np.ndarray) -> Library:
    clone = object.__new__(Library)
    clone.terms = library.terms
    clone.matrix = library.matrix[mask]
    return clone


def _stability(equation: Equation, selections: list[set[str]]) -> float:
    if not equation.terms or not selections:
        return 0.0
    return float(
        np.mean(
            [sum(term.name in selected for selected in selections) / len(selections) for term in equation.terms]
        )
    )


def _binary_scores(truth: np.ndarray, prediction: np.ndarray, datasets: np.ndarray) -> tuple[float, float, float]:
    accuracies: list[float] = []
    f1s: list[float] = []
    maps: list[float] = []
    for threshold in THRESHOLDS:
        actual = truth >= threshold
        predicted = prediction >= threshold
        true_positive = float((actual & predicted).sum())
        true_negative = float((~actual & ~predicted).sum())
        false_positive = float((~actual & predicted).sum())
        false_negative = float((actual & ~predicted).sum())
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        accuracies.append((true_positive + true_negative) / truth.size)
        f1s.append(2.0 * precision * recall / max(precision + recall, 1e-12))
        per_dataset = [
            _average_precision(truth[datasets == label] >= threshold, prediction[datasets == label])
            for label in np.unique(datasets)
        ]
        usable = [value for value in per_dataset if not np.isnan(value)]
        maps.append(float(np.mean(usable)) if usable else 0.0)
    return float(np.mean(accuracies)), float(np.mean(f1s)), float(np.mean(maps))


def _ranking_scores(
    truth: np.ndarray,
    prediction: np.ndarray,
    datasets: np.ndarray,
) -> tuple[float, float, float, float]:
    maps: list[float] = []
    reciprocal_ranks: list[float] = []
    hits: list[float] = []
    regrets: list[float] = []
    for label in np.unique(datasets):
        mask = datasets == label
        actual = truth[mask]
        predicted = prediction[mask]
        if actual.size < 3 or float(actual.max() - actual.min()) < 1e-12:
            continue
        relevant = actual >= actual.max() - RELEVANCE_TOLERANCE
        order = np.argsort(-predicted, kind="stable")
        maps.append(_average_precision(relevant, predicted))
        ranks = [position for position, index in enumerate(order, 1) if relevant[index]]
        reciprocal_ranks.append(1.0 / ranks[0] if ranks else 0.0)
        hits.append(float(relevant[order[0]]))
        regrets.append(float(actual.max() - actual[order[0]]))
    return (
        float(np.mean(maps)),
        float(np.mean(reciprocal_ranks)),
        float(np.mean(hits)),
        float(np.mean(regrets)),
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    if not labels.any():
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(labels[order])
    precision = hits / np.arange(1, labels.size + 1)
    return float((precision * labels[order]).sum() / labels.sum())


def _candidate_errors(table: pl.DataFrame, candidate: dict[str, Any]) -> np.ndarray:
    selected = table.filter(
        (pl.col("base_id") == int(candidate["base_id"]))
        & (pl.col("max_arity") == int(candidate["max_arity"]))
        & (pl.col("requested_terms") == int(candidate["requested_terms"]))
    ).sort("dataset")
    if selected.is_empty():
        raise ValueError("candidate has no per-dataset cell errors")
    return selected["mae"].to_numpy()


def _simplest_eligible(
    finalists: pl.DataFrame,
    fold_errors: pl.DataFrame,
    maximum: dict[str, Any],
) -> dict[str, Any]:
    reference = _candidate_errors(fold_errors, maximum)
    rows = finalists.sort(["complexity", "n_features", "n_terms"]).to_dicts()
    for row in rows:
        errors = _candidate_errors(fold_errors, row)
        _, _, ratio = grammar_margin(errors, reference)
        if ratio <= 1.0:
            return row
    return maximum


def _public_candidate(candidate: dict[str, Any]) -> dict[str, object]:
    names = (
        "base_id",
        "features",
        "n_features",
        "penalty",
        "max_abs_zscore",
        "max_arity",
        "requested_terms",
        "n_terms",
        "complexity",
        "objective",
        "in_sample_r2",
        "loo_dataset_r2",
        "loo_model_r2",
        "loo_cell_r2",
        "four_protocol_floor",
        "stability",
        "terms",
    )
    return {name: candidate[name] for name in names}


def _load_candidate_equation(directory: Path, candidate: dict[str, Any]) -> Equation:
    path = _finalist_shard(directory, int(candidate["base_id"]))
    payload = json.loads(path.read_text(encoding="utf-8"))
    key = _equation_key(int(candidate["max_arity"]), int(candidate["requested_terms"]))
    return Equation.from_dict(payload["equations"][key])


def _base_key(row: dict[str, Any]) -> tuple[str, float, float]:
    return str(row["features"]), float(row["penalty"]), float(row["max_abs_zscore"])


def _equation_key(arity: int, requested_terms: int) -> str:
    return f"arity-{arity}-k-{requested_terms}"


def _canonical_features(features: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(str(value).strip() for value in features if str(value).strip()))


def _encode_features(features: tuple[str, ...]) -> str:
    return json.dumps(features, ensure_ascii=False, separators=(",", ":"))


def _decode_features(value: str) -> tuple[str, ...]:
    payload = json.loads(value)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        raise ValueError(f"malformed feature set: {value}")
    return _canonical_features(payload)


def _parse_feature_set(value: str) -> tuple[str, ...]:
    return _canonical_features(value.split(","))


def _unique(values: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def _settings_payload(settings: SearchSettings) -> dict[str, object]:
    payload = asdict(settings)
    payload["penalties"] = list(settings.penalties)
    payload["zscores"] = list(settings.zscores)
    payload["arities"] = list(settings.arities)
    payload["explicit_feature_sets"] = [list(value) for value in settings.explicit_feature_sets]
    return payload


def _stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hash() -> str:
    """Hash the complete Python package so shards cannot outlive their implementation."""
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def _git_value(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unavailable"


def _write_run_summary(
    output: Path,
    stage: str,
    manifest: dict[str, object],
    started_unix: float,
    started: float,
) -> None:
    completed_unix = time.time()
    _write_json_atomic(
        output / "run_summary.json",
        {
            "search_id": manifest["search_id"],
            "stage": stage,
            "started_unix": started_unix,
            "completed_unix": completed_unix,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def _normalise_jobs(jobs: int) -> int:
    return jobs if jobs != 0 else -1


def _sweep_shard(directory: Path, identifier: int) -> Path:
    return directory / f"sweep-{identifier:04d}.csv"


def _finalist_shard(directory: Path, identifier: int) -> Path:
    return directory / f"finalist-{identifier:04d}.json"


def _write_csv_atomic(path: Path, table: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    table.write_csv(temporary)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
