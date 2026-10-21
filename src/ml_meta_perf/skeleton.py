"""``ml-meta-perf init``: a study file for a CSV, to be edited.

Given a corpus and the names of its group columns and its response, the skeleton lists every
other numeric column as a feature: a column that takes one value within each dataset describes the
dataset, one that varies within a dataset describes the model (or the configuration the run used).
Columns that never vary carry nothing an equation can use and are left out and reported. Every tuned
value is copied from ``config/mcc.json``, so the skeleton is a complete study file that runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ml_meta_perf.config import DEFAULT_CONFIG_PATH, E3_MAX, TRANSFORMS, ConfigError, parse_config


def skeleton(
    corpus: Path,
    *,
    dataset_column: str,
    response: str,
    model_column: str | None = None,
    bounds: tuple[float | None, float | None] | None = None,
    transform: str = "none",
) -> tuple[dict[str, object], dict[str, list[str]]]:
    """The study file as a dict, and the columns it left out by reason (``constant``, ``not numeric``)."""
    frame = pl.read_csv(corpus)
    named = [name for name in (dataset_column, model_column, response) if name is not None]
    missing = [name for name in named if name not in frame.columns]
    if missing:
        raise ConfigError(f"{corpus.name} has no column {missing}; its columns are {frame.columns}")
    if transform not in TRANSFORMS:
        raise ConfigError(f"transform must be one of {TRANSFORMS}, got {transform!r}")
    left_out: dict[str, list[str]] = {"constant": [], "not numeric": []}
    dataset: list[str] = []
    model: list[str] = []
    for name in frame.columns:
        if name in named:
            continue
        if not frame[name].dtype.is_numeric():
            left_out["not numeric"].append(name)
        elif frame[name].n_unique() < 2:
            left_out["constant"].append(name)
        else:
            per_dataset = frame.group_by(dataset_column).agg(pl.col(name).n_unique().alias("n")).get_column("n")
            (dataset if int(per_dataset.max()) == 1 else model).append(name)  # pyright: ignore[reportArgumentType]
    study = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    study["corpus"] = {"path": str(corpus), "dataset_column": dataset_column, "model_column": model_column}
    study["response"] = {"column": response, "bounds": None if bounds is None else list(bounds), "transform": transform}
    study["features"] = {"dataset": dataset, "model": model}
    study["protocols"] = ["IS", "LODO", *(["LOMO", "DHO"] if model_column else [])]
    study["equations"] = ["E1", "E2", "E3", E3_MAX] if model_column else ["E3", E3_MAX]
    study["grammar"] = {"exclude": {}}
    study["selection"]["method"] = "pareto_knee"
    parse_config(study)
    return study, left_out
