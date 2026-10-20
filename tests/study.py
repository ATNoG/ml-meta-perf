"""The MCC study's data description as the tests read it: from ``config/study.json``.

The tests used to import these names from ``ml_meta_perf.data``, where they were constants. They
are the shipped study file's columns and features now, so a test that needs "the dataset features"
asks the file, as the code does.
"""

import dataclasses
import json
from pathlib import Path

from ml_meta_perf.config import RowFilter, Schema, StudyConfig, default_schema, load_config

SCHEMA = default_schema()
DATASET_COLUMN = SCHEMA.dataset_column
MODEL_COLUMN = SCHEMA.model_column
TARGET_COLUMN = SCHEMA.target_column
DATASET_FEATURES = SCHEMA.dataset_features
MODEL_FEATURES = SCHEMA.model_features
ALL_FEATURES = SCHEMA.features

# --- FlexFL: one complete study file per target, in config/flexfl/ -------------------------------

FLEXFL_DIR = Path(__file__).resolve().parents[1] / "config" / "flexfl"
FLEXFL_TARGETS = (
    "performance",
    "total_time_s",
    "comm_bytes_total",
    "n_epochs",
    "compute_time_total_s",
    "compute_time_max_s",
    "comm_time_total_s",
    "validation_time_s",
)
FLEXFL_COST_TARGETS = (
    "total_time_s",
    "comm_bytes_total",
    "compute_time_total_s",
    "compute_time_max_s",
    "comm_time_total_s",
    "validation_time_s",
)
FLEXFL_DECOMPOSITION_TARGETS = ("compute_time_total_s", "compute_time_max_s", "comm_time_total_s", "validation_time_s")
FLEXFL_EPOCH_CAP_COLUMN = "epoch_cap"


def flexfl_path(target: str, task_type: str | None = None) -> Path:
    """The study file of a FlexFL target (``performance`` needs its task type)."""
    name = target if task_type is None or target != "performance" else f"{target}-{task_type}"
    path = FLEXFL_DIR / f"{name}.json"
    if not path.is_file():
        raise ValueError(
            f"no FlexFL study file {path.name}: targets are {FLEXFL_TARGETS}, performance needs a task type"
        )
    return path


def flexfl_study(target: str, task_type: str | None = None) -> StudyConfig:
    return load_config(flexfl_path(target, task_type))


def flexfl_schema(target: str, task_type: str | None = None, log_target: bool = False) -> Schema:
    """The schema of a FlexFL study file; a task type on another target filters its rows, as FlexFL always did."""
    schema = flexfl_study(target, task_type).schema()
    if task_type is not None and target != "performance":
        schema = dataclasses.replace(
            schema, row_filter=RowFilter("is_classification", task_type == "classification", label=task_type)
        )
    return dataclasses.replace(schema, transform="log1p") if log_target else schema


def flexfl_argv(target: str, task_type: str | None = None, log_target: bool = False) -> list[str]:
    """The command-line flags that select a FlexFL study file (and its log variant)."""
    argv = ["--config", str(flexfl_path(target, task_type))]
    if task_type is not None and target != "performance":
        argv += [
            "--corpus.row_filter",
            json.dumps({"column": "is_classification", "equals": task_type == "classification", "label": task_type}),
        ]
    if log_target:
        argv += ["--response.transform", "log1p"]
    return argv


FLEXFL_MODEL_FEATURES = flexfl_study("total_time_s").features.model
