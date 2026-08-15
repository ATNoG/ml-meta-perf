"""Loading and shaping of the meta-dataset.

The meta-dataset has one row per (dataset, model) pair. Two groups of columns matter
and they behave very differently, which is why they are named separately here:

* dataset features are constant across every row of a given dataset, so on their own
  they can only ever predict a per-dataset constant;
* model features vary with the model, and three of them (the operation counts) also
  vary with the dataset, since they are functions of the dataset size.

That asymmetry is the whole point of the two-equation comparison, so the split is
part of the public API rather than something each caller re-derives.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DATASET_COLUMN = "Dataset"
MODEL_COLUMN = "Model"
TARGET_COLUMN = "MCC"

DATASET_FEATURES: tuple[str, ...] = (
    "class_ent",
    "eq_num_attr",
    "gravity",
    "inst_to_attr",
    "nr_attr",
    "nr_bin",
    "nr_class",
    "nr_cor_attr",
    "nr_inst",
    "nr_norm",
    "nr_outliers",
    "ns_ratio",
)

MODEL_FEATURES: tuple[str, ...] = (
    "Processing Units Number",
    "Training Operations",
    "Prediction Operations",
    "Active Regularization Mechanisms",
    "Robust to Outliers",
)

ALL_FEATURES: tuple[str, ...] = DATASET_FEATURES + MODEL_FEATURES

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "meta_dataset.csv"


class SchemaError(ValueError):
    """The CSV does not carry the columns the meta-model needs."""


def load(path: str | Path | None = None) -> pl.DataFrame:
    """Read the meta-dataset and check it has the expected shape.

    Every feature is cast to Float64. The integer-valued columns (``nr_attr``,
    ``nr_class``, ...) are still continuous as far as the equations are concerned,
    and carrying two dtypes through the term library buys nothing.
    """
    resolved = Path(path) if path is not None else DEFAULT_PATH
    if not resolved.is_file():
        raise SchemaError(f"meta-dataset not found: {resolved}")

    frame = pl.read_csv(resolved)
    expected = (DATASET_COLUMN, MODEL_COLUMN, *ALL_FEATURES, TARGET_COLUMN)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    frame = frame.select(expected).with_columns(
        [pl.col(column).cast(pl.Float64) for column in (*ALL_FEATURES, TARGET_COLUMN)]
    )

    nulls = sum(frame.null_count().row(0))
    if nulls:
        raise SchemaError(f"meta-dataset contains {nulls} null values")
    return frame


def aggregate_by_dataset(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse to one row per dataset: the dataset features plus the mean MCC.

    This is the target E1 is actually fitted against. Fitting the 476 raw rows with
    dataset features alone gives the same weights -- least squares on a predictor that
    is constant within a group lands on the group mean either way -- but working on the
    20 aggregated points makes that explicit and keeps the fold count honest.
    """
    return (
        frame.group_by(DATASET_COLUMN)
        .agg(
            [pl.col(column).first() for column in DATASET_FEATURES]
            + [
                pl.col(TARGET_COLUMN).mean().alias(TARGET_COLUMN),
                pl.len().alias("n_models"),
            ]
        )
        .sort(DATASET_COLUMN)
    )


def columns_as_arrays(frame: pl.DataFrame, features: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract the named columns as a mapping of float arrays, the form terms consume."""
    return {name: frame[name].to_numpy().astype(np.float64) for name in features}


def target(frame: pl.DataFrame) -> np.ndarray:
    """The MCC column as a float array."""
    return frame[TARGET_COLUMN].to_numpy().astype(np.float64)


def groups(frame: pl.DataFrame, column: str) -> np.ndarray:
    """The grouping labels used by the leave-one-out splitters."""
    return frame[column].to_numpy()
