"""Loading and shaping of the meta-dataset.

The meta-dataset has one row per (dataset, model) pair, or per run of one dataset when the study has
a single grouping factor. Two groups of columns matter and they behave differently, which is why they
are named separately: dataset features are constant across every row of a given dataset, so on their
own they can only ever predict a per-dataset constant; model features vary with the model (or the
configuration) and may also vary with the dataset. That split, the group columns, the target and the
range it can take are the ``corpus``, ``response`` and ``features`` sections of the study file
(`ml_meta_perf.config.StudyConfig.schema`), not constants here.

**Why a study may list more features than its equations use.** Designing a corpus needs
*identification*: the features must name every dataset and every learner, because two rows that share
a feature vector can never be told apart by an equation over those features. Fitting an equation needs
*compression*. A wide feature set is right at the first stage and a short equation at the second.

Study chapter: [7. Other targets][study-chapter] -- what a study is made of.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/07-generic-targets.md
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf.config import RowFilter, Schema, default_schema, load_config


def resolve_path(path: str | Path) -> Path:
    """Where a relative corpus path from the study file points: the checkout, else the working directory.

    The first covers an editable install and ``PYTHONPATH=src``. The second covers a regular
    install, whose module sits in `site-packages` with no repository around it -- which is why
    a path derived from this file alone failed there -- run from a clone, as CI does. An absolute
    path is returned as it is. Anywhere else, pass ``--data``. When no candidate exists the
    checkout path is returned, so the error `load` raises names where the file was expected.
    """
    given = Path(path)
    if given.is_absolute():
        return given
    checkout = Path(__file__).resolve().parents[2] / given
    working = Path.cwd() / given
    return next((candidate for candidate in (checkout, working) if candidate.is_file()), checkout)


class SchemaError(ValueError):
    """The CSV does not carry the columns the meta-model needs, or a value is out of domain for its transform."""


def load(path: str | Path | None = None, schema: Schema | None = None) -> pl.DataFrame:
    """Read the meta-dataset and check it has the expected shape.

    With no ``path`` the corpus is the one ``config/mcc.json`` names, and with no ``schema``
    the columns are the ones it lists: a bare ``load()`` is the study as configured. An explicit
    ``path`` is used as given, relative to the working directory.

    The schema selects the required columns and optional task type.
    Every feature is cast to Float64. The integer-valued columns (``nr_attr``,
    ``nr_class``, ...) are still continuous as far as the equations are concerned,
    and carrying two dtypes through the term library buys nothing.
    """
    if path is None or schema is None:
        study = load_config()
        schema = schema or study.schema()
        path = path if path is not None else resolve_path(study.corpus.path)
    resolved = Path(path)
    if not resolved.is_file():
        raise SchemaError(f"meta-dataset not found: {resolved}")

    frame = pl.read_csv(resolved)
    expected = (*schema.group_columns, *schema.features, schema.target_column)
    missing = [column for column in expected if column not in frame.columns]
    for requirement in schema.required:
        if requirement.column not in frame.columns:
            hint = f"; {requirement.hint}" if requirement.hint else ""
            raise SchemaError(f"stale meta-dataset {resolved}: no {requirement.column} column{hint}")
    if schema.row_filter is not None:
        frame = _filtered(frame, schema.row_filter)
    present = [column for column in schema.single_valued if column in frame.columns]
    if present:
        fitted = frame
        if schema.drop_missing_response and schema.target_column in frame.columns:
            fitted = frame.filter(pl.col(schema.target_column).is_not_null())
        values = sorted(str(row) for row in fitted.select(present).unique().iter_rows())
        if len(values) > 1:
            raise SchemaError(
                f"meta-dataset {resolved} mixes values of {', '.join(present)}: {', '.join(values)}; "
                "fit runs that share one value at a time"
            )
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    frame = frame.select(expected).with_columns(
        [pl.col(column).cast(pl.Float64) for column in (*schema.features, schema.target_column)]
    )

    checked = frame.drop(schema.target_column) if schema.drop_missing_response else frame
    nulls = sum(checked.null_count().row(0))
    if nulls:
        empty_response = not schema.drop_missing_response and frame[schema.target_column].null_count() > 0
        hint = "; set corpus.drop_missing_response to drop the rows with an empty response" if empty_response else ""
        raise SchemaError(f"meta-dataset contains {nulls} null values{hint}")
    if schema.drop_missing_response:
        frame = _without_empty_response(frame, schema, resolved)
    return frame


def _filtered(frame: pl.DataFrame, row_filter: RowFilter) -> pl.DataFrame:
    """The rows where the filter's column equals its value, or a SchemaError saying why that is impossible."""
    if row_filter.column not in frame.columns:
        raise SchemaError(f"missing columns: {[row_filter.column]}")
    column = pl.col(row_filter.column)
    if isinstance(row_filter.equals, bool):
        kept = frame.filter(column.cast(pl.Boolean) == row_filter.equals)
    elif isinstance(row_filter.equals, str):
        kept = frame.filter(column.cast(pl.String) == row_filter.equals)
    else:
        kept = frame.filter(column.cast(pl.Float64) == float(row_filter.equals))
    if kept.is_empty():
        name = row_filter.label or f"{row_filter.column} == {row_filter.equals}"
        raise SchemaError(f"meta-dataset has no {name} rows")
    return kept


def _without_empty_response(frame: pl.DataFrame, schema: Schema, source: Path) -> pl.DataFrame:
    """Drop the rows with an empty response and say, on stderr, how many and where they were."""
    kept = frame.drop_nulls(schema.target_column)
    dropped = frame.height - kept.height
    if dropped:
        empty = frame.filter(pl.col(schema.target_column).is_null())
        breakdown = "; ".join(
            f"by {column}: "
            + ", ".join(f"{name} {count}" for name, count in empty.group_by(column).len().sort(column).iter_rows())
            for column in schema.group_columns
        )
        message = f"dropped {dropped} of {frame.height} rows with an empty {schema.target_column} from {source}"
        print(f"{message} ({breakdown})", file=sys.stderr)
        if kept.is_empty():
            raise SchemaError(f"meta-dataset has no rows with a {schema.target_column} value")
    return kept


def columns_as_arrays(frame: pl.DataFrame, features: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract the named columns as a mapping of float arrays, the form terms consume."""
    return {name: frame[name].to_numpy().astype(np.float64) for name in features}


def target(frame: pl.DataFrame, schema: Schema | None = None) -> np.ndarray:
    """The schema's target column as a float array, on the scale the equation is fitted on."""
    schema = schema or default_schema()
    values = frame[schema.target_column].to_numpy().astype(np.float64)
    return to_fit_scale(values, schema)


def to_fit_scale(values: np.ndarray, schema: Schema) -> np.ndarray:
    """``values`` on the fitted scale: unchanged, ``log(1 + y)`` for ``log1p`` or ``log(y)`` for ``log``."""
    if schema.transform == "none":
        return values
    if not np.all(np.isfinite(values)):
        raise SchemaError(
            f"{schema.target_column} has non-finite values; a {schema.transform} scale needs finite costs"
        )
    if schema.transform == "log1p":
        if np.any(values < 0.0):
            raise SchemaError(f"{schema.target_column} has negative values; log1p needs non-negative costs")
        return np.log1p(values)
    if np.any(values <= 0.0):
        raise SchemaError(f"{schema.target_column} has zero or negative values; log needs strictly positive costs")
    return np.log(values)


def from_fit_scale(values: np.ndarray, schema: Schema) -> np.ndarray:
    """The inverse of `to_fit_scale`: predictions on the fitted scale back on the target's own scale.

    ``exp`` of a mean on the log scale is the geometric mean, which sits below the arithmetic mean
    of a skewed cost (retransformation bias, chapter 7 section 7.5); no smearing correction is applied.
    """
    if schema.transform == "log1p":
        return np.expm1(values)
    if schema.transform == "log":
        return np.exp(values)
    return values


def groups(frame: pl.DataFrame, column: str) -> np.ndarray:
    """The grouping labels used by the leave-one-out splitters."""
    return frame[column].to_numpy()


def drop_constant_features(frame: pl.DataFrame, schema: Schema) -> Schema:
    """The schema without the features that take a single value in this frame.

    A constant feature cannot explain anything, and a term dividing it by a varying one
    survives the library's constant-term screen as a relabelled copy of that other feature.
    """
    constant = {name for name in schema.features if frame[name].n_unique() <= 1}
    return dataclasses.replace(
        schema,
        dataset_features=tuple(name for name in schema.dataset_features if name not in constant),
        model_features=tuple(name for name in schema.model_features if name not in constant),
    )
