"""Loading and shaping of the meta-dataset.

The meta-dataset has one row per (dataset, model) pair. Two groups of columns matter
and they behave very differently, which is why they are named separately here:

* dataset features are constant across every row of a given dataset, so on their own
  they can only ever predict a per-dataset constant;
* model features vary with the model, and one of them -- ``Processing Units Number`` --
  also varies with the dataset, since it is a function of the dataset's shape. The
  meta-dataset pipeline is the source of this schema; this module loads the generated
  corpus and keeps the package-level column contract explicit.

That asymmetry is the whole point of the two-equation comparison, so the split is part of the
study description -- the ``features.dataset`` and ``features.model`` lists of ``config/study.json``
(`ml_meta_perf.config.default_schema`) -- and not something each caller re-derives.

**Why there are eighteen features when the equation uses fewer.**

The two numbers answer different questions, asked at different stages, and reading the
second as a criticism of the first is the most natural mistake to make about this study.

*Designing the corpus* comes first, before any equation exists and before anyone knows
which terms will be worth having. The requirement there is **identification**: the
features must name every dataset and every learner the corpus contains, because two rows
sharing a feature vector are two rows no equation over those features can ever tell
apart, and a difference between them is then unexplainable rather than merely
unexplained. The right move at that stage is a wide feature set chosen for coverage. This
corpus meets the requirement exactly -- the twelve dataset features give twenty distinct
vectors for twenty datasets, and the six model features separate all twenty-five learners
on every dataset, with zero ambiguous rows of 484. ``tests/test_model_features.py`` pins
both.

Two caveats belong with that claim rather than after it. The model side is **joint**, not
standalone: five of the six are constant per model and separate only 19 of the 25 on their
own -- ``FT-Transformer``/``TabNet``/``TabTransformer``, ``LightGBM_RF``/``XGBoost``,
``DNN``/``MLP``, ``TabICL``/``TabPFN`` and ``BernoulliNB``/``GaussianNB`` collide -- and it
is ``Processing Units Number``, which varies with the dataset, that breaks those ties. So
the six identify a learner *on a given dataset*. And identification was bought with
redundancy: descriptor columns can be strongly correlated, and
``log(inst_to_attr) + log(nr_attr)`` **is** ``log(nr_inst)`` to 2e-15, because
``inst_to_attr`` is defined as their ratio.

*Fitting the equation* comes second, and its criterion is not coverage but **compression**.
An equation is a statement about families of datasets and families of learners, not about
individuals, so it is expected to need fewer features as it gets better -- and the
redundancy above is part of what it is compressing away. E3 searches all eighteen and keeps
fewer (``tests/test_model_features.py`` pins that it does). That is the mechanism working,
not a shortfall in the corpus, and an equation that used all eighteen would be one that had
failed to generalise.

So a column may earn its place at either stage: a descriptor that adds little to a fit can
still be what keeps two learners distinguishable. **Do not read absence from the equation as
evidence against a feature**, and do not read presence as robust either -- which features an
equation keeps is a property of one configuration and one length. The search, not the
configuration, decides which features an equation uses: every equation is offered every
feature it is allowed.

**``nr_inst`` describes the source dataset, not the training set.** The pipeline tunes each
model on a stratified sample capped at 100,000 rows and then evaluates on the available source
data (one dataset is capped at 1,000,000), and ten of the twenty datasets exceed 100,000 rows --
up to seven million. Nothing in the CSV records the size a model was trained on, so ``nr_inst``
and ``inst_to_attr`` are properties of the corpus a dataset was drawn from, and no statement
about "more training data" can be tested against them.

Study chapter: [1. The dataset][study-chapter] -- the rationale, in
prose, with the figures.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/01-dataset.md
"""

from __future__ import annotations

import dataclasses
import json
import sys
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from ml_meta_perf.config import RequiredColumn, RowFilter, Schema, default_schema, load_config
from ml_meta_perf.model import MCC_LOWER, MCC_UPPER

if TYPE_CHECKING:
    import argparse

# The MCC study's columns and features are not constants here: they are the ``corpus`` and
# ``features`` sections of ``config/study.json`` (`ml_meta_perf.config.default_schema`).


# The learner taxonomy, the ordinals behind the model-descriptor columns and the feature glossary
# are not constants here either: they are the files named by ``side_files`` in ``config/study.json``,
# read by `model_family`, `model_ordinals`, `feature_glossary` and friends below.

FLEXFL_DATASET_COLUMN = "dataset"
FLEXFL_MODEL_COLUMN = "fl_algo"
FLEXFL_TARGETS: tuple[str, ...] = (
    "performance",
    "total_time_s",
    "comm_bytes_total",
    "n_epochs",
    "compute_time_total_s",
    "compute_time_max_s",
    "comm_time_total_s",
    "validation_time_s",
)
FLEXFL_COST_TARGETS: tuple[str, ...] = (
    "total_time_s",
    "comm_bytes_total",
    "compute_time_total_s",
    "compute_time_max_s",
    "comm_time_total_s",
    "validation_time_s",
)
FLEXFL_DECOMPOSITION_TARGETS: tuple[str, ...] = (
    "compute_time_total_s",
    "compute_time_max_s",
    "comm_time_total_s",
    "validation_time_s",
)
TASK_TYPES: tuple[str, ...] = ("classification", "regression")
FLEXFL_DATASET_FEATURES: tuple[str, ...] = (
    "n_samples",
    "n_features",
    "n_classes",
    "is_classification",
    "is_categorical",
    "total_parameters",
    "n_layers",
    "mean_layer_width",
    "max_layer_width",
    "weight_decay",
)
FLEXFL_EPOCH_CAP_COLUMN = "epoch_cap"
FLEXFL_EARLY_STOP_RULE_COLUMNS = ("early_stop_on", "min_epochs")
FLEXFL_MODEL_FEATURES: tuple[str, ...] = (
    "fl_algo_CentralizedSync",
    "fl_algo_CentralizedAsync",
    "fl_algo_DecentralizedSync",
    "fl_algo_DecentralizedAsync",
    "strategy_iid",
    "strategy_non_iid",
    "strategy_dirichlet",
    "learning_rate",
    "batch_size",
    "patience",
    "delta",
    "local_epochs",
    FLEXFL_EPOCH_CAP_COLUMN,
    "alpha",
    "distribution_percentage",
    "feat_entropy_mean",
    "feat_entropy_min",
    "feat_entropy_max",
    "feat_entropy_std",
    "num_workers",
    "n_atnog_test1",
    "n_hobbit",
    "n_samwise",
    "worker_rate_mean",
    "worker_rate_min",
    "worker_rate_max",
    "worker_rate_std",
    "worker_rate_cv",
)
SMAPE_LOWER = 0.0
SMAPE_UPPER = 2.0
FLEXFL_PERFORMANCE_BOUNDS: dict[str, tuple[float, float]] = {
    "classification": (MCC_LOWER, MCC_UPPER),
    "regression": (SMAPE_LOWER, SMAPE_UPPER),
}


def flexfl_schema(target: str, task_type: str | None = None, log_target: bool = False) -> Schema:
    """The FlexFL per-run schema for one target, optionally restricted to one task type and fitted on ``log1p``."""
    if target not in FLEXFL_TARGETS:
        raise ValueError(f"unknown FlexFL target {target!r}; expected one of {FLEXFL_TARGETS}")
    if task_type is not None and task_type not in TASK_TYPES:
        raise ValueError(f"unknown task type {task_type!r}; expected one of {TASK_TYPES}")
    if target == "performance" and task_type is None:
        raise ValueError("the performance target needs a task type: MCC and SMAPE do not share a scale")
    if log_target and target not in FLEXFL_COST_TARGETS:
        raise ValueError(f"log1p applies only to the cost targets {FLEXFL_COST_TARGETS}")
    bounds = FLEXFL_PERFORMANCE_BOUNDS[task_type] if target == "performance" and task_type is not None else None
    return Schema(
        dataset_column=FLEXFL_DATASET_COLUMN,
        model_column=FLEXFL_MODEL_COLUMN,
        dataset_features=FLEXFL_DATASET_FEATURES,
        model_features=FLEXFL_MODEL_FEATURES,
        target_column=target,
        bounds=bounds,
        transform="log1p" if log_target else "none",
        row_filter=None
        if task_type is None
        else RowFilter("is_classification", task_type == "classification", label=task_type),
        required=(
            RequiredColumn(FLEXFL_EPOCH_CAP_COLUMN, "re-assemble it with FlexFL's scripts/assemble_meta_dataset.py"),
        ),
        single_valued=FLEXFL_EARLY_STOP_RULE_COLUMNS,
        drop_missing_response=target in FLEXFL_DECOMPOSITION_TARGETS,
        drop_constant_features=True,
    )


def add_target_arguments(parser: argparse.ArgumentParser) -> None:
    """The ``--target``, ``--task-type`` and ``--log-target`` flags both commands share."""
    parser.add_argument(
        "--target",
        choices=("mcc", *FLEXFL_TARGETS),
        default="mcc",
        help="what the equation predicts: MCC, or one FlexFL per-run outcome (needs --data)",
    )
    parser.add_argument(
        "--task-type",
        choices=TASK_TYPES,
        default=None,
        help="restrict a FlexFL target to one task type; required with --target performance",
    )
    parser.add_argument(
        "--log-target",
        action="store_true",
        help="fit log1p of a cost target; metrics are then on the log scale",
    )


def check_target_arguments(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    """Reject incompatible target and task type arguments."""
    if arguments.target == "mcc" and arguments.task_type is not None:
        parser.error("--task-type applies only to a FlexFL --target")
    if arguments.target == "performance" and arguments.task_type is None:
        parser.error("--task-type is required with --target performance")
    if arguments.target != "mcc" and arguments.data is None:
        parser.error("--data is required with a FlexFL --target")
    if arguments.log_target and arguments.target not in FLEXFL_COST_TARGETS:
        parser.error(f"--log-target applies only to --target {' or '.join(FLEXFL_COST_TARGETS)}")


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


@cache
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _side_file(kind: str) -> Any:
    """The parsed side file ``config/study.json`` names under ``side_files.<kind>``."""
    path = getattr(load_config().side_files, kind)
    if path is None:
        raise SchemaError(f"this study has no {kind} file: side_files.{kind} is null in the study file")
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise SchemaError(f"{kind} file not found: {resolved}")
    return _read_json(resolved)


def _mapping(value: object, what: str, values: type) -> dict[str, Any]:
    """``value`` as a dict of strings to ``values``, or a SchemaError saying which file part is malformed."""
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, values) for key, item in value.items()
    ):
        raise SchemaError(f"malformed {what}: expected a mapping from names to {values.__name__}")
    return dict(value)


def model_family() -> dict[str, str]:
    """The learner family of every model, from the taxonomy file (``side_files.taxonomy``)."""
    return _mapping(_side_file("taxonomy").get("family"), "taxonomy.family", str)


def _family_group(name: str) -> tuple[str, ...]:
    groups_ = _side_file("taxonomy").get("groups")
    members = groups_.get(name) if isinstance(groups_, dict) else None
    if not isinstance(members, list) or not all(isinstance(item, str) for item in members):
        raise SchemaError(f"malformed taxonomy.groups.{name}: expected a list of family names")
    return tuple(members)


def tree_families() -> tuple[str, ...]:
    """The families the tabular-ML literature groups together as tree-based."""
    return _family_group("tree")


def neural_families() -> tuple[str, ...]:
    """The families that are neural networks, split by whether the architecture targets tabular data."""
    return _family_group("neural")


def model_ordinals() -> dict[str, dict[str, int]]:
    """The assigned ordinal behind each model-descriptor column, keyed by column then by family or model."""
    columns = _side_file("ordinals").get("columns")
    if not isinstance(columns, dict):
        raise SchemaError("malformed ordinals: expected a 'columns' mapping")
    return {
        column: _mapping(entry.get("values") if isinstance(entry, dict) else None, f"ordinals.{column}", int)
        for column, entry in columns.items()
    }


def feature_glossary() -> dict[str, str]:
    """A plain-language meaning for each feature column, from the glossary file."""
    return _mapping(_side_file("glossary"), "glossary", str)


def load(path: str | Path | None = None, schema: Schema | None = None) -> pl.DataFrame:
    """Read the meta-dataset and check it has the expected shape.

    With no ``path`` the corpus is the one ``config/study.json`` names, and with no ``schema``
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
    expected = (schema.dataset_column, schema.model_column, *schema.features, schema.target_column)
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
            for column in (schema.dataset_column, schema.model_column)
        )
        message = f"dropped {dropped} of {frame.height} rows with an empty {schema.target_column} from {source}"
        print(f"{message} ({breakdown})", file=sys.stderr)
        if kept.is_empty():
            raise SchemaError(f"meta-dataset has no rows with a {schema.target_column} value")
    return kept


def aggregate_by_dataset(frame: pl.DataFrame, schema: Schema | None = None) -> pl.DataFrame:
    """Collapse to one row per schema dataset: its features plus its mean target.

    **E1 is no longer fitted against this**, and the reason is worth recording. Least
    squares on a predictor that is constant within a group lands on that group's mean
    either way, so aggregating first looked free. It was not: it put E1's R2 on a
    20-point denominator, which cannot be compared with E3's 484-row one, and the 0.506
    it produced read as *better* transfer than E3's 0.466 when on the common scale it is
    0.217. All three equations are now fitted on all 484 rows.

    Kept because the aggregated view is still the right one for describing the corpus --
    20 datasets, one row each -- and because chapter 6 quotes the comparison.
    """
    schema = schema or default_schema()
    return (
        frame.group_by(schema.dataset_column)
        .agg(
            [pl.col(column).first() for column in schema.dataset_features]
            + [
                pl.col(schema.target_column).mean().alias(schema.target_column),
                pl.len().alias("n_models"),
            ]
        )
        .sort(schema.dataset_column)
    )


def columns_as_arrays(frame: pl.DataFrame, features: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract the named columns as a mapping of float arrays, the form terms consume."""
    return {name: frame[name].to_numpy().astype(np.float64) for name in features}


def target(frame: pl.DataFrame, schema: Schema | None = None) -> np.ndarray:
    """The schema's target column as a float array, ``log1p`` of it when the schema asks."""
    schema = schema or default_schema()
    values = frame[schema.target_column].to_numpy().astype(np.float64)
    if not schema.log_target:
        return values
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise SchemaError(f"{schema.target_column} has negative or non-finite values; log1p needs non-negative costs")
    return np.log1p(values)


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


def corpus_summary(frame: pl.DataFrame, schema: Schema | None = None) -> pl.DataFrame:
    """The shape of the corpus: how many rows, groups, features, and how many cells absent.

    Every fact a chapter would otherwise state about the size of the meta-dataset, computed
    from the file rather than transcribed from it. Counts only -- the distribution of the
    target is `target_summary`, kept separate so each table can print at its own precision
    and neither has to render a count as ``484.0000``.
    """
    schema = schema or default_schema()
    datasets = int(frame[schema.dataset_column].n_unique())
    models = int(frame[schema.model_column].n_unique())
    return pl.DataFrame(
        [
            {"quantity": "rows", "count": frame.height},
            {"quantity": "datasets", "count": datasets},
            {"quantity": "models", "count": models},
            {"quantity": "cells absent from the dataset-by-model grid", "count": datasets * models - frame.height},
            {"quantity": "dataset features", "count": len(schema.dataset_features)},
            {"quantity": "model features", "count": len(schema.model_features)},
        ],
        schema={"quantity": pl.String, "count": pl.Int64},
    )


def target_summary(frame: pl.DataFrame, schema: Schema | None = None) -> pl.DataFrame:
    """How MCC is distributed across the corpus, including how much of it is pinned.

    The counts at exactly 0 and exactly 1 are here because they are the reason MAE and not
    SMAPE is the reported error, and a reader checking that argument should be able to see
    the counts it rests on. Their frequencies depend on the corpus; the negative
    row is reported separately.

    Not included, because it cannot be: the variation across the five seeds from which each
    row's maximum was selected. That lives upstream, in the corpus builder, and chapter 1
    cites it as an external audit rather than pretending this file can recompute it.
    """
    schema = schema or default_schema()
    values = target(frame, schema)
    pinned = [("at exactly 1", values == 1.0), ("at exactly 0", values == 0.0), ("below 0", values < 0.0)]
    rows: list[dict[str, object]] = [
        {"quantity": "mean", "MCC": float(values.mean()), "rows": frame.height},
        {"quantity": "standard deviation", "MCC": float(values.std()), "rows": frame.height},
        {"quantity": "minimum", "MCC": float(values.min()), "rows": int((values == values.min()).sum())},
        {"quantity": "maximum", "MCC": float(values.max()), "rows": int((values == values.max()).sum())},
    ]
    rows += [{"quantity": label, "MCC": float("nan"), "rows": int(mask.sum())} for label, mask in pinned]
    return pl.DataFrame(rows, schema={"quantity": pl.String, "MCC": pl.Float64, "rows": pl.Int64})


def missing_cells(frame: pl.DataFrame, schema: Schema | None = None) -> pl.DataFrame:
    """Which datasets are short of models, and how large those datasets are.

    The absent cells are **not** missing at random, and the table says so on its face
    rather than in a sentence beside it: the datasets with models missing are the smallest
    ones in the corpus. Reported per dataset with its instance count, so a reader can see the
    pattern instead of being told about it.
    """
    schema = schema or default_schema()
    models = int(frame[schema.model_column].n_unique())
    counts = (
        frame.group_by(schema.dataset_column)
        .agg(pl.len().alias("models_present"), pl.col("nr_inst").first().alias("nr_inst"))
        .filter(pl.col("models_present") < models)
        .sort("nr_inst")
    )
    return counts.with_columns((models - pl.col("models_present")).alias("models_absent")).select(
        pl.col(schema.dataset_column).alias("dataset"), "nr_inst", "models_absent"
    )
