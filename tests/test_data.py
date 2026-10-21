"""Loading, schema validation and the aggregation E1 is fitted on."""

import contextlib
import dataclasses
import io
import os
import tempfile
import unittest
from pathlib import Path

import polars as pl

from ml_meta_perf.config import DEFAULT_CONFIG_PATH, RequiredColumn, RowFilter, Schema, load_config
from ml_meta_perf.data import (
    SchemaError,
    columns_as_arrays,
    groups,
    load,
    resolve_path,
    target,
)
from tests.study import ALL_FEATURES, DATASET_COLUMN, DATASET_FEATURES, MODEL_COLUMN, MODEL_FEATURES, TARGET_COLUMN


def synthetic() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset in ("alpha", "beta"):
        for index, model in enumerate(("m1", "m2", "m3")):
            row: dict[str, object] = {DATASET_COLUMN: dataset, MODEL_COLUMN: model}
            for offset, name in enumerate(DATASET_FEATURES):
                row[name] = float(offset + 1 + (0 if dataset == "alpha" else 10))
            for offset, name in enumerate(MODEL_FEATURES):
                row[name] = float(offset + 1 + index)
            row[TARGET_COLUMN] = 0.1 * index + (0.0 if dataset == "alpha" else 0.5)
            rows.append(row)
    return pl.DataFrame(rows)


class TestLoad(unittest.TestCase):
    def test_loads_the_shipped_dataset(self) -> None:
        frame = load()
        self.assertGreater(frame.height, 0)
        for column in (DATASET_COLUMN, MODEL_COLUMN, *ALL_FEATURES, TARGET_COLUMN):
            self.assertIn(column, frame.columns)

    def test_features_are_floats(self) -> None:
        frame = load()
        for column in (*ALL_FEATURES, TARGET_COLUMN):
            self.assertEqual(frame.schema[column], pl.Float64)

    def test_target_stays_inside_the_mcc_range(self) -> None:
        values = target(load())
        self.assertGreaterEqual(values.min(), -1.0)
        self.assertLessEqual(values.max(), 1.0)

    def test_missing_file_is_reported(self) -> None:
        with self.assertRaises(SchemaError):
            load("/nonexistent/meta_dataset.csv")

    def test_missing_column_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.csv"
            synthetic().drop("gravity").write_csv(path)
            with self.assertRaises(SchemaError) as caught:
                load(path)
            self.assertIn("gravity", str(caught.exception))

    def test_nulls_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "holed.csv"
            frame = synthetic().with_columns(
                pl.when(pl.col(MODEL_COLUMN) == "m1").then(None).otherwise(pl.col("gravity")).alias("gravity")
            )
            frame.write_csv(path)
            with self.assertRaises(SchemaError) as caught:
                load(path)
            self.assertIn("null", str(caught.exception))


class TestLoadFromTheStudyFile(unittest.TestCase):
    """The corpus, its columns and its path come from ``config/mcc.json``, not from the code."""

    def test_a_bare_load_equals_reading_the_csv_with_the_pinned_columns(self) -> None:
        """`load` is the CSV read directly, with the columns the study lists, as Float64."""
        study = load_config()
        columns = [DATASET_COLUMN, MODEL_COLUMN, *DATASET_FEATURES, *MODEL_FEATURES, TARGET_COLUMN]
        oracle = pl.read_csv(resolve_path(study.corpus.path)).select(columns)
        oracle = oracle.with_columns(
            [pl.col(c).cast(pl.Float64) for c in (*DATASET_FEATURES, *MODEL_FEATURES, TARGET_COLUMN)]
        )
        self.assertTrue(load().equals(oracle))

    def test_the_study_file_decides_the_columns(self) -> None:
        """A study naming fewer features loads fewer columns, from the same CSV."""
        schema = Schema(
            dataset_column=DATASET_COLUMN,
            model_column=MODEL_COLUMN,
            dataset_features=("nr_inst",),
            model_features=("Fitting Regime",),
            target_column=TARGET_COLUMN,
            bounds=(-1.0, 1.0),
        )
        frame = load(None, schema)
        self.assertEqual(frame.columns, [DATASET_COLUMN, MODEL_COLUMN, "nr_inst", "Fitting Regime", TARGET_COLUMN])
        self.assertEqual(frame.height, load().height)

    def test_an_explicit_path_is_used_as_given_with_the_study_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "copy.csv"
            synthetic().write_csv(path)
            self.assertEqual(load(path).height, synthetic().height)

    def test_a_column_the_study_names_but_the_csv_lacks_is_reported_by_name(self) -> None:
        schema = Schema(
            dataset_column=DATASET_COLUMN,
            model_column=MODEL_COLUMN,
            dataset_features=("no_such_feature",),
            model_features=("Fitting Regime",),
            target_column=TARGET_COLUMN,
            bounds=None,
        )
        with self.assertRaisesRegex(SchemaError, "no_such_feature"):
            load(None, schema)

    def test_the_default_study_is_the_shipped_file(self) -> None:
        self.assertTrue(DEFAULT_CONFIG_PATH.is_file())
        self.assertEqual(resolve_path(load_config().corpus.path).name, "meta_dataset.csv")


class TestLoaderOptions(unittest.TestCase):
    """What a study file can ask of the loader, on a corpus that is nothing like the MCC one."""

    def schema(self, **options: object) -> Schema:
        return dataclasses.replace(load_config().schema(), **options)  # pyright: ignore[reportArgumentType]

    def written(self, frame: pl.DataFrame) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "corpus.csv"
        frame.write_csv(path)
        return directory, path

    def load_with(self, frame: pl.DataFrame, **options: object) -> pl.DataFrame:
        directory, path = self.written(frame)
        with directory:
            return load(path, self.schema(**options))

    def test_a_boolean_filter_keeps_one_subset(self) -> None:
        frame = synthetic().with_columns((pl.col(MODEL_COLUMN) != "m3").alias("flag"))
        kept = self.load_with(frame, row_filter=RowFilter("flag", True, "flagged"))
        self.assertEqual(sorted(kept[MODEL_COLUMN].unique().to_list()), ["m1", "m2"])
        rest = self.load_with(frame, row_filter=RowFilter("flag", False))
        self.assertEqual(rest[MODEL_COLUMN].unique().to_list(), ["m3"])

    def test_a_text_and_a_numeric_filter(self) -> None:
        frame = synthetic().with_columns(pl.col(MODEL_COLUMN).alias("kind"), pl.lit(2).alias("level"))
        self.assertEqual(self.load_with(frame, row_filter=RowFilter("kind", "m2")).height, 2)
        self.assertEqual(self.load_with(frame, row_filter=RowFilter("level", 2)).height, frame.height)

    def test_a_filter_on_a_missing_column_is_reported(self) -> None:
        with self.assertRaisesRegex(SchemaError, "missing columns: .*nope"):
            self.load_with(synthetic(), row_filter=RowFilter("nope", True))

    def test_a_filter_that_keeps_nothing_names_the_subset(self) -> None:
        frame = synthetic().with_columns(pl.lit(True).alias("flag"))
        with self.assertRaisesRegex(SchemaError, "no regression rows"):
            self.load_with(frame, row_filter=RowFilter("flag", False, "regression"))
        with self.assertRaisesRegex(SchemaError, r"no flag == False rows"):
            self.load_with(frame, row_filter=RowFilter("flag", False))

    def test_a_required_column_reports_its_hint(self) -> None:
        with self.assertRaisesRegex(SchemaError, "stale meta-dataset .*: no epoch_cap column; rebuild it"):
            self.load_with(synthetic(), required=(RequiredColumn("epoch_cap", "rebuild it"),))
        with self.assertRaisesRegex(SchemaError, r"no epoch_cap column$"):
            self.load_with(synthetic(), required=(RequiredColumn("epoch_cap"),))

    def test_a_required_column_that_is_present_passes(self) -> None:
        frame = synthetic().with_columns(pl.lit(5).alias("epoch_cap"))
        self.assertEqual(self.load_with(frame, required=(RequiredColumn("epoch_cap"),)).height, frame.height)

    def test_a_single_valued_column_that_varies_is_rejected_and_one_that_does_not_passes(self) -> None:
        mixed = synthetic().with_columns(pl.col(MODEL_COLUMN).alias("rule"))
        with self.assertRaisesRegex(SchemaError, "mixes values of rule: .*; fit runs that share one value"):
            self.load_with(mixed, single_valued=("rule",))
        fixed = synthetic().with_columns(pl.lit("loss").alias("rule"))
        self.assertEqual(self.load_with(fixed, single_valued=("rule",)).height, fixed.height)

    def test_a_single_valued_column_the_corpus_lacks_is_skipped(self) -> None:
        self.assertEqual(self.load_with(synthetic(), single_valued=("absent",)).height, synthetic().height)

    def test_a_single_valued_column_is_checked_after_the_filter(self) -> None:
        frame = synthetic().with_columns(
            (pl.col(MODEL_COLUMN) != "m3").alias("flag"),
            pl.when(pl.col(MODEL_COLUMN) == "m3").then(1).otherwise(0).alias("rule"),
        )
        kept = self.load_with(frame, row_filter=RowFilter("flag", True), single_valued=("rule",))
        self.assertEqual(kept.height, 4)

    def test_empty_responses_are_rejected_unless_the_study_drops_them(self) -> None:
        frame = synthetic().with_columns(
            pl.when(pl.col(MODEL_COLUMN) == "m1").then(None).otherwise(pl.col(TARGET_COLUMN)).alias(TARGET_COLUMN)
        )
        with self.assertRaisesRegex(SchemaError, "null values; set corpus.drop_missing_response"):
            self.load_with(frame)

    def test_dropped_rows_are_counted_on_stderr_by_group(self) -> None:
        frame = synthetic().with_columns(
            pl.when(pl.col(MODEL_COLUMN) == "m1").then(None).otherwise(pl.col(TARGET_COLUMN)).alias(TARGET_COLUMN)
        )
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            kept = self.load_with(frame, drop_missing_response=True)
        self.assertEqual(kept.height, 4)
        self.assertRegex(
            errors.getvalue(), r"dropped 2 of 6 rows with an empty MCC .*by Dataset: alpha 1, beta 1; by Model: m1 2"
        )

    def test_an_empty_feature_is_never_dropped_silently(self) -> None:
        frame = synthetic().with_columns(
            pl.when(pl.col(MODEL_COLUMN) == "m1").then(None).otherwise(pl.col("gravity")).alias("gravity")
        )
        with self.assertRaisesRegex(SchemaError, r"null values$"):
            self.load_with(frame, drop_missing_response=True)

    def test_a_corpus_with_no_response_at_all_is_rejected(self) -> None:
        frame = synthetic().with_columns(pl.lit(None, dtype=pl.Float64).alias(TARGET_COLUMN))
        with self.assertRaisesRegex(SchemaError, "no rows with a MCC value"):
            self.load_with(frame, drop_missing_response=True)


class TestResolvePath(unittest.TestCase):
    """Where a relative corpus path from the study file points."""

    def test_an_absolute_path_is_returned_as_it_is(self) -> None:
        self.assertEqual(resolve_path("/nonexistent/x.csv"), Path("/nonexistent/x.csv"))

    def test_a_relative_path_is_found_in_the_checkout(self) -> None:
        found = resolve_path("dataset/meta_dataset.csv")
        self.assertTrue(found.is_file())
        self.assertTrue(found.is_absolute())

    def test_a_relative_path_is_found_under_the_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "mine.csv").write_text("a\n1\n")
            previous = Path.cwd()
            os.chdir(directory)
            try:
                self.assertEqual(resolve_path("mine.csv").resolve(), (Path(directory) / "mine.csv").resolve())
            finally:
                os.chdir(previous)

    def test_a_missing_file_names_where_it_was_expected(self) -> None:
        missing = resolve_path("dataset/does_not_exist.csv")
        self.assertFalse(missing.exists())
        self.assertEqual(missing.name, "does_not_exist.csv")
        self.assertTrue(missing.is_absolute())


class TestAccessors(unittest.TestCase):
    def test_columns_as_arrays_returns_every_request(self) -> None:
        arrays = columns_as_arrays(synthetic(), DATASET_FEATURES)
        self.assertEqual(set(arrays), set(DATASET_FEATURES))
        for values in arrays.values():
            self.assertEqual(values.shape, (6,))

    def test_groups_returns_labels(self) -> None:
        self.assertEqual(sorted(set(groups(synthetic(), MODEL_COLUMN).tolist())), ["m1", "m2", "m3"])


if __name__ == "__main__":
    unittest.main()
