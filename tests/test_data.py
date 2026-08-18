"""Loading, schema validation and the aggregation E1 is fitted on."""

import tempfile
import unittest
from pathlib import Path

import polars as pl

from ml_meta_perf.data import (
    ALL_FEATURES,
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_CAPABILITY,
    MODEL_COLUMN,
    MODEL_FAMILY,
    MODEL_FEATURES,
    TARGET_COLUMN,
    SchemaError,
    aggregate_by_dataset,
    columns_as_arrays,
    groups,
    load,
    target,
)


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

    def test_capability_column_matches_its_documented_provenance(self) -> None:
        # The CSV carries `Model Capability` like any other feature, but unlike the others
        # it was assigned here rather than measured upstream. This is what keeps the column
        # and the taxonomy that explains it from drifting apart.
        frame = load()
        for model, capability in zip(
            frame[MODEL_COLUMN].to_list(), frame["Model Capability"].to_list(), strict=True
        ):
            self.assertEqual(capability, float(MODEL_CAPABILITY[MODEL_FAMILY[model]]))

    def test_every_family_sits_somewhere_on_the_capability_ladder(self) -> None:
        self.assertEqual(set(MODEL_FAMILY.values()), set(MODEL_CAPABILITY))
        self.assertEqual(
            sorted(MODEL_CAPABILITY.values()), list(range(1, len(MODEL_CAPABILITY) + 1))
        )

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
                pl.when(pl.col(MODEL_COLUMN) == "m1")
                .then(None)
                .otherwise(pl.col("gravity"))
                .alias("gravity")
            )
            frame.write_csv(path)
            with self.assertRaises(SchemaError) as caught:
                load(path)
            self.assertIn("null", str(caught.exception))


class TestAggregate(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = synthetic()
        self.aggregated = aggregate_by_dataset(self.frame)

    def test_one_row_per_dataset(self) -> None:
        self.assertEqual(self.aggregated.height, 2)
        self.assertEqual(sorted(self.aggregated[DATASET_COLUMN].to_list()), ["alpha", "beta"])

    def test_target_is_the_group_mean(self) -> None:
        expected = self.frame.group_by(DATASET_COLUMN).agg(pl.col(TARGET_COLUMN).mean()).sort(DATASET_COLUMN)
        self.assertEqual(
            self.aggregated.sort(DATASET_COLUMN)[TARGET_COLUMN].to_list(),
            expected[TARGET_COLUMN].to_list(),
        )

    def test_dataset_features_survive_intact(self) -> None:
        # They are constant within a dataset, so taking the first is lossless.
        row = self.aggregated.filter(pl.col(DATASET_COLUMN) == "alpha")
        self.assertEqual(row["class_ent"].item(), 1.0)

    def test_counts_the_models(self) -> None:
        self.assertEqual(set(self.aggregated["n_models"].to_list()), {3})

    def test_real_dataset_aggregates_to_twenty(self) -> None:
        self.assertEqual(aggregate_by_dataset(load()).height, 20)


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
