"""Loading, schema validation and the aggregation E1 is fitted on."""

import dataclasses
import json
import os
import tempfile
import unittest
from math import nextafter
from pathlib import Path
from unittest import mock

import polars as pl

from ml_meta_perf.config import DEFAULT_CONFIG_PATH, Schema, StudyConfig, load_config
from ml_meta_perf.data import (
    SchemaError,
    aggregate_by_dataset,
    columns_as_arrays,
    corpus_summary,
    feature_glossary,
    groups,
    load,
    missing_cells,
    model_family,
    model_ordinals,
    neural_families,
    resolve_path,
    target,
    target_summary,
    tree_families,
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

    def test_capability_column_matches_its_documented_provenance(self) -> None:
        # The CSV carries `Model Capability` like any other feature, but unlike the others
        # it was assigned here rather than measured upstream. This is what keeps the column
        # and the taxonomy that explains it from drifting apart.
        frame = load()
        for model, capability in zip(frame[MODEL_COLUMN].to_list(), frame["Model Capability"].to_list(), strict=True):
            self.assertEqual(capability, float(model_ordinals()["Model Capability"][model_family()[model]]))

    def test_every_family_sits_somewhere_on_the_capability_ladder(self) -> None:
        capability = model_ordinals()["Model Capability"]
        self.assertEqual(set(model_family().values()), set(capability))
        self.assertEqual(sorted(capability.values()), list(range(1, len(capability) + 1)))

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
    """The corpus, its columns and its path come from ``config/study.json``, not from the code."""

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


class TestSideFiles(unittest.TestCase):
    """The taxonomy, the ordinals and the glossary come from the files ``side_files`` names."""

    def study_with(self, **paths: str | None) -> StudyConfig:
        study = load_config()
        return dataclasses.replace(study, side_files=dataclasses.replace(study.side_files, **paths))

    def test_the_taxonomy_covers_every_model_of_the_corpus(self) -> None:
        families = model_family()
        self.assertEqual(set(families), set(load()[MODEL_COLUMN].unique().to_list()))
        self.assertTrue(set(tree_families()) <= set(families.values()))
        self.assertTrue(set(neural_families()) <= set(families.values()))
        self.assertTrue(set(tree_families()).isdisjoint(neural_families()))

    def test_the_glossary_explains_every_feature(self) -> None:
        self.assertEqual(set(feature_glossary()), set(ALL_FEATURES))

    def test_the_ordinals_cover_every_model_descriptor_column_and_every_model(self) -> None:
        ordinals = model_ordinals()
        self.assertEqual(
            set(ordinals),
            {
                "Model Capability",
                "Solution Stochasticity",
                "Loss Margin Behaviour",
                "Input Distribution Modelling",
                "Fitting Regime",
            },
        )
        models = set(model_family())
        for column, table in ordinals.items():
            keys = set(table)
            self.assertTrue(keys == models or keys == set(model_family().values()), column)

    def test_a_study_without_a_taxonomy_says_so(self) -> None:
        with mock.patch("ml_meta_perf.data.load_config", return_value=self.study_with(taxonomy=None)):
            for accessor in (model_family, tree_families, neural_families):
                with self.assertRaisesRegex(SchemaError, "no taxonomy file"):
                    accessor()

    def test_a_study_without_ordinals_or_glossary_says_so(self) -> None:
        with mock.patch("ml_meta_perf.data.load_config", return_value=self.study_with(ordinals=None, glossary=None)):
            with self.assertRaisesRegex(SchemaError, "no ordinals file"):
                model_ordinals()
            with self.assertRaisesRegex(SchemaError, "no glossary file"):
                feature_glossary()

    def test_a_missing_file_names_itself(self) -> None:
        with (
            mock.patch(
                "ml_meta_perf.data.load_config", return_value=self.study_with(glossary="/nonexistent/glossary.json")
            ),
            self.assertRaisesRegex(SchemaError, "glossary.json"),
        ):
            feature_glossary()

    def test_a_malformed_file_is_reported_by_part(self) -> None:
        cases = {
            "taxonomy": ({"family": ["not", "a", "mapping"], "groups": {}}, model_family, "taxonomy.family"),
            "groups": ({"family": {}, "groups": {"tree": "x"}}, tree_families, "taxonomy.groups.tree"),
            "glossary": ({"nr_attr": 3}, feature_glossary, "glossary"),
        }
        for name, (payload, accessor, message) in cases.items():
            kind = "glossary" if name == "glossary" else "taxonomy"
            with tempfile.TemporaryDirectory() as directory, self.subTest(name):
                path = Path(directory) / f"{kind}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                patched = mock.patch("ml_meta_perf.data.load_config", return_value=self.study_with(**{kind: str(path)}))
                with patched, self.assertRaisesRegex(SchemaError, message):
                    accessor()

    def test_malformed_ordinals_are_reported(self) -> None:
        for payload in ({"columns": []}, {"columns": {"Fitting Regime": {"values": {"DT": "x"}}}}):
            with tempfile.TemporaryDirectory() as directory, self.subTest(payload=payload):
                path = Path(directory) / "ordinals.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                patched = mock.patch("ml_meta_perf.data.load_config", return_value=self.study_with(ordinals=str(path)))
                with patched, self.assertRaisesRegex(SchemaError, "malformed"):
                    model_ordinals()


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


class TestCorpusSummary(unittest.TestCase):
    """The corpus description is generated, so these pin what it must keep saying.

    Chapter 1 used to state its counts in prose, and they drifted. The point of moving them
    into `corpus_summary` is lost if the function itself is unpinned.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()
        table = corpus_summary(cls.frame)
        cls.summary = dict(zip(table["quantity"], table["count"], strict=True))

    def test_counts_the_rows_groups_and_features(self) -> None:
        self.assertEqual(self.summary["rows"], self.frame.height)
        self.assertEqual(self.summary["datasets"], 20)
        self.assertEqual(self.summary["models"], 25)
        self.assertEqual(self.summary["dataset features"], len(DATASET_FEATURES))
        self.assertEqual(self.summary["model features"], len(MODEL_FEATURES))

    def test_absent_cells_are_the_difference_from_a_full_grid(self) -> None:
        self.assertEqual(self.summary["cells absent from the dataset-by-model grid"], 20 * 25 - self.frame.height)

    def test_counts_are_integers_not_floats(self) -> None:
        """A count rendered as `484.0000` in a chapter table is a formatting bug with a
        cause: putting counts and a mean in one float column."""
        self.assertEqual(corpus_summary(self.frame).schema["count"], pl.Int64)


class TestTargetSummary(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = load()
        self.table = target_summary(self.frame)
        self.rows = dict(zip(self.table["quantity"], self.table["rows"], strict=True))

    def test_reports_the_pinned_rows(self) -> None:
        values = target(self.frame)
        self.assertEqual(self.rows["at exactly 1"], int((values == 1.0).sum()))
        self.assertEqual(self.rows["at exactly 0"], int((values == 0.0).sum()))
        self.assertEqual(self.rows["below 0"], int((values < 0.0).sum()))

    def test_boundary_counts_distinguish_exact_from_nearby_values(self) -> None:
        """Near-boundary targets must not be rounded into zero or perfect scores."""
        frame = pl.DataFrame({TARGET_COLUMN: [-0.1, 0.0, nextafter(0.0, 1.0), nextafter(1.0, 0.0), 1.0, 1.0]})
        table = target_summary(frame)
        rows = dict(zip(table["quantity"], table["rows"], strict=True))
        self.assertEqual(rows["below 0"], 1)
        self.assertEqual(rows["at exactly 0"], 1)
        self.assertEqual(rows["at exactly 1"], 2)

    def test_extremes_match_the_column(self) -> None:
        values = target(self.frame)
        stats = dict(zip(self.table["quantity"], self.table["MCC"], strict=True))
        self.assertAlmostEqual(stats["minimum"], float(values.min()))
        self.assertAlmostEqual(stats["maximum"], float(values.max()))
        self.assertAlmostEqual(stats["mean"], float(values.mean()))

    def test_extreme_counts_include_ties(self) -> None:
        values = target(self.frame)
        self.assertEqual(self.rows["minimum"], int((values == values.min()).sum()))
        self.assertEqual(self.rows["maximum"], int((values == values.max()).sum()))


class TestMissingCells(unittest.TestCase):
    def test_names_only_datasets_short_of_models(self) -> None:
        frame = load()
        table = missing_cells(frame)
        self.assertGreater(table.height, 0)
        for row in table.iter_rows(named=True):
            present = frame.filter(pl.col(DATASET_COLUMN) == row["dataset"]).height
            self.assertEqual(row["models_absent"], 25 - present)

    def test_the_absences_are_in_the_smallest_datasets(self) -> None:
        """The chapter's claim that the missingness is not at random, as a test."""
        frame = load()
        short = set(missing_cells(frame)["dataset"].to_list())
        sizes = frame.group_by(DATASET_COLUMN).agg(pl.col("nr_inst").first()).sort("nr_inst")
        smallest = set(sizes[DATASET_COLUMN].to_list()[: len(short)])
        self.assertEqual(short, smallest)


if __name__ == "__main__":
    unittest.main()
