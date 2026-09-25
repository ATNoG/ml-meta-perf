"""FlexFL schema, fitting and CLI contracts."""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf import cli, configuration_search, experiment
from ml_meta_perf.configuration_search import SearchSettings, _settings_payload, feature_subsets
from ml_meta_perf.data import (
    ALL_FEATURES,
    FLEXFL_MODEL_FEATURES,
    MCC_SCHEMA,
    aggregate_by_dataset,
    columns_as_arrays,
    drop_constant_features,
    flexfl_schema,
    groups,
    load,
    target,
)
from ml_meta_perf.experiment import DEFAULT
from ml_meta_perf.model import Equation
from ml_meta_perf.search import search
from ml_meta_perf.terms import build_library
from tests.corpus import sample_path

FIXTURE = Path(__file__).parent / "fixtures" / "flexfl_meta_dataset_sample.csv"
TINY = dataclasses.replace(DEFAULT, max_terms=2, pool_size=20, beam_width=1, max_arity=1)
ALGORITHMS = {"CentralizedSync", "CentralizedAsync", "DecentralizedSync", "DecentralizedAsync"}
DROPPED = {
    "strategy_dirichlet", "alpha", "distribution_percentage", "worker_rate_min", "is_classification", "n_classes"
}


def search_argv(output: Path, target: str = "comm_bytes_total") -> list[str]:
    return [
        "all", "--target", target, "--data", str(FIXTURE), "--output", str(output), "--jobs", "1",
        "--penalty", "20", "--zscore", "3", "--arity", "1", "--min-terms", "1", "--max-terms", "2",
        "--pool", "20", "--beam", "1", "--shortlist-top", "1",
    ]


class FlexFLSchemaTests(unittest.TestCase):
    def test_load_accepts_each_target(self) -> None:
        for target_name, task_type, rows in (
            ("total_time_s", None, 24), ("comm_bytes_total", None, 24),
            ("performance", "classification", 12), ("performance", "regression", 12),
        ):
            with self.subTest(target=target_name, task_type=task_type):
                schema = flexfl_schema(target_name, task_type)
                frame = load(FIXTURE, schema)
                self.assertEqual(frame.height, rows)
                for name in (*schema.features, schema.target_column):
                    self.assertEqual(frame[name].dtype, pl.Float64)
                self.assertEqual(frame["dataset"].dtype, pl.String)
                self.assertEqual(frame["fl_algo"].dtype, pl.String)

    def test_performance_filter_keeps_one_task_type(self) -> None:
        classification = load(FIXTURE, flexfl_schema("performance", "classification"))
        regression = load(FIXTURE, flexfl_schema("performance", "regression"))
        self.assertEqual(set(classification["dataset"]), {"clf_a", "clf_b", "clf_c"})
        self.assertGreaterEqual(float(np.min(regression["performance"].to_numpy())), 0.0)

    def test_groups_by_fl_algo(self) -> None:
        frame = load(FIXTURE, flexfl_schema("comm_bytes_total"))
        self.assertEqual(set(groups(frame, "fl_algo")), ALGORITHMS)

    def test_aggregate_by_dataset_uses_the_schema(self) -> None:
        schema = flexfl_schema("comm_bytes_total")
        frame = load(FIXTURE, schema)
        aggregate = aggregate_by_dataset(frame, schema)
        self.assertEqual(aggregate.height, 6)
        self.assertIn("dataset", aggregate.columns)
        actual = aggregate.filter(pl.col("dataset") == "clf_a")["comm_bytes_total"][0]
        expected = frame.filter(pl.col("dataset") == "clf_a")["comm_bytes_total"].mean()
        self.assertAlmostEqual(actual, expected)

    def test_schema_factory(self) -> None:
        with self.assertRaises(ValueError):
            flexfl_schema("performance")
        with self.assertRaises(ValueError):
            flexfl_schema("bogus")
        for name, task_type, expected in (
            ("total_time_s", None, None), ("comm_bytes_total", None, None),
            ("performance", "classification", (-1.0, 1.0)), ("performance", "regression", (0.0, 2.0)),
        ):
            self.assertEqual(flexfl_schema(name, task_type).bounds, expected)
        self.assertEqual(MCC_SCHEMA.features, ALL_FEATURES)
        self.assertTrue(load().equals(load(None, MCC_SCHEMA)))

    def test_drop_constant_features(self) -> None:
        schema = flexfl_schema("performance", "classification")
        frame = load(FIXTURE, schema)
        reduced = drop_constant_features(frame, schema)
        self.assertEqual(set(schema.features) - set(reduced.features), DROPPED)
        self.assertEqual(reduced.features, tuple(name for name in schema.features if name not in DROPPED))
        self.assertEqual(len(reduced.features), 31)

    def test_unbounded_fit_leaves_the_mcc_range(self) -> None:
        schema = flexfl_schema("comm_bytes_total")
        frame = load(FIXTURE, schema)
        schema = drop_constant_features(frame, schema)
        columns = columns_as_arrays(frame, schema.features)
        truth = target(frame, schema)
        library = build_library(schema.dataset_features, schema.model_features, columns, max_arity=1)
        unbounded = search(library, truth, max_terms=2, pool_size=20, beam_width=1, bounds=None).best()
        self.assertIsNone(unbounded.bounds)
        self.assertGreater(np.abs(unbounded.predict(columns)).max(), 1.0)
        bounded = search(library, truth, max_terms=2, pool_size=20, beam_width=1).best().predict(columns)
        self.assertTrue(np.all((bounded >= -1.0) & (bounded <= 1.0)))

    def test_run_flexfl_on_a_cost_target(self) -> None:
        result = experiment.run_flexfl(FIXTURE, flexfl_schema("comm_bytes_total"), TINY)
        self.assertIsNone(result.equation.equation.bounds)
        self.assertNotIn("alpha", result.schema.features)
        for term in result.equation.equation.terms:
            self.assertTrue(set(term.features).isdisjoint(DROPPED))
        self.assertGreater(result.effects.height, 0)
        for row in result.effects.iter_rows(named=True):
            expected = "raises" if row["beta"] > 0 else "lowers"
            self.assertEqual(row["direction"], f"{expected} comm_bytes_total")
        shares = result.shares["share"]
        self.assertTrue(abs(shares.sum() - 1) < 1e-9 or all(value == 0 for value in shares))

    def test_study_cli_argument_errors(self) -> None:
        for argv in (
            ["--target", "performance", "--data", str(FIXTURE)],
            ["--target", "comm_bytes_total"],
            ["--task-type", "regression"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                cli.main(argv)
            self.assertEqual(error.exception.code, 2)

    def test_study_cli_writes_flexfl_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            code = cli.main([
                "--target", "comm_bytes_total", "--data", str(FIXTURE), "--output", str(out), "--quiet",
                "--max-terms", "2", "--pool", "20", "--beam", "1", "--arity", "1",
            ])
            self.assertEqual(code, 0)
            folder = out / "flexfl" / "comm_bytes_total"
            for name in ("equation.json", "equation.txt", "curve.csv", "term_effects.csv", "group_shares.csv"):
                self.assertTrue((folder / name).is_file(), name)
            self.assertIsNone(Equation.load(folder / "equation.json").bounds)
            self.assertTrue((folder / "equation.txt").read_text().startswith("comm_bytes_total = "))
            self.assertFalse((out / "e1.json").exists())
            self.assertFalse((out / "e3.json").exists())

    def test_search_cli_argument_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for argv in (
                ["all", "--target", "performance", "--data", str(FIXTURE), "--output", str(out)],
                ["all", "--target", "comm_bytes_total", "--output", str(out)],
                ["all", "--task-type", "regression", "--output", str(out)],
                [
                    "all", "--target", "comm_bytes_total", "--data", str(FIXTURE),
                    "--max-features", "3", "--output", str(out),
                ],
            ):
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                    configuration_search.main(argv)
                self.assertEqual(error.exception.code, 2)

    def test_search_cli_runs_a_cost_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertEqual(configuration_search.main(search_argv(out)), 0)
            self.assertIsNone(Equation.load(out / "e3_valid.json").bounds)
            self.assertTrue((out / "e3_valid.txt").read_text().startswith("comm_bytes_total = "))
            table = pl.read_csv(out / "equation_search.csv")
            for column in (
                "binary", "ranking", "binary_accuracy", "binary_f1", "binary_map",
                "ranking_map", "ranking_mrr", "ranking_hit1", "ranking_regret1",
            ):
                self.assertTrue(table[column].is_nan().all(), column)
            self.assertTrue(np.isfinite(table["objective"].to_numpy()).all())
            self.assertEqual(json.loads((out / "manifest.json").read_text())["settings"]["target"], "comm_bytes_total")

    def test_search_records_the_fitted_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertEqual(configuration_search.main(search_argv(out)), 0)
            for name in ("equation_search.csv", "finalists.csv", "finalist_fold_errors.csv"):
                table = pl.read_csv(out / name)
                with self.subTest(name=name):
                    for requested, fitted in zip(table["features"], table["fitted_features"], strict=True):
                        requested_set, fitted_set = set(json.loads(requested)), set(json.loads(fitted))
                        self.assertEqual(fitted_set, requested_set - DROPPED)
                        self.assertLess(len(fitted_set), len(requested_set))
            finalists = pl.read_csv(out / "finalists.csv")
            self.assertFalse([column for column in finalists.columns if column.endswith("_right")])
            selected = json.loads((out / "selected_configurations.json").read_text())
            for key in ("e3_valid", "e3_max"):
                with self.subTest(candidate=key):
                    fitted = json.loads(selected[key]["fitted_features"])
                    self.assertEqual(selected[key]["n_fitted_features"], len(fitted))
                    self.assertTrue(set(fitted).isdisjoint(DROPPED))
                    self.assertEqual(selected[key]["n_features"], len(FLEXFL_MODEL_FEATURES))

    def test_mcc_rows_carry_no_fitted_columns(self) -> None:
        self.assertEqual(configuration_search._fitted_columns(SearchSettings(), ("Model Capability",)), {})

    def test_mcc_search_outputs_carry_no_fitted_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            argv = [
                "all", "--data", str(sample_path()), "--output", str(out), "--jobs", "1", "--penalty", "20",
                "--zscore", "3", "--arity", "1", "--feature-set", "Model Capability,Processing Units Number",
                "--min-terms", "1", "--max-terms", "2", "--pool", "20", "--beam", "1", "--shortlist-top", "1",
            ]
            self.assertEqual(configuration_search.main(argv), 0)
            fitted = {"fitted_features", "n_fitted_features"}
            for name in ("equation_search.csv", "finalists.csv", "finalist_fold_errors.csv"):
                self.assertTrue(fitted.isdisjoint(pl.read_csv(out / name).columns), name)
            selected = json.loads((out / "selected_configurations.json").read_text())
            for key in ("e3_valid", "e3_max"):
                self.assertTrue(fitted.isdisjoint(selected[key]), key)

    def test_search_cli_runs_performance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            argv = [*search_argv(out, "performance"), "--task-type", "classification"]
            self.assertEqual(configuration_search.main(argv), 0)
            self.assertEqual(Equation.load(out / "e3_valid.json").bounds, (-1.0, 1.0))
            self.assertTrue((out / "e3_valid.txt").read_text().startswith("performance = "))

    def test_search_settings_for_flexfl(self) -> None:
        settings = SearchSettings(target="comm_bytes_total", minimum_features=27, maximum_features=27)
        self.assertEqual(feature_subsets(settings), [tuple(sorted(FLEXFL_MODEL_FEATURES))])
        SearchSettings(target="comm_bytes_total", minimum_features=1, maximum_features=27,
                       explicit_feature_sets=(("learning_rate", "num_workers"),)).validate()
        with self.assertRaisesRegex(ValueError, "Model Capability"):
            SearchSettings(target="comm_bytes_total", minimum_features=1, maximum_features=27,
                           explicit_feature_sets=(("Model Capability",),)).validate()
        with self.assertRaises(ValueError):
            SearchSettings(task_type="regression").validate()
        self.assertNotIn("target", _settings_payload(SearchSettings()))
        self.assertNotIn("task_type", _settings_payload(SearchSettings()))

    def test_search_cli_feature_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            argv = [*search_argv(out), "--feature-set", "learning_rate,num_workers,local_epochs"]
            self.assertEqual(configuration_search.main(argv), 0)
            table = pl.read_csv(out / "equation_search.csv")
            self.assertEqual({tuple(json.loads(value)) for value in table["features"]}, {
                ("learning_rate", "local_epochs", "num_workers")
            })
        with tempfile.TemporaryDirectory() as directory:
            argv = [*search_argv(Path(directory)), "--feature-set", "Model Capability"]
            with self.assertRaisesRegex(ValueError, "Model Capability"):
                configuration_search.main(argv)

    def test_constant_features_never_reach_an_arity_2_library(self) -> None:
        schema = flexfl_schema("performance", "classification")
        frame = load(FIXTURE, schema)
        reduced = drop_constant_features(frame, schema)
        columns = columns_as_arrays(frame, reduced.features)
        library = build_library(reduced.dataset_features, reduced.model_features, columns, max_arity=2)
        self.assertTrue(all(set(term.features).isdisjoint(DROPPED) for term in library.terms))
        control = build_library(schema.dataset_features, schema.model_features,
                                columns_as_arrays(frame, schema.features), max_arity=2)
        self.assertTrue(any("alpha" in term.features or "n_classes" in term.features for term in control.terms))

    def test_scale_free_objective(self) -> None:
        perfect = {name: 1.0 for name in (
            "in_sample_r2", "loo_dataset_r2", "loo_model_r2", "stability", "brevity",
        )} | {"binary": 0.0, "ranking": 0.0}
        negative = perfect | {name: -0.5 for name in ("in_sample_r2", "loo_dataset_r2", "loo_model_r2")}
        negative |= {"stability": 0.0, "brevity": 0.0}
        self.assertAlmostEqual(configuration_search.scale_free_objective(perfect), 1.0)
        self.assertAlmostEqual(configuration_search.scale_free_objective(negative), 0.0)
