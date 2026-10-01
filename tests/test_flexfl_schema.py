"""FlexFL schema, fitting and CLI contracts."""

from __future__ import annotations

import dataclasses
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

from ml_meta_perf import cli, configuration_search, experiment
from ml_meta_perf.configuration_search import SearchSettings, _settings_payload, feature_subsets, library_points
from ml_meta_perf.data import (
    ALL_FEATURES,
    FLEXFL_COST_TARGETS,
    FLEXFL_DECOMPOSITION_TARGETS,
    FLEXFL_EPOCH_CAP_COLUMN,
    FLEXFL_MODEL_FEATURES,
    MCC_SCHEMA,
    Schema,
    SchemaError,
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
from ml_meta_perf.search import MIN_CONTRIBUTION, pruning_threshold, search
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import score
from tests.corpus import sample_path

FIXTURE = Path(__file__).parent / "fixtures" / "flexfl_meta_dataset_sample.csv"
TINY = dataclasses.replace(DEFAULT, max_terms=2, pool_size=20, beam_width=1, max_arity=1)
ALGORITHMS = {"CentralizedSync", "CentralizedAsync", "DecentralizedSync", "DecentralizedAsync"}
DROPPED = {
    "strategy_dirichlet", "alpha", "distribution_percentage", "worker_rate_min", "is_classification", "n_classes",
    FLEXFL_EPOCH_CAP_COLUMN,
}
N_MODEL_FEATURES = len(FLEXFL_MODEL_FEATURES)
DECOMPOSITION_TARGETS = ("compute_time_total_s", "compute_time_max_s", "comm_time_total_s", "validation_time_s")


def search_argv(output: Path, target: str = "comm_bytes_total") -> list[str]:
    return [
        "all", "--target", target, "--data", str(FIXTURE), "--output", str(output), "--jobs", "1",
        "--penalty", "20", "--zscore", "3", "--arity", "1", "--min-terms", "1", "--max-terms", "2",
        "--pool", "20", "--beam", "1", "--shortlist-top", "1",
    ]


class FlexFLSchemaTests(unittest.TestCase):
    def test_epoch_cap_is_a_flexfl_model_feature(self) -> None:
        self.assertIn(FLEXFL_EPOCH_CAP_COLUMN, FLEXFL_MODEL_FEATURES)
        self.assertIn(FLEXFL_EPOCH_CAP_COLUMN, flexfl_schema("n_epochs").model_features)
        self.assertNotIn(FLEXFL_EPOCH_CAP_COLUMN, MCC_SCHEMA.features)

    def test_load_rejects_a_stale_flexfl_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / "stale.csv"
            pl.read_csv(FIXTURE).drop(FLEXFL_EPOCH_CAP_COLUMN).write_csv(stale)
            with self.assertRaisesRegex(
                SchemaError,
                f"stale FlexFL meta-dataset .*: no {FLEXFL_EPOCH_CAP_COLUMN} column; "
                r"re-assemble it with FlexFL's scripts/assemble_meta_dataset\.py",
            ):
                load(stale, flexfl_schema("comm_bytes_total"))
            schema = flexfl_schema("comm_bytes_total")
            without = dataclasses.replace(
                schema,
                model_features=tuple(name for name in schema.model_features if name != FLEXFL_EPOCH_CAP_COLUMN),
            )
            self.assertNotIn(FLEXFL_EPOCH_CAP_COLUMN, load(stale, without).columns)

    def test_load_rejects_a_stale_flexfl_csv_before_the_task_type_filter(self) -> None:
        frame = pl.read_csv(FIXTURE).drop(FLEXFL_EPOCH_CAP_COLUMN)
        corpora = {
            "no classification rows": frame.with_columns(pl.lit(False).alias("is_classification")),
            "no is_classification column": frame.drop("is_classification"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, corpus in corpora.items():
                with self.subTest(name):
                    stale = Path(directory) / "stale.csv"
                    corpus.write_csv(stale)
                    with self.assertRaisesRegex(SchemaError, f"no {FLEXFL_EPOCH_CAP_COLUMN} column; re-assemble"):
                        load(stale, flexfl_schema("comm_bytes_total", "classification"))

    def test_load_rejects_mixed_early_stop_rules(self) -> None:
        frame = pl.read_csv(FIXTURE)
        mixed = frame.with_columns(
            pl.Series("early_stop_on", ["metric" if index % 2 else "loss" for index in range(frame.height)]),
            pl.Series("min_epochs", [0 if index % 2 else 10 for index in range(frame.height)]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.csv"
            mixed.write_csv(path)
            with self.assertRaisesRegex(
                SchemaError,
                r"mixes early-stop rules \('loss', 10\), \('metric', 0\); fit runs of one rule at a time",
            ):
                load(path, flexfl_schema("comm_bytes_total"))

    def test_load_accepts_one_early_stop_rule(self) -> None:
        frame = pl.read_csv(FIXTURE)
        uniform = frame.with_columns(pl.lit("loss").alias("early_stop_on"), pl.lit(10).alias("min_epochs"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uniform.csv"
            uniform.write_csv(path)
            self.assertEqual(load(path, flexfl_schema("comm_bytes_total")).height, frame.height)

    def test_load_checks_the_early_stop_rule_after_the_task_type_filter(self) -> None:
        frame = pl.read_csv(FIXTURE)
        classification = frame["is_classification"].cast(pl.Boolean)
        by_task = frame.with_columns(
            pl.Series("early_stop_on", ["loss" if flag else "metric" for flag in classification]),
            pl.Series("min_epochs", [10 if flag else 0 for flag in classification]),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "by_task.csv"
            by_task.write_csv(path)
            loaded = load(path, flexfl_schema("comm_bytes_total", "classification"))
            self.assertEqual(loaded.height, int(classification.sum()))
            with self.assertRaisesRegex(SchemaError, "mixes early-stop rules"):
                load(path, flexfl_schema("comm_bytes_total"))

    def test_mixed_epoch_caps_survive_the_constant_drop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mixed = Path(directory) / "mixed.csv"
            frame = pl.read_csv(FIXTURE)
            caps = [10 if index % 2 else 200 for index in range(frame.height)]
            frame.with_columns(pl.Series(FLEXFL_EPOCH_CAP_COLUMN, caps)).write_csv(mixed)
            schema = flexfl_schema("n_epochs")
            reduced = drop_constant_features(load(mixed, schema), schema)
            self.assertIn(FLEXFL_EPOCH_CAP_COLUMN, reduced.model_features)

    def test_log_target_schema(self) -> None:
        schema = flexfl_schema("comm_bytes_total", log_target=True)
        self.assertEqual(schema.slug, "comm_bytes_total-log1p")
        self.assertEqual(schema.label, "log1p(comm_bytes_total)")
        self.assertIsNone(schema.bounds)
        self.assertEqual(
            flexfl_schema("total_time_s", "classification", True).slug, "total_time_s-classification-log1p"
        )
        with self.assertRaises(ValueError):
            flexfl_schema("performance", "classification", True)
        self.assertFalse(MCC_SCHEMA.log_target)
        self.assertEqual(MCC_SCHEMA.label, "MCC")
        self.assertEqual(flexfl_schema("comm_bytes_total").label, "comm_bytes_total")
        self.assertEqual(FLEXFL_COST_TARGETS, (
            "total_time_s", "comm_bytes_total", "compute_time_total_s", "compute_time_max_s",
            "comm_time_total_s", "validation_time_s",
        ))

    def test_decomposition_targets(self) -> None:
        for name in DECOMPOSITION_TARGETS:
            with self.subTest(target=name):
                schema = flexfl_schema(name)
                schema_log = flexfl_schema(name, log_target=True)
                self.assertIsNone(schema.bounds)
                self.assertEqual(schema_log.slug, f"{name}-log1p")
                np.testing.assert_array_equal(
                    target(load(FIXTURE, schema_log), schema_log),
                    np.log1p(target(load(FIXTURE, schema), schema)),
                )

    def blanked(self, directory: str, blanks: dict[str, list[int]]) -> Path:
        frame = pl.read_csv(FIXTURE)
        for column, rows in blanks.items():
            frame = frame.with_columns(
                pl.when(pl.int_range(pl.len()).is_in(rows)).then(None).otherwise(pl.col(column)).alias(column)
            )
        path = Path(directory) / "blanked.csv"
        frame.write_csv(path)
        return path

    def loaded(self, path: Path, schema: Schema) -> tuple[pl.DataFrame, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            frame = load(path, schema)
        return frame, stdout.getvalue(), stderr.getvalue()

    def test_load_drops_rows_with_an_empty_decomposition_target(self) -> None:
        self.assertEqual(FLEXFL_DECOMPOSITION_TARGETS, DECOMPOSITION_TARGETS)
        with tempfile.TemporaryDirectory() as directory:
            for name in DECOMPOSITION_TARGETS:
                for log_target in (False, True):
                    with self.subTest(target=name, log_target=log_target):
                        path = self.blanked(directory, {name: [0, 5]})
                        frame, stdout, stderr = self.loaded(path, flexfl_schema(name, log_target=log_target))
                        self.assertEqual(frame.height, 22)
                        self.assertEqual(frame[name].null_count(), 0)
                        self.assertEqual(stdout, "")
                        self.assertEqual(
                            stderr,
                            f"dropped 2 of 24 rows with an empty {name} from {path} "
                            "(by dataset: clf_a 1, clf_b 1; by fl_algo: CentralizedAsync 1, CentralizedSync 1)\n",
                        )

    def test_load_reports_nothing_when_no_decomposition_target_is_empty(self) -> None:
        frame, stdout, stderr = self.loaded(FIXTURE, flexfl_schema("compute_time_total_s"))
        self.assertEqual(frame.height, 24)
        self.assertEqual((stdout, stderr), ("", ""))

    def test_load_counts_dropped_rows_after_the_task_type_filter(self) -> None:
        indexed = pl.read_csv(FIXTURE).with_row_index()
        classification = indexed.filter(pl.col("is_classification").cast(pl.Boolean))["index"]
        regression = indexed.filter(~pl.col("is_classification").cast(pl.Boolean))["index"]
        schema = flexfl_schema("comm_time_total_s", "classification")
        with tempfile.TemporaryDirectory() as directory:
            path = self.blanked(directory, {"comm_time_total_s": [int(classification[0]), int(regression[0])]})
            frame, stdout, stderr = self.loaded(path, schema)
            self.assertEqual(frame.height, 11)
            self.assertEqual(stdout, "")
            self.assertEqual(
                stderr,
                f"dropped 1 of 12 rows with an empty comm_time_total_s from {path} "
                "(by dataset: clf_a 1; by fl_algo: CentralizedSync 1)\n",
            )
            path = self.blanked(directory, {"comm_time_total_s": [int(regression[0])]})
            frame, stdout, stderr = self.loaded(path, schema)
            self.assertEqual(frame.height, 12)
            self.assertEqual((stdout, stderr), ("", ""))

    def test_load_rejects_a_decomposition_target_with_no_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.blanked(directory, {"validation_time_s": list(range(24))})
            stdout, stderr = io.StringIO(), io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaisesRegex(
                SchemaError, "no rows with a validation_time_s value"
            ):
                load(path, flexfl_schema("validation_time_s"))
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(
                stderr.getvalue(),
                f"dropped 24 of 24 rows with an empty validation_time_s from {path} "
                "(by dataset: clf_a 4, clf_b 4, clf_c 4, reg_a 4, reg_b 4, reg_c 4; "
                "by fl_algo: CentralizedAsync 6, CentralizedSync 6, DecentralizedAsync 6, DecentralizedSync 6)\n",
            )

    def test_load_returns_an_empty_frame_for_a_header_only_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header.csv"
            pl.read_csv(FIXTURE).head(0).write_csv(path)
            frame, stdout, stderr = self.loaded(path, flexfl_schema("compute_time_max_s"))
            self.assertEqual(frame.height, 0)
            self.assertEqual((stdout, stderr), ("", ""))

    def test_load_still_rejects_empty_features_and_older_targets(self) -> None:
        classification = pl.read_csv(FIXTURE).with_row_index().filter(
            pl.col("is_classification").cast(pl.Boolean)
        )["index"]
        cases = {
            "feature beside an empty decomposition target": (
                {"n_samples": [0], "compute_time_max_s": [0]}, flexfl_schema("compute_time_max_s"),
            ),
            "feature with a decomposition target": ({"n_samples": [1]}, flexfl_schema("compute_time_max_s")),
            "performance": ({"performance": [int(classification[0])]}, flexfl_schema("performance", "classification")),
            "total_time_s": ({"total_time_s": [0]}, flexfl_schema("total_time_s")),
            "comm_bytes_total": ({"comm_bytes_total": [0]}, flexfl_schema("comm_bytes_total")),
            "n_epochs": ({"n_epochs": [0]}, flexfl_schema("n_epochs")),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, (blanks, schema) in cases.items():
                with self.subTest(name):
                    path = self.blanked(directory, blanks)
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaisesRegex(
                        SchemaError, "null values"
                    ):
                        load(path, schema)
                    self.assertEqual((stdout.getvalue(), stderr.getvalue()), ("", ""))

    def test_decomposition_target_cli_outputs(self) -> None:
        for name in DECOMPOSITION_TARGETS:
            with self.subTest(target=name), tempfile.TemporaryDirectory() as directory:
                out = Path(directory)
                self.assertEqual(cli.main([
                    "--target", name, "--log-target", "--data", str(FIXTURE), "--output", str(out),
                    "--quiet", "--max-terms", "2", "--pool", "20", "--beam", "1", "--arity", "1",
                ]), 0)
                equation = out / "flexfl" / f"{name}-log1p" / "equation.txt"
                self.assertTrue(equation.read_text().startswith(f"log1p({name}) = "))
                search_out = out / "search"
                self.assertEqual(configuration_search.main([*search_argv(search_out, name), "--log-target"]), 0)
                manifest = json.loads((search_out / "manifest.json").read_text())
                self.assertEqual(manifest["settings"]["target"], name)

    def test_unchosen_decomposition_columns_are_not_targets(self) -> None:
        for column in ("comm_time_max_s", "serial_time_total_s", "comm_skew_clamped"):
            with self.subTest(target=column):
                stderr = io.StringIO()
                with self.assertRaises(SystemExit) as error, redirect_stderr(stderr):
                    cli.main(["--target", column, "--data", str(FIXTURE)])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("invalid choice", stderr.getvalue())

    def test_log_target_schema_requires_unbounded_target(self) -> None:
        with self.assertRaises(ValueError):
            dataclasses.replace(flexfl_schema("performance", "classification"), log_target=True)
        dataclasses.replace(flexfl_schema("comm_bytes_total"), log_target=True)

    def test_search_schema_rejects_log_mcc(self) -> None:
        with self.assertRaises(ValueError):
            SearchSettings(target="mcc", log_target=True).schema()

    def test_target_applies_log1p(self) -> None:
        for name in ("total_time_s", "comm_bytes_total"):
            with self.subTest(target=name):
                frame = load(FIXTURE, flexfl_schema(name))
                np.testing.assert_array_equal(
                    target(frame, flexfl_schema(name, log_target=True)), np.log1p(target(frame, flexfl_schema(name)))
                )

    def test_log_target_rejects_negative_costs(self) -> None:
        frame = load(FIXTURE, flexfl_schema("total_time_s"))
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value):
                bad = frame.with_columns(pl.lit(value).alias("total_time_s"))
                with self.assertRaises(SchemaError):
                    target(bad, flexfl_schema("total_time_s", log_target=True))
                if value == -1.0:
                    np.testing.assert_array_equal(
                        target(bad, flexfl_schema("total_time_s")), np.full(frame.height, -1.0)
                    )
        zero = frame.with_columns(pl.lit(0.0).alias("total_time_s"))
        np.testing.assert_array_equal(
            target(zero, flexfl_schema("total_time_s", log_target=True)), np.zeros(frame.height)
        )

    def test_run_equation_prunes_with_the_schema_threshold(self) -> None:
        schemas = (
            flexfl_schema("comm_bytes_total"),
            flexfl_schema("comm_bytes_total", log_target=True),
            flexfl_schema("performance", "classification"),
        )
        for schema in schemas:
            with self.subTest(schema=schema), mock.patch.object(experiment, "prune", wraps=experiment.prune) as spy:
                experiment.run_flexfl(FIXTURE, schema, TINY)
                actual = spy.call_args.kwargs["min_contribution"]
                truth = target(load(FIXTURE, schema), schema)
                expected = pruning_threshold(schema.bounds, truth)
                self.assertEqual(actual, expected)
                if schema.bounds is None:
                    self.assertNotEqual(actual, MIN_CONTRIBUTION)
                else:
                    self.assertEqual(actual, MIN_CONTRIBUTION)

    def test_search_worker_prunes_with_the_schema_threshold(self) -> None:
        settings = SearchSettings(
            target="comm_bytes_total",
            minimum_features=N_MODEL_FEATURES,
            maximum_features=N_MODEL_FEATURES,
            penalties=(20.0,),
            zscores=(3.0,),
            arities=(1,),
            minimum_terms=1,
            maximum_terms=2,
            pool_size=20,
            beam_width=1,
            shortlist_top=1,
        )
        frame = load(FIXTURE, settings.schema())
        for current in (settings, dataclasses.replace(settings, log_target=True)):
            with self.subTest(log_target=current.log_target):
                with mock.patch.object(configuration_search, "prune", wraps=configuration_search.prune) as spy:
                    configuration_search._evaluate_library_point(library_points(current)[0], current, frame)
                self.assertGreater(spy.call_count, 0)
                expected = pruning_threshold(None, target(frame, current.schema()))
                for call in spy.call_args_list:
                    self.assertEqual(call.kwargs["min_contribution"], expected)

    def test_finalist_worker_prunes_with_the_schema_threshold(self) -> None:
        settings = SearchSettings(
            target="comm_bytes_total",
            minimum_features=N_MODEL_FEATURES,
            maximum_features=N_MODEL_FEATURES,
            penalties=(20.0,),
            zscores=(3.0,),
            arities=(1,),
            minimum_terms=1,
            maximum_terms=2,
            pool_size=20,
            beam_width=1,
            shortlist_top=1,
        )
        frame = load(FIXTURE, settings.schema())
        point = configuration_search.BasePoint(0, tuple(sorted(FLEXFL_MODEL_FEATURES)), 20.0, 3.0)
        for current in (settings, dataclasses.replace(settings, log_target=True)):
            with self.subTest(log_target=current.log_target):
                with mock.patch.object(configuration_search, "prune", wraps=configuration_search.prune) as spy:
                    configuration_search._evaluate_finalist(point, current, frame)
                self.assertGreater(spy.call_count, 0)
                expected = pruning_threshold(None, target(frame, current.schema()))
                for call in spy.call_args_list:
                    self.assertEqual(call.kwargs["min_contribution"], expected)

    def test_run_flexfl_on_a_log_target(self) -> None:
        schema = flexfl_schema("comm_bytes_total", log_target=True)
        result = experiment.run_flexfl(FIXTURE, schema, TINY)
        self.assertIsNone(result.equation.equation.bounds)
        self.assertTrue(result.equation.equation.name.startswith("E3_log1p_k"))
        frame = load(FIXTURE, schema)
        columns = columns_as_arrays(frame, result.schema.features)
        prediction = result.equation.equation.predict(columns)
        self.assertAlmostEqual(
            result.equation.in_sample["r2"],
            score(np.log1p(frame["comm_bytes_total"].to_numpy()), prediction).r2,
            delta=1e-12,
        )
        self.assertLess(np.abs(prediction).max(), 50.0)
        for row in result.effects.iter_rows(named=True):
            expected = "raises" if row["beta"] > 0 else "lowers"
            self.assertEqual(row["direction"], f"{expected} log1p(comm_bytes_total)")

    def test_run_flexfl_on_a_log_target_with_task_type(self) -> None:
        schema = flexfl_schema("total_time_s", "classification", log_target=True)
        result = experiment.run_flexfl(FIXTURE, schema, TINY)
        self.assertIsNone(result.equation.equation.bounds)
        self.assertTrue(result.equation.equation.name.startswith("E3_log1p_k"))
        for row in result.effects.iter_rows(named=True):
            expected = "raises" if row["beta"] > 0 else "lowers"
            self.assertEqual(row["direction"], f"{expected} log1p(total_time_s)")

    def test_log_target_argument_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for entry, argv in (
                (cli.main, ["--log-target"]),
                (
                    cli.main,
                    [
                        "--target",
                        "performance",
                        "--task-type",
                        "classification",
                        "--data",
                        str(FIXTURE),
                        "--log-target",
                    ],
                ),
                (configuration_search.main, ["all", "--log-target", "--output", str(out)]),
                (
                    configuration_search.main,
                    [*search_argv(out, "performance"), "--task-type", "classification", "--log-target"],
                ),
            ):
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error:
                    entry(argv)
                self.assertEqual(error.exception.code, 2)
        with self.assertRaises(ValueError):
            SearchSettings(log_target=True).validate()

    def test_study_cli_writes_log_target_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            argv = [
                "--target",
                "comm_bytes_total",
                "--data",
                str(FIXTURE),
                "--output",
                str(out),
                "--quiet",
                "--max-terms",
                "2",
                "--pool",
                "20",
                "--beam",
                "1",
                "--arity",
                "1",
            ]
            self.assertEqual(cli.main(argv), 0)
            self.assertEqual(cli.main([*argv, "--log-target"]), 0)
            folder = out / "flexfl" / "comm_bytes_total-log1p"
            for name in ("equation.json", "equation.txt", "curve.csv", "term_effects.csv", "group_shares.csv"):
                self.assertTrue((folder / name).is_file(), name)
            self.assertTrue((folder / "equation.txt").read_text().startswith("log1p(comm_bytes_total) = "))
            self.assertIsNone(Equation.load(folder / "equation.json").bounds)
            self.assertTrue(
                (out / "flexfl" / "comm_bytes_total" / "equation.txt").read_text().startswith("comm_bytes_total = ")
            )

    def test_search_cli_runs_a_log_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertEqual(configuration_search.main([*search_argv(out), "--log-target"]), 0)
            for key in ("e3_valid", "e3_max"):
                with self.subTest(key=key):
                    self.assertTrue((out / f"{key}.txt").read_text().startswith("log1p(comm_bytes_total) = "))
                    equation = Equation.load(out / f"{key}.json")
                    self.assertIsNone(equation.bounds)
                    self.assertTrue(equation.name.startswith("E3_log1p_k"))
            self.assertIs(json.loads((out / "manifest.json").read_text())["settings"]["log_target"], True)
            with self.assertRaises(RuntimeError):
                configuration_search.main(search_argv(out))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            self.assertEqual(configuration_search.main(search_argv(out)), 0)
            with self.assertRaises(RuntimeError):
                configuration_search.main([*search_argv(out), "--log-target"])

    def test_mcc_schema_does_not_transform_target(self) -> None:
        self.assertEqual(MCC_SCHEMA.label, MCC_SCHEMA.target_column)
        self.assertEqual(MCC_SCHEMA.slug, MCC_SCHEMA.target_column)
        with mock.patch.object(np, "log1p", wraps=np.log1p) as spy:
            target(load(None, MCC_SCHEMA), MCC_SCHEMA)
        self.assertEqual(spy.call_count, 0)

    def test_log_target_settings_payload(self) -> None:
        self.assertNotIn("log_target", _settings_payload(SearchSettings()))
        self.assertNotIn(
            "log_target",
            _settings_payload(
                SearchSettings(
                    target="comm_bytes_total", minimum_features=N_MODEL_FEATURES, maximum_features=N_MODEL_FEATURES
                )
            ),
        )
        self.assertIs(
            _settings_payload(
                SearchSettings(
                    target="comm_bytes_total", minimum_features=N_MODEL_FEATURES, maximum_features=N_MODEL_FEATURES,
                    log_target=True,
                )
            )["log_target"],
            True,
        )

    def test_load_accepts_each_target(self) -> None:
        for target_name, task_type, rows in (
            ("total_time_s", None, 24), ("comm_bytes_total", None, 24),
            ("performance", "classification", 12), ("performance", "regression", 12),
            ("n_epochs", None, 24), ("n_epochs", "classification", 12),
            ("compute_time_total_s", None, 24), ("compute_time_max_s", None, 24),
            ("comm_time_total_s", None, 24), ("validation_time_s", None, 24),
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
        settings = SearchSettings(
            target="comm_bytes_total", minimum_features=N_MODEL_FEATURES, maximum_features=N_MODEL_FEATURES
        )
        self.assertEqual(feature_subsets(settings), [tuple(sorted(FLEXFL_MODEL_FEATURES))])
        SearchSettings(target="comm_bytes_total", minimum_features=1, maximum_features=N_MODEL_FEATURES,
                       explicit_feature_sets=(("learning_rate", "num_workers"),)).validate()
        with self.assertRaisesRegex(ValueError, "Model Capability"):
            SearchSettings(target="comm_bytes_total", minimum_features=1, maximum_features=N_MODEL_FEATURES,
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

    def test_n_epochs_target(self) -> None:
        schema = flexfl_schema("n_epochs")
        self.assertIsNone(schema.bounds)
        self.assertEqual((schema.slug, schema.label), ("n_epochs", "n_epochs"))
        self.assertEqual(flexfl_schema("n_epochs", "regression").slug, "n_epochs-regression")
        frame = load(FIXTURE, schema)
        self.assertEqual(frame.height, 24)
        self.assertEqual(frame["n_epochs"].dtype, pl.Float64)
        with self.assertRaisesRegex(ValueError, "log1p applies only to the cost targets"):
            flexfl_schema("n_epochs", log_target=True)
        for name in ("comm_bytes_sent", "comm_bytes_recv"):
            with self.subTest(target=name), self.assertRaises(ValueError):
                flexfl_schema(name)

    def test_run_flexfl_on_n_epochs(self) -> None:
        schema = flexfl_schema("n_epochs")
        with mock.patch.object(experiment, "prune", wraps=experiment.prune) as spy:
            result = experiment.run_flexfl(FIXTURE, schema, TINY)
        truth = target(load(FIXTURE, schema), schema)
        self.assertEqual(spy.call_args.kwargs["min_contribution"], pruning_threshold(None, truth))
        self.assertIsNone(result.equation.equation.bounds)
        self.assertNotIn("log1p", result.equation.equation.name)
        self.assertGreater(result.effects.height, 0)
        for row in result.effects.iter_rows(named=True):
            expected = "raises" if row["beta"] > 0 else "lowers"
            self.assertEqual(row["direction"], f"{expected} n_epochs")

    def test_n_epochs_cli_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            code = cli.main([
                "--target", "n_epochs", "--data", str(FIXTURE), "--output", str(out), "--quiet",
                "--max-terms", "2", "--pool", "20", "--beam", "1", "--arity", "1",
            ])
            self.assertEqual(code, 0)
            folder = out / "flexfl" / "n_epochs"
            self.assertTrue((folder / "equation.txt").read_text().startswith("n_epochs = "))
            self.assertIsNone(Equation.load(folder / "equation.json").bounds)
            search_out = out / "search"
            self.assertEqual(configuration_search.main(search_argv(search_out, "n_epochs")), 0)
            self.assertTrue((search_out / "e3_valid.txt").read_text().startswith("n_epochs = "))
            manifest = json.loads((search_out / "manifest.json").read_text())
            self.assertEqual(manifest["settings"]["target"], "n_epochs")

    def test_n_epochs_argument_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            log_error = (
                "--log-target applies only to --target total_time_s or comm_bytes_total or "
                "compute_time_total_s or compute_time_max_s or comm_time_total_s or validation_time_s"
            )
            for entry, argv, messages in (
                (cli.main, ["--target", "n_epochs", "--data", str(FIXTURE), "--log-target"], (log_error,)),
                (configuration_search.main, [*search_argv(out, "n_epochs"), "--log-target"], (log_error,)),
                (cli.main, ["--target", "comm_bytes_sent", "--data", str(FIXTURE)],
                 ("invalid choice", "comm_bytes_sent")),
                (configuration_search.main, search_argv(out, "comm_bytes_recv"), ("invalid choice", "comm_bytes_recv")),
            ):
                stderr = io.StringIO()
                with self.subTest(argv=argv), self.assertRaises(SystemExit) as error, redirect_stderr(stderr):
                    entry(argv)
                self.assertEqual(error.exception.code, 2)
                for message in messages:
                    self.assertIn(message, stderr.getvalue())
