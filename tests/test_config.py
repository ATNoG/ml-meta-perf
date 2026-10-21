"""Tests for the study configuration file and its schema."""

import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path

from ml_meta_perf.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    RequiredColumn,
    RowFilter,
    Schema,
    load_config,
    parse_config,
)


class TestStudyConfiguration(unittest.TestCase):
    """``config/study.json`` is the only source of the tuned hyperparameters."""

    def setUp(self) -> None:
        self.payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_the_shipped_file_parses_into_every_section(self) -> None:
        study = load_config()
        self.assertEqual(study, parse_config(self.payload))
        self.assertLessEqual(study.search.max_arity, study.selection.capability_arity)
        self.assertGreater(study.search.max_terms, study.sweep.readable_terms)

    def test_an_unknown_key_is_an_error_not_a_silent_default(self) -> None:
        self.payload["search"]["pentalty"] = 1.0
        with self.assertRaises(TypeError):
            parse_config(self.payload)

    def test_a_missing_key_is_an_error(self) -> None:
        del self.payload["selection"]["delta"]
        with self.assertRaises(TypeError):
            parse_config(self.payload)

    def test_a_missing_file_names_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.json"
            with self.assertRaisesRegex(FileNotFoundError, "absent.json"):
                load_config(missing)


class TestStudyDescription(unittest.TestCase):
    """The ``corpus``, ``response`` and ``features`` sections: what the corpus is, not how to fit it."""

    def setUp(self) -> None:
        self.payload = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))

    def test_the_sections_are_read_into_their_types(self) -> None:
        study = parse_config(self.payload)
        self.assertEqual(study.corpus.path, "dataset/meta_dataset.csv")
        self.assertEqual((study.corpus.dataset_column, study.corpus.model_column), ("Dataset", "Model"))
        self.assertEqual(study.response.column, "MCC")
        self.assertEqual(study.response.bounds, (-1.0, 1.0))
        self.assertIsInstance(study.features.dataset, tuple)
        self.assertEqual(len(study.features.dataset), 12)
        self.assertEqual(len(study.features.model), 6)

    def test_the_schema_is_the_description_in_runtime_form(self) -> None:
        schema = parse_config(self.payload).schema()
        self.assertIsInstance(schema, Schema)
        self.assertEqual(schema.target_column, "MCC")
        self.assertEqual(schema.features, schema.dataset_features + schema.model_features)
        self.assertEqual(schema.transform, "none")
        self.assertIsNone(schema.row_filter)

    def test_a_target_without_bounds_is_unbounded(self) -> None:
        self.payload["response"] = {"column": "total_time_s", "bounds": None}
        study = parse_config(self.payload)
        self.assertIsNone(study.response.bounds)
        self.assertIsNone(study.schema().bounds)

    def test_a_missing_section_is_an_error(self) -> None:
        for section in ("corpus", "response", "features"):
            payload = {key: value for key, value in self.payload.items() if key != section}
            with self.assertRaises(KeyError, msg=section):
                parse_config(payload)

    def test_an_unknown_key_in_a_new_section_is_an_error(self) -> None:
        self.payload["corpus"]["pth"] = "x.csv"
        with self.assertRaises(TypeError):
            parse_config(self.payload)

    def test_every_way_the_description_can_be_unusable_is_named(self) -> None:
        cases = {
            "empty path": ("corpus", "path", "", "corpus.path is empty"),
            "same group column": ("corpus", "model_column", "Dataset", "differs from it"),
            "empty group column": ("corpus", "dataset_column", "", "non-empty dataset column"),
            "inverted bounds": ("response", "bounds", [1.0, -1.0], "low < high"),
            "equal bounds": ("response", "bounds", [0.5, 0.5], "low < high"),
            "no dataset features": ("features", "dataset", [], "features.dataset is empty"),
            "no model features": ("features", "model", [], "features.model is empty"),
            "feature listed twice": ("features", "model", ["Fitting Regime", "Fitting Regime"], "more than once"),
            "feature in both lists": ("features", "model", ["nr_attr"], "more than once"),
            "feature is the response": ("features", "model", ["MCC"], "response"),
            "feature is a group": ("features", "dataset", ["Dataset"], "group"),
            "response is a group": ("response", "column", "Model", "also a group column"),
        }
        for name, (section, key, value, message) in cases.items():
            payload = json.loads(json.dumps(self.payload))
            payload[section][key] = value
            with self.subTest(name), self.assertRaisesRegex(ConfigError, message):
                parse_config(payload)

    def test_every_way_the_protocols_can_be_unusable_is_named(self) -> None:
        cases = {
            "unknown": (["IS", "LODO", "KFOLD"], "distinct names"),
            "repeated": (["IS", "LODO", "LODO"], "distinct names"),
            "no fit": (["LODO", "LOMO"], "include IS"),
            "nothing held out": (["IS"], "at least one held-out"),
        }
        for name, (protocols, message) in cases.items():
            payload = json.loads(json.dumps(self.payload))
            payload["protocols"] = protocols
            with self.subTest(name), self.assertRaisesRegex(ConfigError, message):
                parse_config(payload)

    def test_a_protocol_that_holds_a_model_out_needs_a_model_column(self) -> None:
        for protocols in (["IS", "LODO", "LOMO"], ["IS", "DHO"]):
            payload = json.loads(json.dumps(self.payload))
            payload["corpus"]["model_column"] = None
            payload["protocols"] = protocols
            with self.subTest(protocols), self.assertRaisesRegex(ConfigError, "corpus.model_column is null"):
                parse_config(payload)

    def test_a_corpus_without_a_model_column_runs_the_dataset_protocols(self) -> None:
        self.payload["corpus"]["model_column"] = None
        self.payload["protocols"] = ["IS", "LODO"]
        study = parse_config(self.payload)
        self.assertEqual((study.corpus.model_column, study.schema().protocols), (None, ("IS", "LODO")))
        self.assertEqual(study.schema().group_columns, ("Dataset",))
        with self.assertRaisesRegex(ValueError, "no model column"):
            _ = study.schema().model_group

    def test_the_mcc_study_is_scored_under_all_four_protocols(self) -> None:
        self.assertEqual(load_config().protocols, ("IS", "LODO", "LOMO", "DHO"))

    def test_the_flexfl_studies_are_scored_without_a_held_out_model(self) -> None:
        for path in sorted((Path(__file__).resolve().parents[1] / "config" / "flexfl").glob("*.json")):
            with self.subTest(path.name):
                self.assertEqual(load_config(path).protocols, ("IS", "LODO"))

    def test_every_way_the_grammar_exclusions_can_be_unusable_is_named(self) -> None:
        cases = {
            "unknown feature": ({"nope": ["log"]}, "listed feature"),
            "unknown operation": ({"nr_attr": ["cube"]}, "distinct operations"),
            "identity is not excludable": ({"nr_attr": ["id"]}, "distinct operations"),
            "empty list": ({"nr_attr": []}, "distinct operations"),
            "repeated": ({"nr_attr": ["log", "log"]}, "distinct operations"),
        }
        for name, (exclude, message) in cases.items():
            payload = json.loads(json.dumps(self.payload))
            payload["grammar"]["exclude"] = exclude
            with self.subTest(name), self.assertRaisesRegex(ConfigError, message):
                parse_config(payload)

    def test_exclusions_reach_the_schema_sorted_by_feature(self) -> None:
        self.payload["grammar"]["exclude"] = {"nr_inst": ["inv", "log"], "class_ent": ["sq"]}
        schema = parse_config(self.payload).schema()
        self.assertEqual(schema.exclusions, (("class_ent", ("sq",)), ("nr_inst", ("inv", "log"))))
        self.assertEqual(schema.excluded, {"class_ent": ("sq",), "nr_inst": ("inv", "log")})

    def test_the_shipped_studies_exclude_nothing(self) -> None:
        root = Path(__file__).resolve().parents[1] / "config"
        for path in (root / "study.json", *sorted((root / "flexfl").glob("*.json"))):
            with self.subTest(path.name):
                self.assertEqual(load_config(path).grammar.exclude, {})

    def test_the_shipped_study_fits_the_three_equations_and_the_ceiling_on_an_untransformed_target(self) -> None:
        study = load_config()
        self.assertEqual(study.equations, ("E1", "E2", "E3", "E3-MAX"))
        self.assertEqual(study.response.transform, "none")
        self.assertIsNone(study.corpus.row_filter)
        self.assertEqual((study.corpus.require, study.corpus.single_valued), ((), ()))
        self.assertFalse(study.corpus.drop_missing_response or study.corpus.drop_constant_features)

    def test_the_transform_is_optional_and_defaults_to_none(self) -> None:
        del self.payload["response"]["transform"]
        self.assertEqual(parse_config(self.payload).response.transform, "none")

    def test_a_log_target_is_read_and_reaches_the_schema(self) -> None:
        self.payload["response"] = {"column": "total_time_s", "bounds": None, "transform": "log1p"}
        schema = parse_config(self.payload).schema()
        self.assertEqual(
            (schema.transform, schema.log_target, schema.label, schema.slug),
            ("log1p", True, "log1p(total_time_s)", "total_time_s-log1p"),
        )
        self.assertEqual(schema.tag("E3"), "E3_log1p")

    def test_a_transform_that_does_not_exist_or_cannot_apply_is_an_error(self) -> None:
        self.payload["response"]["transform"] = "sqrt"
        with self.assertRaisesRegex(ConfigError, "response.transform must be one of"):
            parse_config(self.payload)
        self.payload["response"]["transform"] = "log1p"  # the MCC target can be negative
        with self.assertRaisesRegex(ConfigError, "cannot be negative"):
            parse_config(self.payload)

    def test_a_response_key_that_does_not_exist_is_an_error(self) -> None:
        self.payload["response"]["scale"] = "log"
        with self.assertRaisesRegex(TypeError, "unexpected response keys"):
            parse_config(self.payload)

    def test_the_equations_are_validated(self) -> None:
        for bad in ([], ["E4"], ["E3", "E3"], ["e3"]):
            self.payload["equations"] = bad
            with self.subTest(bad), self.assertRaisesRegex(ConfigError, "equations must be"):
                parse_config(self.payload)
        for good in (["E3"], ["E3", "E3-MAX"]):
            self.payload["equations"] = good
            self.assertEqual(parse_config(self.payload).equations, tuple(good))
        for odd in (["E1", "E2"], ["E3-MAX"], ["E3-MAX", "E3"]):
            self.payload["equations"] = odd
            with self.subTest(odd), self.assertRaisesRegex(ConfigError, "equations must be"):
                parse_config(self.payload)

    def test_a_study_without_equations_is_an_error(self) -> None:
        del self.payload["equations"]
        with self.assertRaises(KeyError):
            parse_config(self.payload)

    def test_the_corpus_options_are_read_and_reach_the_schema(self) -> None:
        self.payload["corpus"].update(
            {
                "row_filter": {"column": "is_classification", "equals": True, "label": "classification"},
                "require": [{"column": "epoch_cap", "hint": "rebuild"}, {"column": "other"}],
                "single_valued": ["early_stop_on", "min_epochs"],
                "drop_missing_response": True,
                "drop_constant_features": True,
            }
        )
        study = parse_config(self.payload)
        self.assertEqual(study.corpus.row_filter, RowFilter("is_classification", True, "classification"))
        self.assertEqual(study.corpus.require, (RequiredColumn("epoch_cap", "rebuild"), RequiredColumn("other")))
        schema = study.schema()
        self.assertEqual(schema.single_valued, ("early_stop_on", "min_epochs"))
        self.assertTrue(schema.drop_missing_response and schema.drop_constant_features)
        self.assertEqual(schema.slug, "MCC-classification")

    def test_the_corpus_options_survive_a_round_trip_through_json(self) -> None:
        self.payload["corpus"].update(
            {
                "row_filter": {"column": "k", "equals": "a"},
                "require": [{"column": "c", "hint": "h"}],
                "single_valued": ["s"],
            }
        )
        study = parse_config(self.payload)
        self.assertEqual(parse_config(json.loads(json.dumps(dataclasses.asdict(study)))), study)

    def test_bad_corpus_options_are_named(self) -> None:
        cases = {
            "unknown filter key": ({"row_filter": {"column": "k", "equal": 1}}, TypeError, "equal"),
            "empty filter column": (
                {"row_filter": {"column": "", "equals": 1}},
                ConfigError,
                "row_filter.column is empty",
            ),
            "empty requirement": ({"require": [{"column": ""}]}, ConfigError, "require holds an empty"),
            "empty single valued": ({"single_valued": [""]}, ConfigError, "single_valued holds an empty"),
        }
        for name, (options, error, message) in cases.items():
            payload = json.loads(json.dumps(self.payload))
            payload["corpus"].update(options)
            with self.subTest(name), self.assertRaisesRegex(error, message):
                parse_config(payload)

    def test_the_schema_rejects_a_log_scale_on_a_bounded_target_and_unknown_transforms(self) -> None:
        base = parse_config(self.payload).schema()
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            dataclasses.replace(base, transform="log1p")
        with self.assertRaisesRegex(ValueError, "unknown transform"):
            dataclasses.replace(base, transform="cbrt")

    def test_bounds_may_be_open_on_either_side(self) -> None:
        cases = {
            "time": ([0, None], (0.0, None)),
            "ceiling": ([None, 5], (None, 5.0)),
            "both": ([-1, 1], (-1.0, 1.0)),
            "open": ([None, None], None),
            "none": (None, None),
        }
        for name, (raw, expected) in cases.items():
            self.payload["response"]["bounds"] = raw
            with self.subTest(name):
                self.assertEqual(parse_config(self.payload).response.bounds, expected)

    def test_bounds_must_be_a_pair(self) -> None:
        for bad in ([1], [1, 2, 3], "0,1", 5):
            self.payload["response"]["bounds"] = bad
            with self.subTest(bad), self.assertRaisesRegex(ConfigError, "null or a .low, high. pair"):
                parse_config(self.payload)

    def test_a_log_scale_needs_a_target_that_cannot_be_negative(self) -> None:
        self.payload["response"]["transform"] = "log1p"
        for bounds, ok in (([0, None], True), ([2, 9], True), (None, True), ([None, 5], False), ([-1, 1], False)):
            self.payload["response"]["bounds"] = bounds
            with self.subTest(bounds):
                if ok:
                    self.assertEqual(parse_config(self.payload).response.transform, "log1p")
                else:
                    with self.assertRaisesRegex(ConfigError, "cannot be negative"):
                        parse_config(self.payload)

    def test_the_fit_bounds_are_the_bounds_on_the_scale_the_equation_is_fitted(self) -> None:
        base = parse_config(self.payload).schema()
        self.assertEqual(base.fit_bounds, (-1.0, 1.0))
        timed = dataclasses.replace(base, bounds=(0.0, None))
        self.assertEqual(timed.fit_bounds, (0.0, None))
        logged = dataclasses.replace(timed, transform="log1p")
        self.assertEqual(logged.fit_bounds, (0.0, None))
        capped = dataclasses.replace(base, bounds=(0.0, math.e - 1.0), transform="log1p")
        self.assertEqual(capped.fit_bounds, (0.0, 1.0))
        self.assertIsNone(dataclasses.replace(base, bounds=None, transform="log1p").fit_bounds)

    def test_a_config_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(ConfigError, ValueError))

    def test_the_description_survives_a_round_trip_through_json(self) -> None:
        """A sweep proposal is `dataclasses.asdict` of the study; it must parse back to the same study."""
        study = load_config()
        self.assertEqual(parse_config(json.loads(json.dumps(dataclasses.asdict(study)))), study)


if __name__ == "__main__":
    unittest.main()
