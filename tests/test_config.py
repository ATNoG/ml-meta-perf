"""Tests for the study configuration file and its schema."""

import json
import tempfile
import unittest
from pathlib import Path

from ml_meta_perf.config import DEFAULT_CONFIG_PATH, ConfigError, Schema, load_config, parse_config


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
        self.assertFalse(schema.log_target)
        self.assertIsNone(schema.task_type)

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
            "same group column": ("corpus", "model_column", "Dataset", "two different"),
            "empty group column": ("corpus", "dataset_column", "", "two different"),
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

    def test_a_config_error_is_a_value_error(self) -> None:
        self.assertTrue(issubclass(ConfigError, ValueError))

    def test_the_description_survives_a_round_trip_through_json(self) -> None:
        """A sweep proposal is `dataclasses.asdict` of the study; it must parse back to the same study."""
        import dataclasses

        study = load_config()
        self.assertEqual(parse_config(json.loads(json.dumps(dataclasses.asdict(study)))), study)


if __name__ == "__main__":
    unittest.main()
