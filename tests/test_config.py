"""Tests for the study configuration file and its schema."""

import json
import tempfile
import unittest
from pathlib import Path

from ml_meta_perf.config import DEFAULT_CONFIG_PATH, load_config, parse_config


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


if __name__ == "__main__":
    unittest.main()
