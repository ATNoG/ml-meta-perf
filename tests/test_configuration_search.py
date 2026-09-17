"""The cluster recalibration has a stable grid, identity and end-to-end contract."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from ml_meta_perf.configuration_search import (
    DEFAULT_MIN_TERMS,
    SearchSettings,
    _select_plateau_valid,
    feature_subsets,
    grid_table,
    historical_objective,
    initialise,
    main,
)
from ml_meta_perf.model import Equation
from tests.corpus import sample_path


class ConfigurationSearchTests(unittest.TestCase):
    def test_default_grid_contains_every_nontrivial_descriptor_subset(self) -> None:
        settings = SearchSettings()
        subsets = feature_subsets(settings)

        self.assertEqual(len(subsets), 57)
        self.assertIn(("Loss Margin Behaviour", "Solution Stochasticity"), subsets)
        self.assertEqual(grid_table(settings).height, 10_944)

    def test_historical_objective_preserves_weights_and_clamps_negative_r2(self) -> None:
        perfect = {
            "in_sample_r2": 1.0,
            "loo_dataset_r2": 1.0,
            "loo_model_r2": 1.0,
            "binary": 1.0,
            "ranking": 1.0,
            "stability": 1.0,
            "brevity": 1.0,
        }
        negative = perfect | {
            "in_sample_r2": -2.0,
            "loo_dataset_r2": -3.0,
            "loo_model_r2": -4.0,
        }

        self.assertAlmostEqual(historical_objective(perfect), 1.0)
        self.assertAlmostEqual(historical_objective(negative), 0.5)

    def test_manifest_refuses_to_mix_different_searches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            initialise(output, sample_path(), SearchSettings(penalties=(20.0,)))

            with self.assertRaisesRegex(RuntimeError, "different search"):
                initialise(output, sample_path(), SearchSettings(penalties=(10.0,)))

    def test_default_search_starts_at_one_term(self) -> None:
        self.assertEqual(DEFAULT_MIN_TERMS, 1)
        self.assertEqual(SearchSettings().minimum_terms, 1)

    def test_plateau_rule_selects_the_best_equation_before_stalling(self) -> None:
        scores = (0.2, 0.4, 0.6, 0.6002, 0.6004, 0.7)
        shared = pl.DataFrame(
            {
                "n_terms": range(1, 7),
                "requested_terms": range(1, 7),
                "max_arity": [2] * 6,
                "complexity": range(2, 14, 2),
                "combined_r2": scores,
                "four_protocol_floor": scores,
            }
        )

        selected, diagnostic = _select_plateau_valid(shared, tolerance=0.001, window=2)

        self.assertEqual(selected["n_terms"], 3)
        self.assertEqual(diagnostic["plateau_starts_at_terms"], 3)

    def test_small_search_runs_every_stage_and_writes_reloadable_equations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            code = main(
                [
                    "all",
                    "--data",
                    str(sample_path()),
                    "--output",
                    str(output),
                    "--jobs",
                    "1",
                    "--penalty",
                    "20",
                    "--zscore",
                    "3",
                    "--arity",
                    "2",
                    "--feature-set",
                    "Model Capability,Processing Units Number",
                    "--min-terms",
                    "2",
                    "--max-terms",
                    "3",
                    "--pool",
                    "40",
                    "--beam",
                    "2",
                    "--shortlist-top",
                    "1",
                ]
            )

            self.assertEqual(code, 0)
            self.assertEqual(Equation.load(output / "e3_valid.json").n_terms, 3)
            self.assertEqual(Equation.load(output / "e3_max.json").n_terms, 3)
            self.assertTrue((output / "equation_search.csv").is_file())
            self.assertTrue((output / "finalists.csv").is_file())
            summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["stage"], "all")
            self.assertGreaterEqual(summary["elapsed_seconds"], 0.0)
            selection_code = main(
                [
                    "select",
                    "--data",
                    str(sample_path()),
                    "--output",
                    str(output),
                    "--penalty",
                    "20",
                    "--zscore",
                    "3",
                    "--arity",
                    "2",
                    "--feature-set",
                    "Model Capability,Processing Units Number",
                    "--min-terms",
                    "2",
                    "--max-terms",
                    "3",
                    "--pool",
                    "40",
                    "--beam",
                    "2",
                    "--shortlist-top",
                    "1",
                ]
            )

            self.assertEqual(selection_code, 0)
            selected = json.loads((output / "selected_configurations.json").read_text(encoding="utf-8"))
            self.assertEqual(
                selected["e3_valid_diagnostics"]["implementation"],
                "best-so-far gain over a forward window",
            )


if __name__ == "__main__":
    unittest.main()
