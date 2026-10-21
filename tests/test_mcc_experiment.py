"""The MCC experiment (``config/mcc.json``) keeps the values it had at the start of the work."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf.config import load_config
from ml_meta_perf.selection import floor_curve
from tests.regression import EXPECTED, VARIANTS, compare


class TestTheExperimentFile(unittest.TestCase):
    """Pinned on purpose: changing one of these changes the published equations."""

    def setUp(self) -> None:
        self.study = load_config()

    def test_the_search_is_the_one_the_work_started_with(self) -> None:
        search = self.study.search
        self.assertEqual(
            (
                search.max_abs_zscore,
                search.penalty,
                search.pool_size,
                search.max_terms,
                search.beam_width,
                search.max_arity,
            ),
            (3.5, 0.1, 600, 30, 6, 3),
        )

    def test_the_length_rule_is_the_plateau_rule_with_its_original_constants(self) -> None:
        selection = self.study.selection
        self.assertEqual(
            (selection.method, selection.delta, selection.window, selection.smoothing, selection.capability_arity),
            ("plateau_knee", 0.01, 4, 3, 3),
        )

    def test_the_sweep_grid_is_the_original_one(self) -> None:
        sweep = self.study.sweep
        self.assertEqual(sweep.zscores, (3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0))
        self.assertEqual(sweep.penalties, (0.1, 0.3, 1.0, 3.0, 10.0))
        self.assertEqual((sweep.arities, sweep.readable_terms, sweep.band), ((2, 3), 20, 0.02))

    def test_the_study_fits_the_four_equations_under_the_four_protocols_without_exclusions(self) -> None:
        self.assertEqual(self.study.equations, ("E1", "E2", "E3", "E3-MAX"))
        self.assertEqual(self.study.protocols, ("IS", "LODO", "LOMO", "DHO"))
        self.assertEqual(self.study.grammar.exclude, {})

    def test_the_target_is_mcc_on_its_range_and_unscaled(self) -> None:
        response = self.study.response
        self.assertEqual((response.column, response.bounds, response.transform), ("MCC", (-1.0, 1.0), "none"))
        self.assertEqual((self.study.corpus.dataset_column, self.study.corpus.model_column), ("Dataset", "Model"))

    def test_the_features_are_the_twelve_dataset_and_six_model_columns(self) -> None:
        self.assertEqual((len(self.study.features.dataset), len(self.study.features.model)), (12, 6))


class TestTheExpectedResults(unittest.TestCase):
    """The headline of the start of the work, read from the stored results."""

    def curve(self, variant: str) -> pl.DataFrame:
        return pl.read_csv(EXPECTED / variant / "curve.csv")

    def terms(self, variant: str) -> int:
        return len(json.loads((EXPECTED / variant / "equation.json").read_text())["terms"])

    def test_e3_valid_is_fifteen_terms_and_e3_max_twenty_four(self) -> None:
        self.assertEqual((self.terms("e3"), self.terms("e3-max")), (15, 24))

    def test_their_floors_are_the_ones_the_paper_pipeline_reported(self) -> None:
        valid, maximum = floor_curve(self.curve("e3")), floor_curve(self.curve("e3-max"))
        self.assertAlmostEqual(float(valid[14]), 0.6478020727290206, places=9)
        self.assertAlmostEqual(float(maximum[23]), 0.6696333584794399, places=9)

    def test_e3_max_is_the_raw_floor_maximum_of_its_curve(self) -> None:
        floor = floor_curve(self.curve("e3-max"))
        self.assertEqual(int(np.argmax(floor)) + 1, 24)

    def test_the_stored_scores_cover_every_variant_and_protocol(self) -> None:
        scores = pl.read_csv(EXPECTED / "scores.csv")
        self.assertEqual(set(scores["variant"]), {"E1", "E2", "E3-Valid", "E3-MAX"})
        self.assertEqual(set(scores["protocol"]), {"IS", "LODO", "LOMO", "DHO"})


class TestTheComparison(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.copy = Path(self.directory.name) / "MCC"
        shutil.copytree(EXPECTED, self.copy)

    def test_the_expected_results_agree_with_themselves(self) -> None:
        self.assertEqual(compare(self.copy), [])

    def test_a_weight_moved_in_the_last_digits_passes_and_in_the_first_does_not(self) -> None:
        path = self.copy / "e3" / "equation.json"
        equation = json.loads(path.read_text())
        equation["weights"][0] *= 1 + 1e-9
        path.write_text(json.dumps(equation))
        self.assertEqual(compare(self.copy), [])
        equation["weights"][0] *= 1.01
        path.write_text(json.dumps(equation))
        self.assertEqual(compare(self.copy), ["e3/equation.json: weights differ beyond rtol 1e-06"])

    def test_a_different_term_is_reported(self) -> None:
        path = self.copy / "e1" / "equation.json"
        equation = json.loads(path.read_text())
        equation["terms"][0]["operands"][0]["feature"] = "other"
        path.write_text(json.dumps(equation))
        self.assertIn("e1/equation.json: the terms differ", compare(self.copy))

    def test_a_changed_curve_and_a_changed_score_are_reported(self) -> None:
        curve = pl.read_csv(self.copy / "e2" / "curve.csv")
        curve.with_columns((pl.col("r2_in_sample") + 0.01).alias("r2_in_sample")).write_csv(
            self.copy / "e2" / "curve.csv"
        )
        scores = pl.read_csv(self.copy / "scores.csv")
        scores.with_columns(pl.when(pl.col("variant") == "E1").then(0.0).otherwise(pl.col("r2")).alias("r2")).write_csv(
            self.copy / "scores.csv"
        )
        problems = compare(self.copy)
        self.assertIn("e2/curve.csv: column r2_in_sample differs beyond rtol 1e-06", problems)
        self.assertIn("scores.csv: column r2 differs beyond rtol 1e-06", problems)

    def test_a_curve_with_a_different_shape_is_reported(self) -> None:
        curve = pl.read_csv(self.copy / "e3" / "curve.csv")
        curve.head(10).write_csv(self.copy / "e3" / "curve.csv")
        self.assertTrue(any(problem.startswith("e3/curve.csv: shape") for problem in compare(self.copy)))

    def test_a_changed_label_is_reported(self) -> None:
        path = self.copy / "e3" / "equation.json"
        equation = json.loads(path.read_text())
        equation["target"] = "not MCC"
        path.write_text(json.dumps(equation))
        self.assertTrue(any("target" in problem for problem in compare(self.copy)))

    def test_every_variant_is_compared(self) -> None:
        self.assertEqual(VARIANTS, ("e1", "e2", "e3", "e3-max"))
        shutil.rmtree(self.copy / "e2")
        with self.assertRaises(FileNotFoundError):
            compare(self.copy)

    def test_the_command_exits_nonzero_on_a_difference_and_with_usage_on_no_argument(self) -> None:
        from tests.regression import main

        self.assertEqual(main([str(self.copy)]), 0)
        (self.copy / "e1" / "curve.csv").write_text("n_terms\n1\n")
        self.assertEqual(main([str(self.copy)]), 1)
        self.assertEqual(main([]), 2)


if __name__ == "__main__":
    unittest.main()
