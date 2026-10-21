"""E3-Valid and E3-MAX: the two variants of the all-features equation, and the rule that picks E3-Valid."""

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml_meta_perf import cli
from ml_meta_perf.config import LENGTH_METHODS, ConfigError, SelectionConfig, default_configuration, load_config
from ml_meta_perf.experiment import fit_path, run_capability, run_equation
from ml_meta_perf.lengths import LENGTH_RULES, candidate_table
from ml_meta_perf.selection import floor_argmax
from tests.synthetic import DATASET_FEATURES, MODEL_FEATURES, planted

CONFIG = dataclasses.replace(default_configuration(), max_terms=6, pool_size=80, beam_width=2, max_arity=2)
SELECTION = dataclasses.replace(load_config().selection, method="pareto_knee", capability_arity=3, window=2)


class TestSelectionMethod(unittest.TestCase):
    def test_the_rules_the_file_may_name_are_the_rules_that_exist(self) -> None:
        self.assertEqual(set(LENGTH_METHODS), set(LENGTH_RULES))

    def test_an_unknown_method_is_named(self) -> None:
        with self.assertRaisesRegex(ConfigError, "selection.method must be one of"):
            dataclasses.replace(SELECTION, method="argmax")
        with self.assertRaises(ConfigError):
            SelectionConfig(delta=0.01, window=4, smoothing=3, method="", capability_arity=3)

    def test_the_mcc_study_keeps_the_rule_of_its_golden(self) -> None:
        self.assertEqual(load_config().selection.method, "plateau_knee")

    def test_the_flexfl_studies_use_the_knee_of_the_front_and_ask_for_both_variants(self) -> None:
        for path in sorted((Path(__file__).resolve().parents[1] / "config" / "flexfl").glob("*.json")):
            with self.subTest(path.name):
                study = load_config(path)
                self.assertEqual((study.selection.method, study.equations), ("pareto_knee", ("E3", "E3-MAX")))


class TestRunEquation(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = planted(3, noise=0.1, seed=2)
        self.schema = self.corpus.schema()

    def run_with(self, method: str):
        selection = dataclasses.replace(SELECTION, method=method)
        return run_equation(
            self.corpus.frame, DATASET_FEATURES, MODEL_FEATURES, CONFIG, selection, "E3", schema=self.schema
        )

    def test_the_plateau_method_fits_only_the_configured_arity_as_before(self) -> None:
        report = self.run_with("plateau_knee")
        self.assertEqual(report.arity, CONFIG.max_arity)

    def test_a_pareto_method_returns_the_candidate_the_rule_picks_from_the_table_of_every_arity(self) -> None:
        report = self.run_with("pareto_knee")
        curves = {
            arity: fit_path(
                self.corpus.frame,
                DATASET_FEATURES,
                MODEL_FEATURES,
                dataclasses.replace(CONFIG, max_arity=arity),
                "E3",
                schema=self.schema,
            ).curve
            for arity in (1, 2)
        }
        pick = LENGTH_RULES["pareto_knee"](
            candidate_table(curves, SELECTION.smoothing),
            delta=SELECTION.delta,
            window=SELECTION.window,
            smoothing=SELECTION.smoothing,
            fixed_arity=CONFIG.max_arity,
        )
        self.assertEqual((report.arity, report.n_terms), (pick.arity, pick.n_terms))
        self.assertEqual(len(report.equation.terms), report.n_terms)
        self.assertLessEqual(report.arity, CONFIG.max_arity)

    def test_the_pareto_pick_never_spends_more_than_the_plateau_pick_at_the_top_arity(self) -> None:
        plateau, knee = self.run_with("plateau_knee"), self.run_with("pareto_knee")
        self.assertLessEqual(knee.arity * knee.n_terms, plateau.arity * plateau.n_terms)


class TestMaximum(unittest.TestCase):
    def test_e3_max_is_the_raw_floor_maximum_at_the_capability_arity(self) -> None:
        corpus = planted(3, noise=0.1, seed=3)
        schema = corpus.schema()
        report = run_capability(corpus.frame, CONFIG, SELECTION, schema=schema)
        fitted = fit_path(
            corpus.frame,
            DATASET_FEATURES,
            MODEL_FEATURES,
            dataclasses.replace(CONFIG, max_arity=SELECTION.capability_arity),
            "E3-MAX",
            schema=schema,
        )
        self.assertEqual((report.arity, report.n_terms), (SELECTION.capability_arity, floor_argmax(fitted.curve)))

    def test_e3_max_scores_at_least_as_well_as_any_length_of_its_own_curve(self) -> None:
        corpus = planted(3, noise=0.1, seed=3)
        report = run_capability(corpus.frame, CONFIG, SELECTION, schema=corpus.schema())
        floor = np.min(
            np.column_stack(
                [report.curve[c].to_numpy() for c in ("r2_in_sample", "r2_loo_dataset", "r2_loo_model", "r2_loo_cell")]
            ),
            axis=1,
        )
        self.assertAlmostEqual(float(floor[report.n_terms - 1]), float(floor.max()))


class TestCommandLine(unittest.TestCase):
    def test_a_study_of_one_equation_writes_e3_max_beside_e3_valid(self) -> None:
        corpus = planted(3, noise=0.1, seed=4)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv = root / "corpus.csv"
            corpus.frame.write_csv(csv)
            study = json.loads(
                Path(__file__).resolve().parents[1].joinpath("config/flexfl/total_time_s.json").read_text()
            )
            study["corpus"] = {"path": str(csv), "dataset_column": "d", "model_column": "m"}
            study["response"] = {"column": "y", "bounds": None, "transform": "none"}
            study["features"] = {"dataset": list(DATASET_FEATURES), "model": list(MODEL_FEATURES)}
            study["search"] |= {"max_terms": 3, "pool_size": 40, "beam_width": 1, "max_arity": 2}
            study["selection"] |= {"capability_arity": 2}
            (root / "study.json").write_text(json.dumps(study))
            code = cli.main(["--config", str(root / "study.json"), "--output", str(root / "out"), "--quiet"])
            self.assertEqual(code, 0)
            folder = root / "out" / "flexfl" / "y"
            for name in ("equation.json", "equation.txt", "curve.csv"):
                self.assertTrue((folder / name).is_file(), name)
                self.assertTrue((folder / "e3-max" / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
