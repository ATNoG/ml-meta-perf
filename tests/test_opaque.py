"""The opaque-regressor comparison.

These pin the *shape* of the result rather than its values. The values move with the
scikit-learn version and with the seed, and pinning them would turn a routine upgrade into a
failing build; the shape is what the study's argument rests on and does not move at all.
"""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.data import DATASET_COLUMN, MODEL_COLUMN, groups, load, target
from ml_meta_perf.opaque import ESTIMATORS, _design, evaluate, opaque_baselines
from ml_meta_perf.validate import leave_one_group_out


class TestOpaqueBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()
        cls.outcome = evaluate(cls.frame)
        cls.rows = {row["model"]: row for row in cls.outcome.table.to_dicts()}

    def test_every_estimator_is_reported(self) -> None:
        self.assertEqual(set(self.rows), {label for label, _ in ESTIMATORS})

    def test_the_forest_fits_far_better_than_it_transfers(self) -> None:
        """The study's argument in two numbers. If this ever stopped holding, the trade the
        whole project is making would need re-arguing rather than the test relaxing."""
        forest = self.rows["RandomForest (300 trees)"]
        self.assertGreater(forest["r2_in_sample"], 0.85)
        self.assertLess(forest["r2_loo_dataset"], 0.4)

    def test_no_opaque_model_transfers_as_well_as_the_equation(self) -> None:
        from ml_meta_perf.experiment import run_e3

        equation = float(run_e3(self.frame).cross_validated["loo_dataset"]["r2"])
        for label, row in self.rows.items():
            with self.subTest(model=label):
                self.assertLess(row["r2_loo_dataset"], equation)

    def test_holding_out_a_model_is_easier_than_holding_out_a_dataset(self) -> None:
        """Dataset features are constant within a dataset and most model features within a
        model, but there are 25 models against 20 datasets and a held-out model's rows are
        spread across every dataset. Every estimator finds the model split the easier one."""
        for label, row in self.rows.items():
            with self.subTest(model=label):
                self.assertGreater(row["r2_loo_model"], row["r2_loo_dataset"])

    def test_predictions_are_kept_for_every_protocol(self) -> None:
        for label, held in self.outcome.predictions.items():
            with self.subTest(model=label):
                self.assertEqual(set(held), {"in_sample", "loo_dataset", "loo_model"})
                for values in held.values():
                    self.assertEqual(values.shape, target(self.frame).shape)

    def test_the_table_is_scored_from_the_kept_predictions(self) -> None:
        """The decision comparisons reuse these predictions, so a table scored from anything
        else could disagree with the ranking and threshold tables built beside it."""
        from ml_meta_perf.stats import r2_score

        truth = target(self.frame)
        for label, held in self.outcome.predictions.items():
            with self.subTest(model=label):
                self.assertAlmostEqual(
                    self.rows[label]["r2_loo_dataset"], r2_score(truth, held["loo_dataset"]), places=12
                )

    def test_out_of_fold_predictions_stay_inside_the_training_range(self) -> None:
        """The same clip every reported number gets. Without it the ridge control is scored on
        an unbounded extrapolation the equation is never scored on."""
        truth = target(self.frame)
        labels = groups(self.frame, DATASET_COLUMN)
        for label, held in self.outcome.predictions.items():
            for _, train, test in leave_one_group_out(labels):
                with self.subTest(model=label):
                    self.assertGreaterEqual(held["loo_dataset"][test].min(), float(truth[train].min()) - 1e-12)
                    self.assertLessEqual(held["loo_dataset"][test].max(), float(truth[train].max()) + 1e-12)

    def test_opaque_baselines_is_the_table_of_a_run(self) -> None:
        self.assertEqual(opaque_baselines(self.frame).columns, self.outcome.table.columns)

    def test_is_deterministic(self) -> None:
        repeated = evaluate(self.frame).table
        for column in ("r2_in_sample", "r2_loo_dataset", "r2_loo_model"):
            np.testing.assert_allclose(repeated[column].to_numpy(), self.outcome.table[column].to_numpy())


class TestDesign(unittest.TestCase):
    def test_uses_the_raw_columns_and_nothing_else(self) -> None:
        """Handing the opaque side the grammar's transforms would compare the equation with
        itself. The comparison a reader means is against the columns."""
        from ml_meta_perf.data import ALL_FEATURES

        frame = load()
        design = _design(frame)
        self.assertEqual(design.shape, (frame.height, len(ALL_FEATURES)))
        for index, name in enumerate(ALL_FEATURES):
            np.testing.assert_allclose(design[:, index], frame[name].to_numpy().astype(float))


class TestOpaqueEntersTheComparisons(unittest.TestCase):
    """The opaque rows have to reach the ranking and decision tables, each naming its protocol.

    A comparison is only a comparison if every side names the protocol it was scored under,
    and this project has twice shipped one that did not.
    """

    @classmethod
    def setUpClass(cls) -> None:
        # Both the forest folds and the beam search are seconds each, so they are paid once
        # for the class rather than once per assertion.
        from ml_meta_perf.experiment import DEFAULT_E3, decision_baselines, ranking_baselines, run_e3

        frame = load()
        equation, outcome = run_e3(frame), evaluate(frame)
        cls.ranking = ranking_baselines(frame, equation, DEFAULT_E3, outcome)
        cls.decision = decision_baselines(
            frame, DEFAULT_E3, equation.paths.get("loo_dataset"), equation, outcome
        )

    def test_ranking_baselines_carry_the_opaque_rows(self) -> None:
        for label, _ in ESTIMATORS:
            self.assertIn(f"{label} (loo-dataset)", self.ranking["predictor"].to_list())

    def test_decision_baselines_carry_the_opaque_rows(self) -> None:
        for label, _ in ESTIMATORS:
            self.assertIn(f"{label} (loo-dataset)", self.decision["predictor"].to_list())

    def test_every_predictor_names_its_protocol(self) -> None:
        for table in (self.ranking, self.decision):
            for predictor in table["predictor"].to_list():
                with self.subTest(predictor=predictor):
                    self.assertRegex(predictor, r"\((in-sample|loo-dataset|loo-model|loo-cell)")

    def test_the_equation_beats_every_opaque_model_on_the_decision(self) -> None:
        """The finding this comparison added: on the go/no-go decision the opaque models lose
        to the equation *and* to the trivial per-model centres. Pinned at the 0.7 threshold
        the chapters report, on MCC, which is the metric the decision is scored by."""
        at_threshold = {
            row["predictor"]: float(row["mcc"])
            for row in self.decision.to_dicts()
            if abs(float(row["threshold"]) - 0.7) < 1e-9
        }
        strictest = at_threshold["equation (loo-cell: both held out)"]
        for label, _ in ESTIMATORS:
            with self.subTest(model=label):
                self.assertGreater(strictest, at_threshold[f"{label} (loo-dataset)"])


class TestGroupsAreDistinct(unittest.TestCase):
    def test_the_corpus_has_more_models_than_datasets(self) -> None:
        """The premise of `test_holding_out_a_model_is_easier_than_holding_out_a_dataset`."""
        frame = load()
        self.assertGreater(
            pl.Series(groups(frame, MODEL_COLUMN)).n_unique(), pl.Series(groups(frame, DATASET_COLUMN)).n_unique()
        )


if __name__ == "__main__":
    unittest.main()
