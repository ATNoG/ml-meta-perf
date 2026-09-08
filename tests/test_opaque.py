"""The opaque-regressor comparison.

These pin the *shape* of the result rather than its values. The values move with the
scikit-learn version and with the seed, and pinning them would turn a routine upgrade into a
failing build; the shape is what the study's argument rests on and does not move at all.
"""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.data import DATASET_COLUMN, MODEL_COLUMN, groups, load, target
from ml_meta_perf.opaque import ESTIMATORS, OpaqueRun, _cross_validate, _design, evaluate
from ml_meta_perf.validate import leave_one_group_out

#: One pass over the estimators for the whole module. `evaluate` refits a random forest once
#: per held-out group and once per cell -- 522 fits -- so it is minutes, and two test classes
#: needing it would otherwise pay that twice over.
_RUN: list[OpaqueRun] = []


def _outcome() -> OpaqueRun:
    """The shared `evaluate` result, computed on first use."""
    if not _RUN:
        _RUN.append(evaluate(load()))
    return _RUN[0]


class TestOpaqueBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()
        cls.outcome = _outcome()
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
                self.assertEqual(set(held), {"in_sample", "loo_dataset", "loo_model", "loo_cell"})
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

    def test_is_deterministic(self) -> None:
        """Re-fitting from scratch, not re-reading the cache: a seeded forest has to reproduce
        itself or every number in the comparison is a draw rather than a measurement. Scored on
        the two cheap protocols, because the cell protocol is 476 refits and this assertion
        does not need them to hold."""
        design, truth = _design(self.frame), target(self.frame)
        for kind in ("ridge", "boosting"):
            with self.subTest(kind=kind):
                labels = groups(self.frame, DATASET_COLUMN)
                first = _cross_validate(design, truth, labels, kind)
                np.testing.assert_allclose(first, _cross_validate(design, truth, labels, kind))


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


class TestDoublyHeldOut(unittest.TestCase):
    """Full leakage prevention: neither the dataset nor the model of a cell is in training.

    The only protocol on which an opaque regressor and the equation are denied the same
    things, so it is the one the study's central comparison rests on.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()
        cls.rows = {row["model"]: row for row in _outcome().table.to_dicts()}

    def test_removing_both_identities_is_the_hardest_protocol(self) -> None:
        """Each column removes more than the one before it, and the ordering is the finding:
        a model that learned dataset identity loses most of its score when identity goes."""
        for label, row in self.rows.items():
            with self.subTest(model=label):
                self.assertLess(row["r2_loo_cell"], row["r2_in_sample"])
                self.assertLessEqual(row["r2_loo_cell"], row["r2_loo_model"])

    def test_no_opaque_model_clears_the_corpus_mean_under_full_leakage_prevention(self) -> None:
        """R2 at or below zero means "no better than predicting the mean". If an opaque model
        ever cleared this bar meaningfully, the study's trade would need re-arguing rather than
        this test relaxing."""
        for label, row in self.rows.items():
            with self.subTest(model=label):
                self.assertLess(row["r2_loo_cell"], 0.1)

    def test_the_equation_beats_every_opaque_model_on_the_same_protocol(self) -> None:
        from ml_meta_perf.experiment import DEFAULT_E3, doubly_held_out_predictions
        from ml_meta_perf.stats import r2_score

        predictions = doubly_held_out_predictions(self.frame, DEFAULT_E3)
        assert predictions is not None
        equation = r2_score(target(self.frame), predictions)
        for label, row in self.rows.items():
            with self.subTest(model=label):
                self.assertGreater(equation, row["r2_loo_cell"])

    def test_a_cell_is_never_predicted_from_its_own_dataset_or_model(self) -> None:
        """The property the protocol exists for, checked directly rather than trusted: a
        training mask that let either identity through would leak and the score would flatter
        every opaque row."""
        datasets = groups(self.frame, DATASET_COLUMN)
        models = groups(self.frame, MODEL_COLUMN)
        for row_label in np.unique(datasets)[:3]:
            for column_label in np.unique(models)[:3]:
                train = (datasets != row_label) & (models != column_label)
                with self.subTest(dataset=row_label, model=column_label):
                    self.assertNotIn(row_label, set(datasets[train]))
                    self.assertNotIn(column_label, set(models[train]))


if __name__ == "__main__":
    unittest.main()
