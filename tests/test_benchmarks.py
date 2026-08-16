"""Reference models. scikit-learn is optional, so these skip cleanly without it."""

import unittest

from metafit.benchmarks import available, reference_models
from metafit.data import load

REQUIRES_SKLEARN = unittest.skipUnless(available(), "scikit-learn is not installed")


class TestAvailability(unittest.TestCase):
    def test_available_returns_a_bool(self) -> None:
        self.assertIsInstance(available(), bool)

    def test_table_has_a_schema_even_when_empty(self) -> None:
        # Without scikit-learn the table is empty but still typed, so callers can select
        # columns from it without branching.
        table = reference_models(load())
        for column in ("model", "r2_in_sample", "r2_loo_dataset", "r2_loo_model"):
            self.assertIn(column, table.columns)


@REQUIRES_SKLEARN
class TestReferenceModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Two estimators, not three: the full benchmark takes minutes and this suite
        # runs on every commit. RidgeCV and RandomForest carry both claims under test.
        cls.table = reference_models(load(), only=("RidgeCV (linear)", "RandomForest"), trees=25)

    def test_reports_the_requested_estimators(self) -> None:
        self.assertEqual(self.table.height, 2)
        self.assertIn("RandomForest", self.table["model"].to_list())
        self.assertNotIn("GradientBoosting", self.table["model"].to_list())

    def test_cross_validated_never_beats_in_sample_for_the_ensembles(self) -> None:
        for row in self.table.iter_rows(named=True):
            if row["model"] == "RidgeCV (linear)":
                continue
            self.assertLess(row["r2_loo_dataset"], row["r2_in_sample"])

    def test_trees_overfit_dataset_identity(self) -> None:
        # The point of the comparison: a forest memorises 476 rows almost perfectly and
        # then transfers to an unseen dataset far worse than a 12-term equation does.
        rows = {row["model"]: row for row in self.table.iter_rows(named=True)}
        forest = rows["RandomForest"]
        self.assertGreater(forest["r2_in_sample"], 0.8)
        self.assertLess(forest["r2_loo_dataset"], 0.2)

    def test_scores_are_finite(self) -> None:
        for row in self.table.iter_rows(named=True):
            for column in ("r2_in_sample", "r2_loo_dataset", "r2_loo_model"):
                self.assertEqual(row[column], row[column])  # not NaN


if __name__ == "__main__":
    unittest.main()
