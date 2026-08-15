"""Statistics are hand-rolled to avoid a scipy dependency, so they get checked hard."""

import unittest

import numpy as np

from metafit.stats import mae, pearson, r2_score, rankdata, rmse, spearman


class TestRankdata(unittest.TestCase):
    def test_simple_ranks_start_at_one(self) -> None:
        np.testing.assert_allclose(rankdata(np.array([3.0, 1.0, 2.0])), [3.0, 1.0, 2.0])

    def test_ties_receive_average_rank(self) -> None:
        # Ranks 2 and 3 are shared, so both become 2.5. This is the case that matters:
        # 17% of the meta-dataset sits at MCC exactly 1.0.
        np.testing.assert_allclose(rankdata(np.array([1.0, 5.0, 5.0, 9.0])), [1.0, 2.5, 2.5, 4.0])

    def test_all_tied(self) -> None:
        np.testing.assert_allclose(rankdata(np.array([2.0, 2.0, 2.0])), [2.0, 2.0, 2.0])

    def test_rank_sum_is_preserved_under_ties(self) -> None:
        values = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 7.0])
        self.assertAlmostEqual(float(rankdata(values).sum()), 6 * 7 / 2)


class TestCorrelation(unittest.TestCase):
    def test_perfect_linear(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(pearson(x, 2.0 * x + 5.0), 1.0)
        self.assertAlmostEqual(pearson(x, -x), -1.0)

    def test_constant_input_is_zero_not_nan(self) -> None:
        self.assertEqual(pearson(np.ones(5), np.arange(5.0)), 0.0)
        self.assertEqual(spearman(np.ones(5), np.arange(5.0)), 0.0)

    def test_spearman_sees_monotone_nonlinearity_pearson_misses(self) -> None:
        # The premise of the guided screening: a monotone but strongly curved relation
        # scores near 1 in rank correlation while linear correlation lags well behind.
        x = np.linspace(1.0, 100.0, 50)
        y = np.log(x)
        self.assertAlmostEqual(spearman(x, y), 1.0, places=6)
        self.assertLess(pearson(x, y), 0.92)

    def test_spearman_of_short_input(self) -> None:
        self.assertEqual(spearman(np.array([1.0]), np.array([2.0])), 0.0)


class TestErrorMetrics(unittest.TestCase):
    def test_r2_of_perfect_prediction(self) -> None:
        y = np.array([0.1, 0.5, 0.9])
        self.assertAlmostEqual(r2_score(y, y.copy()), 1.0)

    def test_r2_of_mean_prediction_is_zero(self) -> None:
        y = np.array([0.1, 0.5, 0.9])
        self.assertAlmostEqual(r2_score(y, np.full_like(y, y.mean())), 0.0)

    def test_r2_can_be_negative(self) -> None:
        y = np.array([0.0, 1.0])
        self.assertLess(r2_score(y, np.array([1.0, 0.0])), 0.0)

    def test_r2_of_constant_truth_is_zero(self) -> None:
        self.assertEqual(r2_score(np.ones(4), np.zeros(4)), 0.0)

    def test_mae_and_rmse(self) -> None:
        truth = np.array([0.0, 0.0, 0.0, 0.0])
        prediction = np.array([1.0, -1.0, 1.0, -1.0])
        self.assertAlmostEqual(mae(truth, prediction), 1.0)
        self.assertAlmostEqual(rmse(truth, prediction), 1.0)

    def test_rmse_penalises_outliers_more_than_mae(self) -> None:
        truth = np.zeros(4)
        prediction = np.array([0.0, 0.0, 0.0, 4.0])
        self.assertGreater(rmse(truth, prediction), mae(truth, prediction))


if __name__ == "__main__":
    unittest.main()
