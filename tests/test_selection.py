"""Knee detection and the Pareto front over equation length."""

import unittest

import numpy as np
import polars as pl

from metafit.selection import (
    DETECTORS,
    knee_index,
    knee_terms,
    pareto_front,
    pareto_table,
    recommend,
    simplify_curve,
)


def curve(
    sizes: list[int],
    in_sample: list[float],
    loo: list[float] | None = None,
) -> pl.DataFrame:
    data: dict[str, list[float] | list[int]] = {"n_terms": sizes, "r2_in_sample": in_sample}
    if loo is not None:
        data["r2_loo_dataset"] = loo
    return pl.DataFrame(data)


class TestKneeIndex(unittest.TestCase):
    def test_finds_the_bend_of_a_saturating_curve(self) -> None:
        sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        scores = np.array([0.1, 0.4, 0.6, 0.68, 0.70, 0.71, 0.715, 0.717])
        index = knee_index(sizes, scores)
        # The bend is somewhere in the early-middle, not at either extreme.
        self.assertGreater(index, 0)
        self.assertLess(index, len(sizes) - 1)

    def test_short_curves_fall_back_to_the_last_point(self) -> None:
        self.assertEqual(knee_index(np.array([1.0, 2.0]), np.array([0.1, 0.5])), 1)

    def test_single_point(self) -> None:
        self.assertEqual(knee_index(np.array([1.0]), np.array([0.5])), 0)

    def test_returns_a_valid_index(self) -> None:
        sizes = np.arange(2.0, 20.0, 2.0)
        scores = 1.0 - np.exp(-sizes / 4.0)
        index = knee_index(sizes, scores)
        self.assertTrue(0 <= index < len(sizes))


class TestDetectorsAndSmoothing(unittest.TestCase):
    def setUp(self) -> None:
        self.sizes = np.arange(1.0, 33.0)
        self.saturating = 1.0 - np.exp(-self.sizes / 5.0)
        rng = np.random.default_rng(5)
        self.noisy = self.saturating + rng.normal(0.0, 0.02, self.sizes.shape)

    def test_every_detector_returns_a_valid_index(self) -> None:
        for detector in DETECTORS:
            index = knee_index(self.sizes, self.saturating, detector=detector)
            self.assertTrue(0 <= index < self.sizes.shape[0], detector)

    def test_an_unknown_detector_falls_back_to_the_default(self) -> None:
        index = knee_index(self.sizes, self.saturating, detector="nonsense")
        self.assertEqual(index, knee_index(self.sizes, self.saturating, detector="autoelbow"))

    def test_simplification_keeps_the_endpoints(self) -> None:
        kept = simplify_curve(self.sizes, self.saturating, 0.01)
        self.assertIn(0, kept.tolist())
        self.assertIn(self.sizes.shape[0] - 1, kept.tolist())

    def test_simplification_reduces_the_point_count(self) -> None:
        kept = simplify_curve(self.sizes, self.noisy, 0.05)
        self.assertLess(kept.shape[0], self.sizes.shape[0])

    def test_smoothed_knee_is_still_an_index_into_the_original(self) -> None:
        index = knee_index(self.sizes, self.noisy, tolerance=0.01)
        self.assertTrue(0 <= index < self.sizes.shape[0])

    def test_smoothing_a_short_curve_falls_back(self) -> None:
        short = np.array([1.0, 2.0, 3.0])
        self.assertTrue(0 <= knee_index(short, np.array([0.1, 0.5, 0.6]), tolerance=0.5) < 3)


class TestKneeTerms(unittest.TestCase):
    def test_maps_the_index_back_to_a_term_count(self) -> None:
        table = curve([2, 4, 6, 8, 10, 12], [0.2, 0.45, 0.55, 0.58, 0.585, 0.587])
        chosen = knee_terms(table)
        self.assertIn(chosen, [2, 4, 6, 8, 10, 12])

    def test_reads_the_requested_column(self) -> None:
        table = curve([2, 4, 6, 8], [0.2, 0.5, 0.55, 0.56], [0.1, 0.3, 0.32, 0.33])
        self.assertIn(knee_terms(table, "r2_loo_dataset"), [2, 4, 6, 8])


class TestParetoFront(unittest.TestCase):
    def test_keeps_only_lengths_nothing_shorter_beats(self) -> None:
        table = curve([2, 4, 6, 8], [0.0] * 4, [0.10, 0.30, 0.20, 0.40])
        # 6 scores 0.20, worse than 4's 0.30, so it is dominated by a shorter equation.
        self.assertEqual(pareto_front(table)["n_terms"].to_list(), [2, 4, 8])

    def test_a_monotone_curve_is_entirely_on_the_front(self) -> None:
        table = curve([2, 4, 6], [0.0] * 3, [0.1, 0.2, 0.3])
        self.assertEqual(pareto_front(table)["n_terms"].to_list(), [2, 4, 6])

    def test_a_decreasing_curve_keeps_only_the_shortest(self) -> None:
        table = curve([2, 4, 6], [0.0] * 3, [0.3, 0.2, 0.1])
        self.assertEqual(pareto_front(table)["n_terms"].to_list(), [2])

    def test_front_is_never_empty(self) -> None:
        table = curve([2, 4], [0.0, 0.0], [0.5, 0.5])
        self.assertGreater(pareto_front(table).height, 0)


class TestParetoTable(unittest.TestCase):
    def test_flags_both_fronts(self) -> None:
        table = pareto_table(curve([2, 4, 6], [0.2, 0.5, 0.6], [0.1, 0.3, 0.2]))
        self.assertEqual(table["front_in_sample"].to_list(), [True, True, True])
        self.assertEqual(table["front_loo_dataset"].to_list(), [True, True, False])

    def test_in_sample_only_curve_gets_one_front(self) -> None:
        table = pareto_table(curve([2, 4], [0.2, 0.5]))
        self.assertIn("front_in_sample", table.columns)
        self.assertNotIn("front_loo_dataset", table.columns)

    def test_one_row_per_length(self) -> None:
        table = pareto_table(curve([2, 4, 6, 8], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(table["n_terms"].to_list(), [2, 4, 6, 8])


class TestRecommend(unittest.TestCase):
    def setUp(self) -> None:
        self.table = curve(
            [2, 4, 6, 8, 10, 12, 14],
            [0.20, 0.45, 0.52, 0.55, 0.560, 0.564, 0.566],
            [0.05, 0.22, 0.26, 0.31, 0.330, 0.372, 0.441],
        )

    def test_reports_every_rule(self) -> None:
        rules = recommend(self.table)["rule"].to_list()
        self.assertIn("knee (in-sample)", rules)
        self.assertIn("knee (loo-dataset)", rules)
        self.assertIn("best loo-dataset", rules)

    def test_best_rule_picks_the_maximum(self) -> None:
        rows = {row["rule"]: row for row in recommend(self.table).iter_rows(named=True)}
        self.assertEqual(rows["best loo-dataset"]["n_terms"], 14)
        self.assertAlmostEqual(rows["best loo-dataset"]["r2_loo_dataset"], 0.441)

    def test_every_recommendation_is_a_length_on_the_curve(self) -> None:
        available = set(self.table["n_terms"].to_list())
        for value in recommend(self.table)["n_terms"].to_list():
            self.assertIn(value, available)

    def test_works_without_a_cross_validated_column(self) -> None:
        plain = curve([2, 4, 6, 8], [0.2, 0.45, 0.52, 0.55])
        table = recommend(plain)
        self.assertEqual(table["rule"].to_list(), ["knee (in-sample)"])


if __name__ == "__main__":
    unittest.main()
