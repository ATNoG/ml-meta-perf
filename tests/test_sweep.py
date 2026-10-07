"""Tests for the configuration sweep: its grid, its per-point evaluation and its choice."""

import dataclasses
import unittest

import polars as pl

from ml_meta_perf.config import load_config
from ml_meta_perf.sweep import choose, evaluate, grid
from tests import corpus


def candidates(rows: list[tuple[int, float, float, int, float]]) -> pl.DataFrame:
    """(n_terms, smoothed_floor, dho_ap, max_arity, penalty) per configuration."""
    return pl.DataFrame(
        [
            {"n_terms": n, "smoothed_floor": floor, "dho_ap": ap, "max_arity": arity, "penalty": penalty}
            for n, floor, ap, arity, penalty in rows
        ]
    )


class TestChoice(unittest.TestCase):
    """Within the readable R2 band, the best DHO ranking wins."""

    def setUp(self) -> None:
        self.sweep = dataclasses.replace(load_config().sweep, readable_terms=20, band=0.02)

    def test_the_best_ranking_inside_the_band_wins_over_the_best_r2(self) -> None:
        table = candidates([(19, 0.624, 0.70, 3, 0.3), (17, 0.612, 0.85, 2, 0.3), (12, 0.598, 0.90, 2, 3.0)])
        self.assertEqual(choose(table, self.sweep)["n_terms"], 17)

    def test_an_unreadable_equation_can_neither_win_nor_set_the_band(self) -> None:
        table = candidates([(27, 0.655, 0.95, 3, 0.3), (17, 0.612, 0.85, 2, 0.3), (15, 0.600, 0.80, 2, 1.0)])
        self.assertEqual(choose(table, self.sweep)["n_terms"], 17)

    def test_ties_go_to_fewer_terms_then_lower_arity_then_heavier_penalty(self) -> None:
        table = candidates([(17, 0.61, 0.85, 2, 0.1), (17, 0.61, 0.85, 2, 0.3), (17, 0.61, 0.85, 3, 1.0)])
        chosen = choose(table, self.sweep)
        self.assertEqual((chosen["max_arity"], chosen["penalty"]), (2, 0.3))

    def test_it_refuses_when_nothing_is_readable(self) -> None:
        with self.assertRaises(ValueError):
            choose(candidates([(27, 0.65, 0.9, 3, 0.3)]), self.sweep)


class TestGridAndEvaluation(unittest.TestCase):
    def test_the_grid_is_the_product_over_the_search_base(self) -> None:
        study = load_config()
        points = grid(study)
        sweep = study.sweep
        self.assertEqual(len(points), len(sweep.zscores) * len(sweep.penalties) * len(sweep.arities))
        self.assertTrue(all(point.pool_size == study.search.pool_size for point in points))

    def test_a_point_is_scored_at_the_length_the_study_rule_chooses(self) -> None:
        row, curve = evaluate(corpus.E3, corpus.SELECTION, str(corpus.sample_path()))
        self.assertIn(row["reached_by"], {"knee", "maximum"})
        self.assertIn(row["n_terms"], curve["n_terms"].to_list())
        self.assertTrue(0.0 <= float(row["dho_ap"]) <= 1.0)
        self.assertEqual(curve["max_abs_zscore"].unique().to_list(), [corpus.E3.max_abs_zscore])


if __name__ == "__main__":
    unittest.main()
