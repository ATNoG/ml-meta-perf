"""Rules that choose the arity and the number of terms, on tables whose answer is known."""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.lengths import (
    MEASURED_RULES,
    Pick,
    candidate_table,
    floor_argmax,
    pareto_elbow,
    pareto_front,
    pareto_knee,
    pareto_plateau,
    plateau_knee,
)
from ml_meta_perf.selection import plateau_knee as curve_plateau_knee

PARAMS = {"delta": 0.01, "window": 2, "smoothing": 1, "fixed_arity": 1}


def curve(floors: list[float]) -> pl.DataFrame:
    """A curve whose only protocol column is the floor itself."""
    return pl.DataFrame({"n_terms": list(range(1, len(floors) + 1)), "r2_in_sample": floors})


def table(**by_arity: list[float]) -> pl.DataFrame:
    return candidate_table({int(name.removeprefix("a")): curve(floors) for name, floors in by_arity.items()}, 1)


SATURATING = [0.30, 0.55, 0.75, 0.90, 0.903, 0.905, 0.905, 0.905, 0.905, 0.905]


class TestCandidateTable(unittest.TestCase):
    def test_it_has_one_row_per_arity_and_length_with_complexity_arity_times_terms(self) -> None:
        frame = table(a1=[0.1, 0.2, 0.3], a2=[0.2, 0.4, 0.5, 0.6])
        self.assertEqual(frame.height, 7)
        self.assertTrue((frame["complexity"] == frame["arity"] * frame["n_terms"]).all())

    def test_smoothing_runs_along_the_length_within_an_arity(self) -> None:
        spiky = candidate_table({1: curve([0.5, 0.9, 0.5, 0.5, 0.5]), 2: curve([0.1] * 5)}, 3)
        self.assertEqual(spiky.filter(pl.col("arity") == 1)["smooth"].to_list(), [0.7, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual(spiky.filter(pl.col("arity") == 2)["smooth"].to_list(), [0.1] * 5)

    def test_an_empty_table_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            candidate_table({}, 3)


class TestParetoFront(unittest.TestCase):
    def test_the_front_improves_strictly_with_complexity(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(20):
            frame = table(a1=list(rng.uniform(0, 1, 8)), a2=list(rng.uniform(0, 1, 8)), a3=list(rng.uniform(0, 1, 8)))
            front = pareto_front(frame)
            self.assertTrue(np.all(np.diff(front["complexity"].to_numpy()) > 0))
            self.assertTrue(np.all(np.diff(front["smooth"].to_numpy()) > 0))

    def test_every_candidate_is_matched_by_a_front_point_no_dearer(self) -> None:
        rng = np.random.default_rng(2)
        frame = table(a1=list(rng.uniform(0, 1, 8)), a2=list(rng.uniform(0, 1, 8)))
        front = pareto_front(frame)
        for row in frame.iter_rows(named=True):
            dominating = front.filter((pl.col("complexity") <= row["complexity"]) & (pl.col("smooth") >= row["smooth"]))
            self.assertGreater(dominating.height, 0)

    def test_equal_complexity_keeps_the_better_floor_then_the_fewer_terms(self) -> None:
        # (arity 1, 4 terms) and (arity 2, 2 terms) both cost 4.
        frame = table(a1=[0.1, 0.2, 0.3, 0.4], a2=[0.3, 0.6])
        front = pareto_front(frame)
        self.assertEqual(front.filter(pl.col("complexity") == 4).row(0, named=True)["arity"], 2)
        tied = table(a1=[0.1, 0.2, 0.3, 0.6], a2=[0.3, 0.6])
        self.assertEqual(pareto_front(tied).filter(pl.col("complexity") == 4).row(0, named=True)["n_terms"], 2)


class TestRulesOnKnownCurves(unittest.TestCase):
    def test_the_reference_takes_the_best_floor_and_the_cheaper_of_equals(self) -> None:
        self.assertEqual(floor_argmax(table(a1=SATURATING)), Pick(1, 6))
        self.assertEqual(floor_argmax(table(a1=[0.2, 0.9], a2=[0.1, 0.9])), Pick(1, 2))

    def test_a_saturating_curve_is_cut_where_it_stops_gaining(self) -> None:
        frame = table(a1=SATURATING)
        self.assertEqual(pareto_plateau(frame, **PARAMS), Pick(1, 4))
        self.assertEqual(pareto_knee(frame), Pick(1, 4))
        self.assertEqual(pareto_elbow(frame), Pick(1, 4))

    def test_the_pareto_rules_prefer_the_grammar_that_buys_the_floor_cheaper(self) -> None:
        # Arity 2 reaches 0.95 with two terms (complexity 4); arity 1 needs four to reach 0.90.
        frame = table(a1=SATURATING, a2=[0.60, 0.95, 0.951, 0.952, 0.952, 0.952])
        for rule in (pareto_knee, pareto_plateau, pareto_elbow):
            with self.subTest(rule=rule.__name__):
                self.assertEqual(rule(frame, **PARAMS), Pick(2, 2))

    def test_a_curve_that_never_stops_improving_ends_at_the_end_of_the_front(self) -> None:
        rising = table(a1=[0.1 * n for n in range(1, 9)])
        self.assertEqual(pareto_plateau(rising, **PARAMS), Pick(1, 8))

    def test_a_flat_curve_costs_one_term(self) -> None:
        flat = table(a1=[0.5] * 6)
        for rule in (pareto_knee, pareto_plateau, pareto_elbow):
            self.assertEqual(rule(flat, **PARAMS), Pick(1, 1))


class TestRuleContracts(unittest.TestCase):
    def tables(self) -> list[pl.DataFrame]:
        rng = np.random.default_rng(3)
        return [
            table(a1=list(rng.uniform(0, 1, n)), a2=list(rng.uniform(0, 1, n)), a3=list(rng.uniform(0, 1, n)))
            for n in (2, 3, 5, 9, 14)
        ]

    def test_every_rule_returns_a_candidate_of_the_table(self) -> None:
        for frame in self.tables():
            members = {(r["arity"], r["n_terms"]) for r in frame.iter_rows(named=True)}
            for name, rule in MEASURED_RULES.items():
                with self.subTest(rule=name, rows=frame.height):
                    pick = rule(frame, **PARAMS)
                    self.assertIn((pick.arity, pick.n_terms), members)

    def test_no_rule_spends_more_than_the_reference(self) -> None:
        for frame in self.tables():
            ceiling = floor_argmax(frame).complexity
            for name in ("pareto_knee", "pareto_plateau", "pareto_elbow"):
                with self.subTest(rule=name):
                    self.assertLessEqual(MEASURED_RULES[name](frame, **PARAMS).complexity, ceiling)

    def test_the_plateau_rule_at_one_arity_is_the_study_rule(self) -> None:
        rng = np.random.default_rng(4)
        for _ in range(15):
            floors = list(np.sort(rng.uniform(0, 1, 12)) + rng.normal(0, 0.03, 12))
            frame = candidate_table({3: curve(floors)}, 3)
            expected = curve_plateau_knee(curve(floors), delta=0.01, window=4, smoothing=3)
            picked = plateau_knee(frame, delta=0.01, window=4, smoothing=3, fixed_arity=3)
            self.assertEqual(picked, Pick(3, expected))

    def test_the_fixed_arity_must_exist(self) -> None:
        with self.assertRaisesRegex(ValueError, "arity 4"):
            plateau_knee(table(a1=SATURATING), **{**PARAMS, "fixed_arity": 4})

    def test_a_scaled_floor_scale_does_not_move_the_pareto_choice(self) -> None:
        # R2 is a fraction; rescaling the whole floor axis must not change where the front bends.
        base = table(a1=SATURATING, a2=[0.60, 0.95, 0.951, 0.952, 0.952, 0.952])
        scaled = base.with_columns(
            (pl.col("smooth") * 0.5 + 0.1).alias("smooth"), (pl.col("floor") * 0.5 + 0.1).alias("floor")
        )
        for rule in (pareto_knee, pareto_elbow):
            self.assertEqual(rule(base, **PARAMS), rule(scaled, **PARAMS))


if __name__ == "__main__":
    unittest.main()
