"""The resampling rules, the information criteria and ``auto``: toy tables with known answers."""

import dataclasses
import math
import unittest
from typing import ClassVar
from unittest import mock

import numpy as np
import polars as pl

from ml_meta_perf import lengths
from ml_meta_perf.config import ConfigError, SelectionConfig, default_configuration, load_config
from ml_meta_perf.experiment import candidate_details, fit_path, run_equation
from ml_meta_perf.lengths import (
    LENGTH_RULES,
    MissingColumns,
    Pick,
    aic,
    aicc,
    all_picks,
    auto,
    auto_choice,
    bend_angle,
    bic,
    candidate_table,
    ebic,
    l_method,
    log_slope,
    mcs,
    mcs_members,
    mdl,
    one_se,
    projection,
    stable_within_delta,
)
from ml_meta_perf.terms import Atom, Term, description_length
from tests.synthetic import DATASET_FEATURES, MODEL_FEATURES, planted


def curve(floors: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"n_terms": list(range(1, len(floors) + 1)), "r2_in_sample": floors})


def detail(n_terms: int, **columns: object) -> dict[str, object]:
    return {"n_terms": n_terms, **columns}


def table_with(floors: list[float], details: list[dict[str, object]]) -> pl.DataFrame:
    """One arity; ``details`` is one dict of the extra columns per length."""
    return candidate_table({1: curve(floors)}, 1, {1: pl.DataFrame(details)})


def losses_table(per_candidate: list[list[float]], sst: float = 1.0) -> pl.DataFrame:
    """Candidate n has n terms and the given per-dataset errors; floors follow from the errors."""
    floors = [1.0 - sum(losses) / sst for losses in per_candidate]
    return table_with(floors, [detail(i + 1, loss=losses, sst=sst) for i, losses in enumerate(per_candidate)])


class TestOneSeAndProjection(unittest.TestCase):
    A: ClassVar[list[float]] = [0.01, 0.03, 0.01, 0.03, 0.01, 0.03]  # total 0.12, the best

    def test_the_cheapest_candidate_within_one_standard_error_of_the_best_is_taken(self) -> None:
        cheap_bad = [0.2 / 6] * 6  # total 0.20: outside
        near = [0.13 / 6] * 6  # total 0.13: inside (SE of A is 0.027)
        table = losses_table([cheap_bad, near, self.A])
        self.assertEqual(one_se(table), Pick(1, 2))

    def test_a_larger_multiple_admits_a_cheaper_candidate(self) -> None:
        cheap_bad = [0.2 / 6] * 6
        table = losses_table([cheap_bad, [0.13 / 6] * 6, self.A])
        self.assertEqual(one_se(table, se_multiple=4.0), Pick(1, 1))

    def test_the_best_candidate_is_always_acceptable(self) -> None:
        table = losses_table([[0.5 / 6] * 6, self.A])
        self.assertEqual(one_se(table), Pick(1, 2))

    def test_pairing_rejects_a_constant_offset_that_the_unpaired_rule_forgives(self) -> None:
        worse = [value + 0.003 for value in self.A]  # same pattern, always a little worse
        table = losses_table([worse, self.A])
        self.assertEqual(one_se(table), Pick(1, 1))
        self.assertEqual(projection(table), Pick(1, 2))

    def test_projection_accepts_a_candidate_whose_difference_is_noise(self) -> None:
        rng = np.random.default_rng(0)
        reference = list(0.02 + rng.normal(0, 0.005, 20))
        noisy = list(np.array(reference) + rng.normal(0, 0.01, 20))
        table = losses_table([noisy, reference])
        self.assertEqual(projection(table), Pick(1, 1))

    def test_with_one_dataset_there_is_no_standard_error_and_the_cheapest_wins(self) -> None:
        table = losses_table([[0.3], [0.2], [0.1]])
        self.assertEqual(projection(table), Pick(1, 1))

    def test_a_table_without_losses_says_what_it_lacks(self) -> None:
        bare = candidate_table({1: curve([0.1, 0.5, 0.6])}, 1)
        for rule in (one_se, projection, mcs):
            with self.subTest(rule=rule.__name__), self.assertRaises(MissingColumns):
                rule(bare)


class TestModelConfidenceSet(unittest.TestCase):
    def matrix(self) -> np.ndarray:
        rng = np.random.default_rng(1)
        groups = 20
        good = 0.05 + rng.normal(0, 0.004, (groups, 2))  # two indistinguishable candidates
        bad = 0.10 + rng.normal(0, 0.004, (groups, 1))
        return np.hstack([bad, good])

    def test_a_clearly_worse_candidate_is_eliminated_and_two_equal_ones_survive(self) -> None:
        members = mcs_members(self.matrix(), alpha=0.1, rounds=500, seed=0)
        self.assertEqual(sorted(members.tolist()), [1, 2])

    def test_the_same_seed_gives_the_same_set_and_a_smaller_alpha_a_larger_one(self) -> None:
        losses = self.matrix()
        np.testing.assert_array_equal(mcs_members(losses, 0.1, 300, 5), mcs_members(losses, 0.1, 300, 5))
        loose, strict = mcs_members(losses, 0.5, 300, 5), mcs_members(losses, 0.01, 300, 5)
        self.assertGreaterEqual(len(strict), len(loose))
        self.assertTrue(set(loose.tolist()) <= set(strict.tolist()))

    def test_identical_candidates_all_stay(self) -> None:
        same = np.tile(np.random.default_rng(2).uniform(0.01, 0.05, (15, 1)), (1, 4))
        self.assertEqual(len(mcs_members(same, 0.1, 200, 0)), 4)

    def test_the_rule_takes_the_cheapest_member(self) -> None:
        rng = np.random.default_rng(3)
        columns = [0.10 + rng.normal(0, 0.004, 20), 0.05 + rng.normal(0, 0.004, 20), 0.05 + rng.normal(0, 0.004, 20)]
        table = losses_table([list(c) for c in columns])
        self.assertEqual(mcs(table), Pick(1, 2))


class TestInformationCriteria(unittest.TestCase):
    """100 rows, sst 100, rss falling 50, 20, 10, 9.8, 9.7: the criteria differ by what they charge."""

    RSS: ClassVar[list[float]] = [50.0, 20.0, 10.0, 9.8, 9.7]

    def table(self, library: int = 600, code: float = 0.0) -> pl.DataFrame:
        return table_with(
            [0.5, 0.8, 0.9, 0.902, 0.903],
            [
                detail(i + 1, rss=rss, n_rows=100, sst=100.0, library=library, code=code * (i + 1))
                for i, rss in enumerate(self.RSS)
            ],
        )

    def test_aic_takes_the_fourth_term_and_bic_stops_at_the_third(self) -> None:
        self.assertEqual(aic(self.table()), Pick(1, 4))
        self.assertEqual(bic(self.table()), Pick(1, 3))

    def test_the_small_sample_correction_moves_aic_back_to_three_terms(self) -> None:
        self.assertEqual(aicc(self.table()), Pick(1, 3))

    def test_the_extended_bic_charges_for_searching_a_large_library_and_equals_bic_for_a_small_one(self) -> None:
        # The third term gains 6.45 on the BIC scale; searching 600 terms for it costs 6.8.
        def marginal(library: int) -> pl.DataFrame:
            rss = [50.0, 20.0, 17.9, 17.85]
            return table_with(
                [0.5, 0.8, 0.82, 0.821],
                [detail(i + 1, rss=r, n_rows=100, sst=100.0, library=library, code=0.0) for i, r in enumerate(rss)],
            )

        self.assertEqual(bic(marginal(600)), Pick(1, 3))
        self.assertEqual(ebic(marginal(600)), Pick(1, 2))
        self.assertEqual(ebic(marginal(3)), bic(marginal(3)))

    def test_a_description_length_with_free_structure_agrees_with_bic_and_a_dear_one_shortens_the_equation(
        self,
    ) -> None:
        self.assertEqual(mdl(self.table(code=0.0)), bic(self.table()))
        self.assertEqual(mdl(self.table(code=50.0)), Pick(1, 1))

    def test_a_perfect_fit_does_not_divide_by_zero(self) -> None:
        details = [detail(1, rss=0.0, n_rows=100, sst=100.0, library=10, code=1.0)]
        table = table_with([1.0], details)
        for rule in (aic, aicc, bic, ebic, mdl):
            self.assertEqual(rule(table), Pick(1, 1))

    def test_the_description_length_of_a_term_counts_its_choices(self) -> None:
        atom = Term("atom", (Atom("a", "log"),))
        ratio = Term("ratio", (Atom("a", "log"), Atom("b", "id")))
        feature = math.log(10) + math.log(5)
        self.assertAlmostEqual(description_length(atom, 10), math.log(5) + feature)
        self.assertAlmostEqual(description_length(ratio, 10), math.log(5) + 2 * feature)
        nested = Term("product", (ratio, Atom("c", "id")))
        self.assertAlmostEqual(description_length(nested, 10), math.log(5) + description_length(ratio, 10) + feature)


class TestShapeRules(unittest.TestCase):
    def corner(self) -> pl.DataFrame:
        """Rises by 0.2 a term to 1.0 at five terms, then 0.001 a term: the corner is at five."""
        return candidate_table({1: curve([0.2 * n for n in range(1, 6)] + [1.0 + 0.001 * k for k in range(1, 6)])}, 1)

    def test_the_l_method_splits_at_the_corner(self) -> None:
        self.assertEqual(l_method(self.corner()), Pick(1, 5))

    def test_the_bend_angle_is_largest_at_the_corner(self) -> None:
        self.assertEqual(bend_angle(self.corner()), Pick(1, 5))

    def test_the_log_slope_prefers_the_largest_relative_drop_in_error_per_slot(self) -> None:
        table = candidate_table({1: curve([0.5, 0.9, 0.91, 0.92])}, 1)
        self.assertEqual(log_slope(table), Pick(1, 2))

    def test_short_fronts_return_their_last_point(self) -> None:
        short = candidate_table({1: curve([0.1, 0.5])}, 1)
        for rule in (l_method, bend_angle):
            self.assertEqual(rule(short), Pick(1, 2))
        self.assertEqual(log_slope(candidate_table({1: curve([0.3])}, 1)), Pick(1, 1))


class TestStability(unittest.TestCase):
    def table(self, first: float, second: float) -> pl.DataFrame:
        return table_with(
            [0.90, 0.905, 0.91],
            [detail(1, sign_stability=first), detail(2, sign_stability=second), detail(3, sign_stability=1.0)],
        )

    def test_an_unstable_cheap_candidate_is_skipped_for_a_stable_one_within_delta(self) -> None:
        self.assertEqual(stable_within_delta(self.table(0.5, 0.95), delta=0.02), Pick(1, 2))

    def test_when_nothing_is_stable_it_falls_back_to_the_cheapest_within_delta(self) -> None:
        table = table_with([0.90, 0.91], [detail(1, sign_stability=0.1), detail(2, sign_stability=0.2)])
        self.assertEqual(stable_within_delta(table, delta=0.02), Pick(1, 1))

    def test_a_missing_stability_passes(self) -> None:
        table = table_with([0.90, 0.91], [detail(1, sign_stability=float("nan")), detail(2, sign_stability=1.0)])
        self.assertEqual(stable_within_delta(table, delta=0.02), Pick(1, 1))

    def test_a_candidate_outside_delta_is_never_taken(self) -> None:
        self.assertEqual(stable_within_delta(self.table(1.0, 1.0), delta=0.001), Pick(1, 3))


class TestAuto(unittest.TestCase):
    def table(self) -> pl.DataFrame:
        floors = {(1, 2): 0.80, (1, 3): 0.90, (2, 2): 0.90, (1, 4): 0.905}
        rows = [(a, n, f) for (a, n), f in floors.items()]
        return pl.DataFrame(
            {
                "arity": [a for a, _, _ in rows],
                "n_terms": [n for _, n, _ in rows],
                "complexity": [a * n for a, n, _ in rows],
                "floor": [f for _, _, f in rows],
                "smooth": [f for _, _, f in rows],
            }
        )

    def choose(self, picks: dict[str, Pick], **kwargs: float) -> tuple[str, Pick]:
        with mock.patch.dict(
            lengths.SELECTABLE_RULES, {name: (lambda _t, _p=p, **_k: _p) for name, p in picks.items()}, clear=True
        ):
            winner, pick, seen = auto_choice(self.table(), **kwargs)
        self.assertEqual(set(seen), set(picks))
        return winner, pick

    def test_the_highest_value_wins(self) -> None:
        winner, pick = self.choose({"low": Pick(1, 2), "high": Pick(1, 4), "mid": Pick(1, 3)})
        self.assertEqual((winner, pick), ("high", Pick(1, 4)))

    def test_equal_values_go_to_fewer_terms_then_lower_arity_then_the_earlier_rule(self) -> None:
        self.assertEqual(self.choose({"a": Pick(2, 2), "b": Pick(1, 3)})[1], Pick(2, 2))  # 2 terms beat 3
        two_terms = {"first": Pick(2, 2), "second": Pick(2, 2)}
        self.assertEqual(self.choose(two_terms)[0], "first")

    def test_auto_returns_the_pick_of_the_winner(self) -> None:
        picks = {"deep": Pick(1, 4), "lean": Pick(1, 3)}
        with mock.patch.dict(
            lengths.SELECTABLE_RULES, {n: (lambda _t, _p=p, **_k: _p) for n, p in picks.items()}, clear=True
        ):
            self.assertEqual(auto(self.table()), Pick(1, 4))
            self.assertEqual(auto(self.table(), tolerance=0.01), Pick(1, 3))

    def test_a_tolerance_trades_value_for_parsimony(self) -> None:
        picks = {"deep": Pick(1, 4), "lean": Pick(1, 3)}
        self.assertEqual(self.choose(picks)[1], Pick(1, 4))
        self.assertEqual(self.choose(picks, tolerance=0.01)[1], Pick(1, 3))

    def test_a_rule_the_table_cannot_serve_is_left_out(self) -> None:
        bare = candidate_table({1: curve([0.2, 0.6, 0.7, 0.71, 0.715, 0.72])}, 1)
        picks = all_picks(bare, delta=0.01, window=2, smoothing=1, fixed_arity=1)
        self.assertIn("pareto_knee", picks)
        self.assertNotIn("one_se", picks)
        self.assertNotIn("bic", picks)


class TestRegistryAndConfig(unittest.TestCase):
    def test_the_study_file_may_name_exactly_the_rules_that_exist(self) -> None:
        from ml_meta_perf.config import LENGTH_METHODS

        self.assertEqual(set(LENGTH_METHODS), set(LENGTH_RULES))
        self.assertEqual(LENGTH_METHODS[-1], "auto")

    def test_the_constants_of_the_rules_are_validated(self) -> None:
        base = load_config().selection
        for field, value, message in (
            ("alpha", 0.0, "alpha"),
            ("alpha", 1.0, "alpha"),
            ("se_multiple", 0.0, "se_multiple"),
            ("bootstrap_rounds", 10, "bootstrap_rounds"),
            ("stability_min", 1.5, "stability_min"),
            ("tolerance", -0.1, "tolerance"),
        ):
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ConfigError, message):
                dataclasses.replace(base, **{field: value})

    def test_a_study_file_without_the_constants_gets_the_conventional_ones(self) -> None:
        bare = SelectionConfig(delta=0.01, window=4, smoothing=3, method="auto", capability_arity=3)
        self.assertEqual((bare.alpha, bare.se_multiple, bare.stability_min, bare.tolerance), (0.1, 1.0, 0.9, 0.0))


class TestOnAFittedCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.corpus = planted(3, noise=0.2, seed=7)
        cls.config = dataclasses.replace(default_configuration(), max_terms=5, pool_size=60, beam_width=2, max_arity=2)

    def test_the_details_reproduce_the_floor_and_the_in_sample_fit(self) -> None:
        fitted = fit_path(
            self.corpus.frame, DATASET_FEATURES, MODEL_FEATURES, self.config, "E3", schema=self.corpus.schema()
        )
        details = candidate_details(fitted)
        floor = lengths.floor_curve(fitted.curve)
        for row, expected in zip(details.iter_rows(named=True), floor, strict=True):
            self.assertAlmostEqual(1.0 - sum(row["loss"]) / row["sst"], expected, places=9)
            self.assertGreater(row["rss"], 0.0)
            self.assertEqual(row["n_rows"], 112)
            self.assertTrue(0.0 <= row["sign_stability"] <= 1.0)
        self.assertEqual(len(details["loss"][0]), 14)

    def test_the_in_sample_losses_sum_to_the_residual_and_a_study_without_lodo_has_no_stability(self) -> None:
        from ml_meta_perf.experiment import _group_losses  # pyright: ignore[reportPrivateUsage]

        schema = self.corpus.schema(protocols=("IS", "LOMO"))
        fitted = fit_path(self.corpus.frame, DATASET_FEATURES, MODEL_FEATURES, self.config, "E3", schema=schema)
        details = candidate_details(fitted)
        for row in details.iter_rows(named=True):
            self.assertAlmostEqual(
                float(_group_losses(fitted, row["n_terms"], "in_sample").sum()), row["rss"], places=9
            )
            self.assertTrue(math.isnan(row["sign_stability"]))

    def test_auto_runs_every_rule_and_chooses_by_value_then_terms_then_arity(self) -> None:
        selection = dataclasses.replace(load_config().selection, method="auto", window=2, capability_arity=2)
        report = run_equation(
            self.corpus.frame,
            DATASET_FEATURES,
            MODEL_FEATURES,
            self.config,
            selection,
            "E3",
            schema=self.corpus.schema(),
        )
        rules = report.rules
        assert rules is not None
        self.assertEqual(rules.height, len(lengths.SELECTABLE_RULES))
        self.assertEqual(int(rules["chosen"].sum()), 1)
        best = rules["floor"].max()
        expected = rules.filter(pl.col("floor") >= best).sort(["n_terms", "arity"]).row(0, named=True)
        self.assertEqual((report.arity, report.n_terms), (expected["arity"], expected["n_terms"]))

    def test_a_single_rule_leaves_a_one_row_table(self) -> None:
        selection = dataclasses.replace(load_config().selection, method="one_se", window=2)
        report = run_equation(
            self.corpus.frame,
            DATASET_FEATURES,
            MODEL_FEATURES,
            self.config,
            selection,
            "E3",
            schema=self.corpus.schema(),
        )
        assert report.rules is not None
        self.assertEqual(report.rules["rule"].to_list(), ["one_se"])

    def test_the_plateau_method_keeps_its_fast_path_and_has_no_table(self) -> None:
        selection = dataclasses.replace(load_config().selection, method="plateau_knee", window=2)
        report = run_equation(
            self.corpus.frame,
            DATASET_FEATURES,
            MODEL_FEATURES,
            self.config,
            selection,
            "E3",
            schema=self.corpus.schema(),
        )
        self.assertIsNone(report.rules)


if __name__ == "__main__":
    unittest.main()
