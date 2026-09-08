"""Choosing the equation length: the rule, the front, and the alternatives it beats.

Knee detection was removed on 2026-09-07 and these tests replaced the ones covering it. The
reason is in `selection`'s module docstring: every geometric reading of this curve -- three
detectors, with and without gRDP smoothing, plus the Pareto-front knee by three standard
rules -- puts the bend at four to eight terms, and every one of those lengths is
significantly worse than the selected equation when paired over the twenty held-out datasets.
The geometry stays as a reported diagnostic; `best_length` decides.
"""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.selection import (
    adjusted_consensus,
    best_configuration,
    best_length,
    complexity,
    consensus_curve,
    most_capable,
    pareto_front,
    pareto_knee,
    pareto_table,
    recommend,
)


def curve(
    sizes: list[int],
    in_sample: list[float],
    loo: list[float] | None = None,
    loo_model: list[float] | None = None,
) -> pl.DataFrame:
    data: dict[str, list[float] | list[int]] = {"n_terms": sizes, "r2_in_sample": in_sample}
    if loo is not None:
        data["r2_loo_dataset"] = loo
    if loo_model is not None:
        data["r2_loo_model"] = loo_model
    return pl.DataFrame(data)


class TestConsensusCurve(unittest.TestCase):
    """One score per length across the protocols, so no single curve chooses the length."""

    def setUp(self) -> None:
        self.table = curve(
            [2, 4, 6, 8],
            [0.40, 0.50, 0.60, 0.66],
            [0.30, 0.45, 0.10, 0.62],  # 0.10 is a crater, as loo-dataset genuinely has
            [0.35, 0.47, 0.58, 0.60],
        )

    def test_the_median_ignores_a_single_protocol_crater(self) -> None:
        """Why median and not mean.

        At the cratered length the three protocols read 0.60 / 0.10 / 0.58. The median takes
        0.58 -- one fold extrapolating outside the training hull is a property of that fold,
        not of the length -- while the mean is dragged down by it.
        """
        median = consensus_curve(self.table, "median")
        mean = consensus_curve(self.table, "mean")
        self.assertAlmostEqual(float(median[2]), 0.58)
        self.assertAlmostEqual(float(mean[2]), (0.60 + 0.10 + 0.58) / 3)
        self.assertGreater(float(median[2]), float(mean[2]))

    def test_min_is_the_conservative_reading(self) -> None:
        self.assertAlmostEqual(float(consensus_curve(self.table, "min")[2]), 0.10)

    def test_a_single_protocol_is_its_own_consensus(self) -> None:
        plain = curve([2, 4, 6], [0.2, 0.5, 0.6])
        np.testing.assert_allclose(consensus_curve(plain), [0.2, 0.5, 0.6])

    def test_it_refuses_a_curve_with_no_protocol_columns(self) -> None:
        with self.assertRaises(ValueError):
            consensus_curve(pl.DataFrame({"n_terms": [2, 4]}))


class TestBestLength(unittest.TestCase):
    """The rule: argmax of the consensus curve. No threshold, no smoothing, no sensitivity."""

    def test_it_picks_the_maximum_of_the_consensus(self) -> None:
        table = curve(
            [2, 4, 6, 8, 10],
            [0.40, 0.50, 0.60, 0.66, 0.70],
            [0.30, 0.45, 0.62, 0.55, 0.50],
            [0.35, 0.47, 0.61, 0.54, 0.49],
        )
        self.assertEqual(best_length(table), 6)

    def test_it_never_runs_on_in_sample_alone_by_default(self) -> None:
        """In-sample is monotone in terms, so its argmax is always the longest equation.

        A rule reading it alone would be vacuous, which is why the default is the consensus.
        """
        table = curve(
            [2, 4, 6, 8],
            [0.40, 0.50, 0.60, 0.70],  # monotone: argmax is always the last
            [0.30, 0.62, 0.45, 0.40],
            [0.35, 0.61, 0.44, 0.39],
        )
        self.assertEqual(best_length(table, "r2_in_sample"), 8)
        self.assertEqual(best_length(table), 4)

    def test_it_returns_a_length_on_the_curve(self) -> None:
        table = curve([3, 9, 27], [0.1, 0.5, 0.4], [0.1, 0.6, 0.3], [0.1, 0.5, 0.3])
        self.assertIn(best_length(table), {3, 9, 27})

    def test_no_length_is_hardcoded(self) -> None:
        """Shifting the curve's peak shifts the answer, which a written-down value would not."""
        base = [0.1, 0.2, 0.9, 0.3, 0.2]
        for peak in range(5):
            scores = [0.1] * 5
            scores[peak] = 0.9
            table = curve([1, 2, 3, 4, 5], base, scores, scores)
            self.assertEqual(best_length(table), peak + 1)


class TestParetoFront(unittest.TestCase):
    def setUp(self) -> None:
        self.table = curve(
            [2, 4, 6, 8, 10],
            [0.20, 0.45, 0.52, 0.55, 0.56],
            [0.10, 0.40, 0.30, 0.50, 0.45],
        )

    def test_a_length_beaten_by_a_shorter_one_is_dominated(self) -> None:
        """Six terms scores 0.30 where four already reached 0.40, so it is off the front."""
        kept = pareto_front(self.table)["n_terms"].to_list()
        self.assertIn(4, kept)
        self.assertNotIn(6, kept)
        self.assertIn(8, kept)

    def test_the_in_sample_front_is_every_length(self) -> None:
        """Fit is monotone in terms, so nothing is ever dominated.

        That is why the in-sample curve cannot choose a length by itself.
        """
        table = pareto_table(self.table)
        self.assertTrue(all(table["front_in_sample"].to_list()))

    def test_both_fronts_are_flagged(self) -> None:
        table = pareto_table(self.table)
        self.assertIn("front_in_sample", table.columns)
        self.assertIn("front_loo_dataset", table.columns)


class TestParetoKnee(unittest.TestCase):
    """The three standard multi-criteria rules, reported as diagnostics rather than used."""

    def setUp(self) -> None:
        sizes = list(range(1, 17))
        scores = [1.0 - 0.75 * (0.6**k) for k in range(16)]
        self.table = curve(sizes, scores, scores, scores)

    def test_all_three_rules_are_reported(self) -> None:
        self.assertEqual(
            set(pareto_knee(self.table)),
            {"closest_to_ideal", "furthest_from_nadir", "furthest_from_chord"},
        )

    def test_each_rule_picks_a_length_on_the_curve(self) -> None:
        available = set(self.table["n_terms"].to_list())
        for length in pareto_knee(self.table).values():
            self.assertIn(length, available)

    def test_it_finds_the_bend_of_a_saturating_curve(self) -> None:
        """Not the first or the last point -- the rules would be worthless otherwise."""
        for length in pareto_knee(self.table).values():
            self.assertGreater(length, 1)
            self.assertLess(length, 16)

    def test_a_front_too_short_to_bend_returns_nothing(self) -> None:
        self.assertEqual(pareto_knee(curve([2, 4], [0.3, 0.5], [0.2, 0.4])), {})


class TestRecommend(unittest.TestCase):
    def setUp(self) -> None:
        self.table = curve(
            [2, 4, 6, 8, 10, 12, 14],
            [0.20, 0.45, 0.52, 0.55, 0.560, 0.564, 0.566],
            [0.05, 0.22, 0.26, 0.31, 0.330, 0.372, 0.441],
            [0.10, 0.25, 0.28, 0.33, 0.350, 0.380, 0.430],
        )

    def test_no_rule_is_a_knee_detector(self) -> None:
        """Knee detection was removed; nothing may reintroduce it under its old label."""
        rules = " ".join(recommend(self.table)["rule"].to_list())
        self.assertNotIn("knee (in-sample)", rules)
        self.assertNotIn("knee (loo-dataset)", rules)

    def test_it_reports_the_alternatives_the_rule_beats(self) -> None:
        """A selection rule is only defensible if what it beats is on the page."""
        rules = recommend(self.table)["rule"].to_list()
        self.assertTrue(any(rule.startswith("pareto front") for rule in rules))
        self.assertIn("best loo-dataset", rules)
        self.assertIn("best consensus (the rule)", rules)

    def test_best_rule_picks_the_maximum(self) -> None:
        rows = {row["rule"]: row for row in recommend(self.table).iter_rows(named=True)}
        self.assertEqual(rows["best loo-dataset"]["n_terms"], 14)
        self.assertAlmostEqual(rows["best loo-dataset"]["r2_loo_dataset"], 0.441)

    def test_the_rule_row_agrees_with_best_length(self) -> None:
        rows = {row["rule"]: row for row in recommend(self.table).iter_rows(named=True)}
        self.assertEqual(rows["best consensus (the rule)"]["n_terms"], best_length(self.table))

    def test_the_published_length_is_reported_when_given(self) -> None:
        rows = {row["rule"]: row for row in recommend(self.table, published=8).iter_rows(named=True)}
        self.assertEqual(rows["published"]["n_terms"], 8)

    def test_every_recommendation_is_a_length_on_the_curve(self) -> None:
        available = set(self.table["n_terms"].to_list())
        for value in recommend(self.table, published=8)["n_terms"].to_list():
            self.assertIn(value, available)

    def test_works_without_a_cross_validated_column(self) -> None:
        plain = curve([2, 4, 6, 8], [0.2, 0.45, 0.52, 0.55])
        self.assertGreater(recommend(plain).height, 0)


class TestAdjustedConsensus(unittest.TestCase):
    """The rule that picks the equation: one number, one argmax over arity x length.

    It replaced a chain of conditions ("the simplest grammar whose own best is not
    significantly worse than the best overall"), which produced the same answer and could not
    be read off a table. **It is also the rule flagged for revision** -- it was arrived at
    knowing the answer it had to reproduce, so these pin its mechanics rather than its verdict.
    """

    def test_complexity_charges_for_the_grammar_not_the_coefficients(self) -> None:
        """Fifteen terms at arity 3 fit the same sixteen numbers as fifteen at arity 2, so a
        coefficient count cannot tell them apart. This is what does."""
        self.assertEqual(complexity(2, 15), 30)
        self.assertEqual(complexity(3, 15), 45)

    def test_the_penalty_grows_with_complexity(self) -> None:
        table = curve([10, 10], [0.7, 0.7], [0.7, 0.7], [0.7, 0.7])
        cheap = adjusted_consensus(table, arity=2, rows=476)[0]
        dear = adjusted_consensus(table, arity=4, rows=476)[0]
        self.assertGreater(cheap, dear)

    def test_equal_complexity_scores_equally(self) -> None:
        table = curve([6], [0.6], [0.6], [0.6])
        self.assertAlmostEqual(
            float(adjusted_consensus(table, arity=2, rows=476)[0]),
            float(adjusted_consensus(curve([4], [0.6], [0.6], [0.6]), arity=3, rows=476)[0]),
            places=3,
        )

    def test_a_longer_equation_must_earn_its_length(self) -> None:
        """The whole point of the discount: more terms at the same consensus scores worse."""
        table = curve([5, 25], [0.6, 0.6], [0.6, 0.6], [0.6, 0.6])
        scores = adjusted_consensus(table, arity=2, rows=476)
        self.assertGreater(float(scores[0]), float(scores[1]))

    def test_it_refuses_a_configuration_with_no_degrees_of_freedom_left(self) -> None:
        """`rows - p - 1 <= 0` has no correction to make, and must not return a number."""
        table = curve([300], [0.6], [0.6], [0.6])
        self.assertFalse(np.isfinite(adjusted_consensus(table, arity=4, rows=476)[0]))

    def test_best_configuration_takes_the_argmax_over_arities(self) -> None:
        curves = {
            2: curve([5, 10], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60]),
            3: curve([5, 10], [0.50, 0.605], [0.50, 0.605], [0.50, 0.605]),
        }
        # At 476 rows a slot costs about 0.0008 of consensus, so arity 3's ten extra slots
        # cost ~0.008 and it only buys 0.005. A gap wider than the price would win, correctly.
        self.assertEqual(best_configuration(curves, rows=476), (2, 10))

    def test_most_capable_ignores_complexity(self) -> None:
        """The bound is not put forward as an equation to read, so length is not charged."""
        curves = {
            2: curve([5, 10], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60]),
            3: curve([5, 28], [0.50, 0.65], [0.50, 0.65], [0.50, 0.65]),
        }
        self.assertEqual(most_capable(curves), (3, 28))

    def test_no_configuration_is_hardcoded(self) -> None:
        """Moving where the consensus peaks moves the answer."""
        for peak in (6, 12, 18):
            sizes = [6, 12, 18]
            scores = [0.5 + 0.1 * (size == peak) for size in sizes]
            curves = {2: curve(sizes, scores, scores, scores)}
            self.assertEqual(best_configuration(curves, rows=476)[1], peak)

if __name__ == "__main__":
    unittest.main()
