"""Choosing the equation length: the rule, the front, and the alternatives it beats.

Knee detection was removed on 2026-09-07 and these tests replaced the ones covering it. The
reason is in `selection`'s module docstring: every geometric reading of this curve -- three
detectors, with and without gRDP smoothing, plus the Pareto-front knee by three standard
rules -- puts the bend at four to eight terms, and every one of those lengths is
significantly worse than the selected equation when paired over the twenty held-out datasets.
The geometry stays as a reported diagnostic; `best_length` decides.
"""

import inspect
import unittest

import numpy as np
import polars as pl

from ml_meta_perf.selection import (
    arity_candidates,
    best_configuration,
    best_length,
    complexity,
    consensus_curve,
    floor_curve,
    grammar_margin,
    most_capable,
    pareto_front,
    pareto_knee,
    pareto_table,
    protocol_spread,
    recommend,
)


def curve(
    sizes: list[int],
    in_sample: list[float],
    loo: list[float] | None = None,
    loo_model: list[float] | None = None,
    loo_cell: list[float] | None = None,
) -> pl.DataFrame:
    data: dict[str, list[float] | list[int]] = {"n_terms": sizes, "r2_in_sample": in_sample}
    if loo is not None:
        data["r2_loo_dataset"] = loo
    if loo_model is not None:
        data["r2_loo_model"] = loo_model
    if loo_cell is not None:
        data["r2_loo_cell"] = loo_cell
    return pl.DataFrame(data)


def flat_errors(curves: dict[int, pl.DataFrame], error: float = 0.10) -> dict[int, dict[int, np.ndarray]]:
    """Per-group errors that are identical everywhere, so no paired test can separate anything.

    The default state for a rule test: with nothing distinguishable, `best_configuration` must
    fall back on complexity alone, which is the behaviour worth pinning separately from the
    behaviour when a difference is real.
    """
    return {
        arity: {int(size): np.full(20, error) for size in table["n_terms"].to_list()} for arity, table in curves.items()
    }


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


class TestFloorCurve(unittest.TestCase):
    """A length scores as its worst protocol, over all four including the cell protocol."""

    def test_it_takes_the_minimum_over_every_protocol_present(self) -> None:
        table = curve([4], [0.70], [0.65], [0.60], [0.55])
        self.assertAlmostEqual(float(floor_curve(table)[0]), 0.55)

    def test_the_cell_protocol_is_read_and_the_median_would_have_hidden_it(self) -> None:
        """The point of the minimum. Three protocols agree at 0.66 and the cell reads 0.50;
        a median of the four sits at 0.66 and reports a number no protocol achieved."""
        table = curve([4], [0.66], [0.66], [0.66], [0.50])
        self.assertAlmostEqual(float(floor_curve(table)[0]), 0.50)
        self.assertGreater(float(consensus_curve(table)[0]), 0.60)

    def test_it_works_on_a_curve_that_has_not_been_scored_on_every_protocol(self) -> None:
        np.testing.assert_allclose(floor_curve(curve([2, 4], [0.5, 0.6], [0.4, 0.55])), [0.4, 0.55])

    def test_it_refuses_a_curve_with_no_protocol_columns(self) -> None:
        with self.assertRaises(ValueError):
            floor_curve(pl.DataFrame({"n_terms": [2, 4]}))


class TestGrammarMargin(unittest.TestCase):
    """Gain, the paired spread of that gain, and the ratio the rule compares against one."""

    def test_a_gain_with_no_fold_to_fold_variation_is_perfectly_consistent(self) -> None:
        """Not unmeasurable -- the reference wins on every group by the same amount, which is
        the strongest evidence a paired comparison can carry."""
        gain, scale, ratio = grammar_margin(np.full(20, 0.12), np.full(20, 0.10))
        self.assertAlmostEqual(gain, 0.02)
        self.assertAlmostEqual(scale, 0.0)
        self.assertEqual(ratio, float("inf"))

    def test_identical_errors_give_a_zero_ratio(self) -> None:
        """`most_capable` compares against itself in the loop and must never reject itself."""
        errors = np.linspace(0.05, 0.20, 20)
        self.assertAlmostEqual(grammar_margin(errors, errors)[2], 0.0)

    def test_a_larger_grammar_that_is_worse_can_never_be_taken(self) -> None:
        """A gain of zero or less is no gain, whatever its spread."""
        self.assertAlmostEqual(grammar_margin(np.full(20, 0.08), np.full(20, 0.10))[2], 0.0)

    def test_concentrating_a_gain_in_fewer_groups_lowers_the_ratio(self) -> None:
        """The bar reads the mean's signal-to-noise, so the same mean gain scores lower the
        fewer groups deliver it. It is looser than a sign test and `grammar_margin`'s
        docstring says so: one group of twenty carrying all of it lands at 1.02, on the bar
        rather than under it, where a sign test would call that a tie."""
        reference = np.full(20, 0.10)
        ratios = [grammar_margin(np.array([0.10] * (20 - n) + [0.10 + 0.02 / n] * n), reference)[2] for n in (1, 2, 4)]
        self.assertEqual(ratios, sorted(ratios))
        self.assertAlmostEqual(ratios[0], 1.02, places=2)

    def test_it_refuses_mismatched_folds(self) -> None:
        with self.assertRaises(ValueError):
            grammar_margin(np.zeros(20), np.zeros(19))


class TestProtocolSpread(unittest.TestCase):
    """How far a length falls from its fit to its worst protocol. Reported, never selected on."""

    def test_it_measures_the_drop_from_the_fit_to_the_floor(self) -> None:
        table = curve([4], [0.70], [0.65], [0.63], [0.60])
        self.assertAlmostEqual(float(protocol_spread(table)[0]), 0.10)

    def test_the_floor_alone_cannot_tell_two_equations_apart_and_this_can(self) -> None:
        """Two lengths reaching the same worst protocol from a different fit are one number to
        `floor_curve`. They are not the same equation, and this is the column that says so."""
        table = curve([4, 8], [0.66, 0.80], [0.64, 0.70], [0.63, 0.68], [0.60, 0.60])
        np.testing.assert_allclose(floor_curve(table), [0.60, 0.60])
        np.testing.assert_allclose(protocol_spread(table), [0.06, 0.20])

    def test_it_refuses_a_curve_with_nothing_to_measure_the_drop_from(self) -> None:
        with self.assertRaises(ValueError):
            protocol_spread(pl.DataFrame({"n_terms": [2], "r2_loo_dataset": [0.5]}))

    def test_it_does_not_decide_anything(self) -> None:
        """The spread is not in the selection score, deliberately: weighting a level against a
        spread is the free parameter this rule was revised to remove. Give the simplest
        candidate the *worse* spread and it is still selected."""
        curves = {
            2: curve([15], [0.90], [0.60], [0.60], [0.60]),
            3: curve([23], [0.62], [0.61], [0.61], [0.61]),
        }
        self.assertGreater(float(protocol_spread(curves[2])[0]), float(protocol_spread(curves[3])[0]))
        self.assertEqual(best_configuration(curves, flat_errors(curves)), (2, 15))


class TestBestConfiguration(unittest.TestCase):
    """The rule that picks the equation: a length per grammar by argmax, then a paired test.

    **This replaced the adjusted-consensus rule on 2026-09-08**, and the defect it fixes is
    the one `best_configuration`'s docstring sets out: the old rule priced a feature slot by
    the corpus size, so holding the curve and the folds fixed and growing the corpus from 476
    rows to 5,000 flipped its answer from (2, 15) to (3, 23) with the plateau unmoved. There
    is no row count in this one to flip it, which `test_no_corpus_size_enters_the_rule` pins.

    The disclosure the old tests carried still applies: the rule was written knowing the
    answer it had to reproduce, so these pin its mechanics rather than its verdict.
    """

    def test_complexity_charges_for_the_grammar_not_the_coefficients(self) -> None:
        """Fifteen terms at arity 3 fit the same sixteen numbers as fifteen at arity 2, so a
        coefficient count cannot tell them apart. This is what does."""
        self.assertEqual(complexity(2, 15), 30)
        self.assertEqual(complexity(3, 15), 45)

    def test_each_grammar_contributes_one_candidate_at_its_own_floor_argmax(self) -> None:
        curves = {
            2: curve([5, 10, 15], [0.50, 0.60, 0.58], [0.50, 0.60, 0.58], [0.50, 0.60, 0.58], [0.50, 0.60, 0.58]),
            3: curve([5, 10, 15], [0.50, 0.58, 0.62], [0.50, 0.58, 0.62], [0.50, 0.58, 0.62], [0.50, 0.58, 0.62]),
        }
        self.assertEqual(arity_candidates(curves), {2: 10, 3: 15})

    def test_a_candidate_is_the_worst_protocol_argmax_not_the_median_one(self) -> None:
        """Ten terms wins on three protocols and craters on the cell; five is the candidate."""
        curves = {
            2: curve([5, 10], [0.60, 0.66], [0.60, 0.66], [0.60, 0.66], [0.58, 0.30]),
        }
        self.assertEqual(arity_candidates(curves), {2: 5})

    def test_the_simplest_indistinguishable_grammar_wins(self) -> None:
        """With nothing separable, the smallest `complexity` is the answer."""
        curves = {
            2: curve([5, 15], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60]),
            3: curve([5, 23], [0.50, 0.61], [0.50, 0.61], [0.50, 0.61], [0.50, 0.61]),
        }
        self.assertEqual(most_capable(curves), (3, 23))
        self.assertEqual(best_configuration(curves, flat_errors(curves)), (2, 15))

    def test_a_grammar_that_is_measurably_better_is_taken(self) -> None:
        """The comparison is not decoration: give arity 3 a real per-group advantage and the
        rule stops preferring the simpler grammar."""
        curves = {
            2: curve([5, 15], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60]),
            3: curve([5, 23], [0.50, 0.61], [0.50, 0.61], [0.50, 0.61], [0.50, 0.61]),
        }
        errors = flat_errors(curves)
        errors[3][23] = np.full(20, 0.04)
        self.assertEqual(best_configuration(curves, errors), (3, 23))

    def test_the_decision_is_a_ratio_to_overcome_not_a_failed_test(self) -> None:
        """The correction that produced `grammar_margin`.

        A gain the size of its own spread is the boundary. Below it the simpler grammar
        stands; above it the larger one is taken. Both sides are measured, so the rule is a
        comparison of two numbers rather than a failure to reject a null.
        """
        rng = np.random.default_rng(0)
        base = 0.10 + rng.normal(0.0, 0.02, 20)
        curves = {
            2: curve([15], [0.60], [0.60], [0.60], [0.60]),
            3: curve([23], [0.61], [0.61], [0.61], [0.61]),
        }
        # Independent per-fold noise on both sides, so the *difference* varies from fold to
        # fold as it does on the real corpus. A difference with no variation at all is the
        # separate degenerate case `TestGrammarMargin` covers.
        small = {2: {15: base}, 3: {23: base - 0.001 + rng.normal(0.0, 0.02, 20)}}
        large = {2: {15: base}, 3: {23: base - 0.050 + rng.normal(0.0, 0.02, 20)}}
        self.assertLess(grammar_margin(small[2][15], small[3][23])[2], 1.0)
        self.assertGreater(grammar_margin(large[2][15], large[3][23])[2], 1.0)
        self.assertEqual(best_configuration(curves, small), (2, 15))
        self.assertEqual(best_configuration(curves, large), (3, 23))

    def test_a_grammar_that_is_merely_better_on_average_is_not_taken(self) -> None:
        """Half the groups better and half worse is a mean, not a result -- the standing rule
        against reading a difference of two means over twenty folds."""
        curves = {
            2: curve([5, 15], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60]),
            3: curve([5, 23], [0.50, 0.61], [0.50, 0.61], [0.50, 0.61], [0.50, 0.61]),
        }
        errors = flat_errors(curves)
        errors[3][23] = np.array([0.02] * 10 + [0.18] * 10)
        self.assertEqual(best_configuration(curves, errors), (2, 15))

    def test_no_corpus_size_enters_the_rule(self) -> None:
        """The invariance the old rule failed, as a signature check.

        Adjusted consensus took ``rows`` and its verdict moved with it. Nothing here does, so
        the required check -- hold the curve fixed, vary the hypothetical corpus size, require
        the answer not to move -- is satisfied by construction rather than by measurement.
        """
        for function in (best_configuration, most_capable, arity_candidates, floor_curve):
            parameters = set(inspect.signature(function).parameters)
            self.assertNotIn("rows", parameters, f"{function.__name__} must not price by corpus size")
            self.assertNotIn("n", parameters, f"{function.__name__} must not price by corpus size")

    def test_most_capable_ignores_complexity(self) -> None:
        """The bound is not put forward as an equation to read, so length is not charged."""
        curves = {
            2: curve([5, 10], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60]),
            3: curve([5, 28], [0.50, 0.65], [0.50, 0.65], [0.50, 0.65], [0.50, 0.65]),
        }
        self.assertEqual(most_capable(curves), (3, 28))

    def test_no_configuration_is_hardcoded(self) -> None:
        """Moving where the floor peaks moves the answer."""
        for peak in (6, 12, 18):
            sizes = [6, 12, 18]
            scores = [0.5 + 0.1 * (size == peak) for size in sizes]
            curves = {2: curve(sizes, scores, scores, scores, scores)}
            self.assertEqual(best_configuration(curves, flat_errors(curves))[1], peak)


if __name__ == "__main__":
    unittest.main()
