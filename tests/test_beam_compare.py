"""The decision stage: whether a beam policy actually beats the published one.

The sweep proposes and this decides, so the thing to pin is the *verdict logic*. A sign error
here would report a worse beam as better, and the numbers involved are small enough that
nobody reading the table would catch it.
"""

import unittest

import numpy as np

from ml_meta_perf.beam_compare import (
    BETTER,
    PROTOCOLS,
    SPEED_TOLERANCE,
    TIE,
    TIMING_REPEATS,
    WORSE,
    PolicyRun,
    compare,
    compare_all,
)
from ml_meta_perf.experiment import DEFAULT_E3


def _run(
    name: str,
    fold_mae: list[float],
    seconds: float = 1.0,
    r2: float = 0.6,
    cell: float | None = None,
) -> PolicyRun:
    return PolicyRun(
        policy=name,
        configuration=DEFAULT_E3,
        features=("Model Capability",),
        n_terms=15,
        terms=("a", "b", "c"),
        r2_in_sample=r2 + 0.02,
        r2_loo_dataset=r2,
        r2_loo_model=r2 - 0.01,
        r2_loo_cell=r2 - 0.02 if cell is None else cell,
        ranking_map=(0.8, 0.75, 0.74, 0.72),
        fold_mae=np.array(fold_mae),
        seconds=seconds,
        seconds_median=seconds,
        repeats=TIMING_REPEATS,
    )


class TestVerdicts(unittest.TestCase):
    """A verdict names the *challenger*, and lower error is better. Both inversions in one
    place, because getting either backwards flips every row of the reported table."""

    def test_a_challenger_with_lower_error_everywhere_is_better(self) -> None:
        incumbent = _run("vanilla", [0.20] * 20)
        challenger = _run("cap-2", [0.10] * 20)
        verdict = compare(incumbent, challenger)
        self.assertEqual(verdict["verdict"], BETTER)
        self.assertGreater(float(verdict["mae_gain"]), 0.0)  # type: ignore[arg-type]

    def test_a_challenger_with_higher_error_everywhere_is_worse(self) -> None:
        incumbent = _run("vanilla", [0.10] * 20)
        challenger = _run("cap-2", [0.20] * 20)
        verdict = compare(incumbent, challenger)
        self.assertEqual(verdict["verdict"], WORSE)
        self.assertLess(float(verdict["mae_gain"]), 0.0)  # type: ignore[arg-type]

    def test_identical_errors_are_a_tie(self) -> None:
        """A tie is not a win: matching the incumbent means more machinery for no gain."""
        errors = [0.1, 0.2, 0.15, 0.3] * 5
        self.assertEqual(compare(_run("vanilla", errors), _run("cap-2", errors))["verdict"], TIE)

    def test_a_few_large_wins_against_many_small_losses_is_a_tie(self) -> None:
        """The failure mode `paired_comparison` exists for: a favourable *mean* produced by a
        couple of folds, which this project has read as a result three times."""
        incumbent = _run("vanilla", [0.20] + [0.10] * 19)
        challenger = _run("cap-2", [0.02] + [0.11] * 19)
        self.assertEqual(compare(incumbent, challenger)["verdict"], TIE)

    def test_reports_the_speedup_against_the_incumbent(self) -> None:
        """A policy that buys accuracy by spending time should not read as simply better."""
        verdict = compare(_run("vanilla", [0.2] * 20, seconds=4.0), _run("cap-2", [0.1] * 20, seconds=2.0))
        self.assertAlmostEqual(float(verdict["speedup"]), 2.0)  # type: ignore[arg-type]

    def test_counts_shared_terms(self) -> None:
        incumbent = _run("vanilla", [0.2] * 20)
        challenger = _run("cap-2", [0.1] * 20)
        self.assertEqual(compare(incumbent, challenger)["shared_terms"], 3)


class TestCompareAll(unittest.TestCase):
    def test_orders_by_gain_with_the_best_first(self) -> None:
        runs = [
            _run("vanilla", [0.20] * 20),
            _run("worse", [0.30] * 20),
            _run("better", [0.10] * 20),
        ]
        table = compare_all(runs)
        self.assertEqual(table["policy"].to_list(), ["better", "worse"])

    def test_the_incumbent_is_not_compared_with_itself(self) -> None:
        runs = [_run("vanilla", [0.2] * 20), _run("cap-2", [0.1] * 20)]
        self.assertNotIn("vanilla", compare_all(runs)["policy"].to_list())

    def test_a_missing_incumbent_reports_no_comparison_rather_than_raising(self) -> None:
        """Hours of cluster time should not end in a traceback because one fit failed."""
        self.assertEqual(compare_all([_run("cap-2", [0.1] * 20)]).height, 0)

    def test_no_challengers_reports_no_comparison(self) -> None:
        self.assertEqual(compare_all([_run("vanilla", [0.2] * 20)]).height, 0)


class TestProtocolSpread(unittest.TestCase):
    """The study's claim is that R2 holds across all four protocols, so the spread between them
    is a reported quantity and not a derived curiosity."""

    def test_spread_is_the_range_over_the_four_protocols(self) -> None:
        run = _run("vanilla", [0.2] * 20, r2=0.60, cell=0.50)
        # in-sample 0.62, loo-dataset 0.60, loo-model 0.59, loo-cell 0.50
        self.assertAlmostEqual(run.r2_spread, 0.12)
        self.assertAlmostEqual(run.r2_worst, 0.50)

    def test_a_policy_that_gives_up_the_strictest_protocol_shows_it(self) -> None:
        """`cap-2` at the published configuration loses 0.21 of leave-one-cell while the paired
        test on leave-one-dataset-out calls it a tie. The spread is what makes that visible."""
        incumbent = _run("vanilla", [0.2] * 20, r2=0.64, cell=0.62)
        challenger = _run("cap-2", [0.2] * 20, r2=0.44, cell=0.40)
        verdict = compare(incumbent, challenger)
        self.assertEqual(verdict["verdict"], TIE)
        self.assertLess(float(verdict["r2_loo_cell_delta"]), -0.2)  # type: ignore[arg-type]
        self.assertGreater(float(verdict["r2_spread_delta"]), 0.0)  # type: ignore[arg-type]

    def test_every_protocol_gets_a_ranking_column(self) -> None:
        verdict = compare(_run("vanilla", [0.2] * 20), _run("cap-2", [0.1] * 20))
        for name in PROTOCOLS:
            self.assertIn(f"rank_map_{name}", verdict)


class TestStandingAndFrontier(unittest.TestCase):
    """Accuracy is not the only axis any more: the corpus is going to grow, so a policy that
    ties on accuracy and finishes sooner is worth having. `standing` is the column a reader
    acts on, so these pin every word it can say."""

    def _standing_of(self, incumbent: PolicyRun, challenger: PolicyRun) -> str:
        table = compare_all([incumbent, challenger])
        return str(table["standing"][0])

    def test_a_win_on_accuracy_is_better(self) -> None:
        self.assertEqual(
            self._standing_of(_run("vanilla", [0.20] * 20), _run("cap-2", [0.10] * 20)), "better"
        )

    def test_a_tie_that_is_genuinely_faster_is_cheaper(self) -> None:
        self.assertEqual(
            self._standing_of(
                _run("vanilla", [0.2] * 20, seconds=1.0), _run("cheap", [0.2] * 20, seconds=0.5)
            ),
            "cheaper",
        )

    def test_a_speedup_inside_the_tolerance_is_only_equal(self) -> None:
        """Three policies producing the *identical* equation still measured 0.977x to 1.001x,
        so a bare `speedup > 1` would promote the clock."""
        challenger = _run("noise", [0.2] * 20, seconds=1.0 / (1.0 + SPEED_TOLERANCE / 2))
        table = compare_all([_run("vanilla", [0.2] * 20, seconds=1.0), challenger])
        self.assertGreater(float(table["speedup"][0]), 1.0)
        self.assertEqual(str(table["standing"][0]), "equal")

    def test_speed_bought_with_the_strictest_protocol_is_worse(self) -> None:
        """The trade this column exists to refuse: twice as fast, ties on leave-one-dataset-out,
        and pays for it where the study's claim actually lives. The paired test alone calls this
        a tie, which is why `standing` reads all four protocols and not just one."""
        incumbent = _run("vanilla", [0.2] * 20, seconds=1.0, r2=0.64, cell=0.62)
        challenger = _run("cheap-but-leaky", [0.2] * 20, seconds=0.5, r2=0.64, cell=0.40)
        table = compare_all([incumbent, challenger])
        self.assertEqual(table["verdict"][0], TIE)
        self.assertGreater(float(table["speedup"][0]), 1.0 + SPEED_TOLERANCE)
        self.assertEqual(str(table["standing"][0]), "worse")

    def test_actionable_rows_sort_to_the_top(self) -> None:
        runs = [
            _run("vanilla", [0.20] * 20, seconds=1.0),
            _run("nothing", [0.20] * 20, seconds=1.0),
            _run("gain", [0.10] * 20, seconds=1.0),
        ]
        self.assertEqual(compare_all(runs)["policy"].to_list(), ["gain", "nothing"])

    def test_a_row_beaten_on_both_axes_is_off_the_frontier(self) -> None:
        runs = [
            _run("vanilla", [0.20] * 20, seconds=1.0),
            _run("fast-and-good", [0.10] * 20, seconds=0.5),
            _run("slow-and-bad", [0.30] * 20, seconds=2.0),
        ]
        marks = dict(zip(*compare_all(runs).select("policy", "frontier").iter_columns(), strict=True))
        self.assertTrue(marks["fast-and-good"])
        self.assertFalse(marks["slow-and-bad"])

    def test_a_row_that_wins_on_only_one_axis_stays_on_the_frontier(self) -> None:
        """The frontier is the point: accurate-but-slow and fast-but-tied are both answers,
        and collapsing them to one number is the thing this replaced."""
        runs = [
            _run("vanilla", [0.20] * 20, seconds=1.0),
            _run("accurate", [0.10] * 20, seconds=2.0),
            _run("quick", [0.20] * 20, seconds=0.5),
        ]
        marks = dict(zip(*compare_all(runs).select("policy", "frontier").iter_columns(), strict=True))
        self.assertTrue(marks["accurate"])
        self.assertTrue(marks["quick"])

    def test_the_incumbent_competes_for_the_frontier_too(self) -> None:
        """Doing nothing is always an option. Leaving it out of the Pareto set marked the
        least-bad losing policy `true`, which reads as a win and is the opposite of the truth."""
        runs = [_run("vanilla", [0.20] * 20, seconds=1.0), _run("slower-no-better", [0.20] * 20, seconds=2.0)]
        table = compare_all(runs)
        self.assertEqual(str(table["standing"][0]), "equal")
        self.assertFalse(bool(table["frontier"][0]))


class TestRunPolicy(unittest.TestCase):
    """One real fit, to check the run carries what the paired test needs."""

    @classmethod
    def setUpClass(cls) -> None:
        from ml_meta_perf.beam_compare import run_policy
        from ml_meta_perf.data import load
        from ml_meta_perf.experiment import EQUATION_MODEL_FEATURES

        cls.outcome = run_policy(load(), DEFAULT_E3, EQUATION_MODEL_FEATURES)

    def test_produces_one_error_per_held_out_dataset(self) -> None:
        assert self.outcome is not None
        self.assertEqual(self.outcome.fold_mae.shape, (20,))
        self.assertTrue(np.isfinite(self.outcome.fold_mae).all())

    def test_scores_every_protocol_including_the_strictest(self) -> None:
        """Four protocols, because the study's claim is that the equation holds across all of
        them. Leave-one-cell is the one an opaque regressor cannot survive."""
        assert self.outcome is not None
        for name, value in self.outcome.r2_by_protocol.items():
            self.assertTrue(np.isfinite(value), name)
        self.assertLess(self.outcome.r2_loo_cell, self.outcome.r2_in_sample)
        self.assertGreater(self.outcome.r2_loo_cell, 0.5)

    def test_ranking_is_scored_under_every_protocol_too(self) -> None:
        """R2 *and* ranking, or a policy could trade one away unseen."""
        assert self.outcome is not None
        self.assertEqual(len(self.outcome.ranking_map), len(PROTOCOLS))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in self.outcome.ranking_map))

    def test_the_timing_is_repeated_and_reports_its_best(self) -> None:
        """One measurement of a sub-second fit is not a measurement."""
        assert self.outcome is not None
        self.assertEqual(self.outcome.repeats, TIMING_REPEATS)
        self.assertLessEqual(self.outcome.seconds, self.outcome.seconds_median)

    def test_a_single_repeat_is_allowed_and_says_so(self) -> None:
        """`repeats=1` is the escape hatch for an expensive fit, and the run records it so no
        speed claim is made from one sample by accident."""
        from ml_meta_perf.beam_compare import run_policy
        from ml_meta_perf.data import load
        from ml_meta_perf.experiment import EQUATION_MODEL_FEATURES

        once = run_policy(load(), DEFAULT_E3, EQUATION_MODEL_FEATURES, repeats=1)
        assert once is not None
        self.assertEqual(once.repeats, 1)

    def test_the_default_policy_reproduces_the_published_scores(self) -> None:
        """`run_policy` under `beam.VANILLA` is the published path, so it must agree with it.
        If these drift apart, the comparison is against something other than the study."""
        from ml_meta_perf.data import load
        from ml_meta_perf.experiment import run_e3

        assert self.outcome is not None
        published = run_e3(load())
        self.assertAlmostEqual(
            self.outcome.r2_loo_dataset, float(published.cross_validated["loo_dataset"]["r2"]), places=9
        )
        self.assertEqual(self.outcome.n_terms, len(published.equation.terms))


if __name__ == "__main__":
    unittest.main()
