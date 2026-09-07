"""The decision stage: whether a beam policy actually beats the published one.

The sweep proposes and this decides, so the thing to pin is the *verdict logic*. A sign error
here would report a worse beam as better, and the numbers involved are small enough that
nobody reading the table would catch it.
"""

import unittest

import numpy as np

from ml_meta_perf.beam_compare import BETTER, TIE, WORSE, PolicyRun, compare, compare_all
from ml_meta_perf.experiment import DEFAULT_E3


def _run(name: str, fold_mae: list[float], seconds: float = 1.0, r2: float = 0.6) -> PolicyRun:
    return PolicyRun(
        policy=name,
        configuration=DEFAULT_E3,
        features=("Model Capability",),
        n_terms=15,
        terms=("a", "b", "c"),
        r2_in_sample=r2 + 0.02,
        r2_loo_dataset=r2,
        r2_loo_model=r2 - 0.01,
        fold_mae=np.array(fold_mae),
        seconds=seconds,
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
