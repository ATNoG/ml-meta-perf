"""Validation protocols, baselines and the leakage they are designed to avoid."""

import unittest
from dataclasses import replace

import numpy as np

from ml_meta_perf.search import search
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import (
    CrossValidation,
    cross_validate_doubly_held_out,
    cross_validate_fixed_form,
    decision_report,
    fold_selections,
    leave_one_group_out,
    paired_comparison,
    ranking_report,
    score,
    sign_test,
    term_stability,
)


def grid(n_groups: int = 6, per_group: int = 5, seed: int = 4):
    """A small dataset-by-model grid with a known additive structure."""
    rng = np.random.default_rng(seed)
    outer, inner = [], []
    for g in range(n_groups):
        for m in range(per_group):
            outer.append(f"d{g}")
            inner.append(f"m{m}")
    outer_labels = np.array(outer)
    inner_labels = np.array(inner)
    columns = {
        "f1": np.array([2.0 + g for g in range(n_groups) for _ in range(per_group)]),
        "f2": np.array([5.0 + 2 * g for g in range(n_groups) for _ in range(per_group)]),
        "g1": np.array([1.0 + m for _ in range(n_groups) for m in range(per_group)]),
        "g2": np.array([3.0 + 0.5 * m for _ in range(n_groups) for m in range(per_group)]),
    }
    target = 0.05 * columns["f1"] + 0.04 * columns["g1"] + rng.normal(0, 0.01, len(outer))
    return columns, target, outer_labels, inner_labels


class TestSplitters(unittest.TestCase):
    def test_leave_one_group_out_covers_every_row_once(self) -> None:
        labels = np.array(["a", "a", "b", "c", "c", "c"])
        seen = np.zeros(6, dtype=int)
        for label, train, test in leave_one_group_out(labels):
            self.assertEqual(test.sum() + train.sum(), 6)
            self.assertFalse(np.any(train & test))
            self.assertTrue(np.all(labels[test] == label))
            seen += test
        np.testing.assert_array_equal(seen, np.ones(6, dtype=int))

    def test_one_fold_per_distinct_group(self) -> None:
        labels = np.array(["a", "a", "b", "c"])
        self.assertEqual(len(list(leave_one_group_out(labels))), 3)


class TestScores(unittest.TestCase):
    def test_score_reports_every_metric(self) -> None:
        truth = np.array([0.0, 0.5, 1.0])
        result = score(truth, truth.copy())
        self.assertAlmostEqual(result.r2, 1.0)
        self.assertAlmostEqual(result.mae, 0.0)
        self.assertEqual(result.n, 3)
        self.assertIn("rmse", result.as_dict())


class TestCrossValidation(unittest.TestCase):
    """The reported protocol: one equation, its form fixed, its weights refit per fold."""

    def setUp(self) -> None:
        self.columns, self.target, self.outer, self.inner = grid()
        self.library = build_library(("f1", "f2"), ("g1", "g2"), self.columns)
        result = search(self.library, self.target, max_terms=4, penalty=1.0, pool_size=30)
        self.equations = result.equations

    def path(self, penalty: float = 1.0) -> dict[int, CrossValidation]:
        return cross_validate_fixed_form(
            self.library, self.columns, self.target, self.outer, self.equations, penalty=penalty
        )

    def test_predicts_every_row(self) -> None:
        result = self.path()[2]
        self.assertEqual(result.predictions.shape, self.target.shape)
        self.assertTrue(np.all(np.isfinite(result.predictions)))

    def test_returns_every_size_it_was_given(self) -> None:
        self.assertEqual(sorted(self.path()), sorted(self.equations))

    def test_predictions_stay_inside_each_training_fold_range(self) -> None:
        """Tighter than the theoretical MCC range, and the reason `_clip_to_training` exists.

        A fold predicted outside the values its own training rows took is extrapolating, and
        one such fold once reached 89% of the corpus's total squared error on its own.
        """
        predictions = self.path()[3].predictions
        for _, train, test in leave_one_group_out(self.outer):
            self.assertGreaterEqual(predictions[test].min(), self.target[train].min())
            self.assertLessEqual(predictions[test].max(), self.target[train].max())

    def test_a_bounded_target_is_clipped_to_the_training_range(self) -> None:
        """The target grows with the group index, so the last group is held out above the rest."""
        result = self.path()[2]
        self.assertGreater(result.clipped, 0)

    def test_an_unbounded_target_is_never_clipped(self) -> None:
        """``bounds=None`` must hold in every protocol: extrapolating above training is the point."""
        unbounded = {size: replace(equation, bounds=None) for size, equation in self.equations.items()}
        result = cross_validate_fixed_form(self.library, self.columns, self.target, self.outer, unbounded, penalty=1.0)[
            2
        ]
        self.assertEqual(result.clipped, 0)
        last = self.outer == "d5"
        self.assertGreater(result.predictions[last].max(), self.target[~last].max())

    def test_unbounded_and_bounded_agree_where_the_clip_does_not_fire(self) -> None:
        unbounded = {size: replace(equation, bounds=None) for size, equation in self.equations.items()}
        free = cross_validate_fixed_form(self.library, self.columns, self.target, self.outer, unbounded, penalty=1.0)[2]
        clipped = self.path()[2]
        inside = np.ones_like(self.target, dtype=bool)
        for _, train, test in leave_one_group_out(self.outer):
            inside[test] = (free.predictions[test] >= self.target[train].min()) & (
                free.predictions[test] <= self.target[train].max()
            )
        self.assertTrue(inside.any())
        np.testing.assert_array_equal(free.predictions[inside], clipped.predictions[inside])

    def test_doubly_held_out_respects_the_bounds_too(self) -> None:
        unbounded = {size: replace(equation, bounds=None) for size, equation in self.equations.items()}
        free = cross_validate_doubly_held_out(
            self.library, self.columns, self.target, self.outer, self.inner, unbounded, penalty=1.0
        )[2]
        bounded = cross_validate_doubly_held_out(
            self.library, self.columns, self.target, self.outer, self.inner, self.equations, penalty=1.0
        )[2]
        self.assertEqual(free.clipped, 0)
        self.assertGreater(bounded.clipped, 0)

    def test_a_target_open_on_one_side_is_clipped_at_that_side_only(self) -> None:
        """A time cannot be negative but has no ceiling: floor at 0, nothing above."""
        floored = {size: replace(equation, bounds=(0.0, None)) for size, equation in self.equations.items()}
        result = cross_validate_fixed_form(self.library, self.columns, self.target, self.outer, floored, penalty=1.0)[2]
        self.assertTrue(np.all(result.predictions >= 0.0))
        last = self.outer == "d5"
        self.assertGreater(result.predictions[last].max(), self.target[~last].max())

    def test_the_clip_rule_by_kind_of_bound(self) -> None:
        from ml_meta_perf.validate import _clip_to_training  # pyright: ignore[reportPrivateUsage]

        training = np.array([2.0, 3.0, 4.0])
        wild = np.array([-9.0, 3.5, 99.0])
        np.testing.assert_array_equal(_clip_to_training(wild, training, None), wild)
        np.testing.assert_array_equal(_clip_to_training(wild, training, (0.0, None)), [0.0, 3.5, 99.0])
        np.testing.assert_array_equal(_clip_to_training(wild, training, (None, 50.0)), [-9.0, 3.5, 50.0])
        np.testing.assert_array_equal(_clip_to_training(wild, training, (-1.0, 100.0)), [2.0, 3.5, 4.0])

    def test_records_one_fold_score_per_group(self) -> None:
        result = self.path()[2]
        self.assertEqual(set(result.per_fold), set(np.unique(self.outer).tolist()))

    def test_the_form_is_identical_in_every_fold(self) -> None:
        """That is what "fixed form" means; only the weights may move."""
        result = self.path()[3]
        self.assertEqual(len({tuple(names) for names in result.selected}), 1)

    def test_fold_equations_keep_the_source_equation_bounds(self) -> None:
        unbounded = {size: replace(equation, bounds=None) for size, equation in self.equations.items()}
        result = cross_validate_fixed_form(self.library, self.columns, self.target, self.outer, unbounded, penalty=1.0)[
            2
        ]
        self.assertTrue(result.equations)
        for equation in result.equations.values():
            self.assertIsNone(equation.bounds)

    def test_dispersion_summarises_the_per_fold_scores(self) -> None:
        result = self.path()[2]
        dispersion = result.dispersion()
        folds = [fold.r2 for fold in result.per_fold.values()]
        self.assertAlmostEqual(dispersion["worst_fold_r2"], min(folds))
        self.assertEqual(dispersion["folds"], float(len(folds)))


class TestFoldSelections(unittest.TestCase):
    """Re-selection is kept for `term_stability` only, and cannot produce a score."""

    def setUp(self) -> None:
        self.columns, self.target, self.outer, self.inner = grid()
        self.library = build_library(("f1", "f2"), ("g1", "g2"), self.columns)

    def test_returns_one_term_list_per_fold(self) -> None:
        selections = fold_selections(self.library, self.target, self.outer, n_terms=2, penalty=1.0, pool_size=30)
        self.assertLessEqual(len(selections), len(np.unique(self.outer)))
        for names in selections:
            self.assertEqual(len(names), 2)

    def test_stability_counts_never_exceed_the_fold_count(self) -> None:
        selections = fold_selections(self.library, self.target, self.outer, n_terms=2, penalty=1.0, pool_size=30)
        stability = term_stability(selections)
        self.assertLessEqual(int(stability["folds"].max()), len(np.unique(self.outer)))  # pyright: ignore[reportArgumentType]
        self.assertLessEqual(float(stability["frequency"].max()), 1.0)  # pyright: ignore[reportArgumentType]

    def test_required_term_never_selected_is_exported_with_zero_frequency(self) -> None:
        stability = term_stability([["selected"]], required_terms=("selected", "never selected"))
        rows = {row["term"]: row for row in stability.iter_rows(named=True)}
        self.assertEqual(rows["never selected"]["folds"], 0)
        self.assertEqual(rows["never selected"]["frequency"], 0.0)


class TestDecisionReport(unittest.TestCase):
    """The go/no-go decision the equation supports."""

    def test_perfect_prediction_decides_perfectly(self) -> None:
        truth = np.array([0.1, 0.4, 0.8, 0.95])
        table = decision_report(truth, truth.copy(), thresholds=(0.5,))
        row = table.row(0, named=True)
        self.assertAlmostEqual(row["accuracy"], 1.0)
        self.assertAlmostEqual(row["precision"], 1.0)
        self.assertAlmostEqual(row["recall"], 1.0)

    def test_one_row_per_threshold(self) -> None:
        truth = np.linspace(0.0, 1.0, 20)
        self.assertEqual(decision_report(truth, truth.copy(), thresholds=(0.3, 0.6, 0.9)).height, 3)

    def test_majority_is_the_bar_to_clear(self) -> None:
        # 18 of 20 above the threshold: always saying yes scores 0.9.
        truth = np.concatenate([np.full(18, 0.9), np.full(2, 0.1)])
        row = decision_report(truth, truth.copy(), thresholds=(0.5,)).row(0, named=True)
        self.assertAlmostEqual(row["majority"], 0.9)

    def test_counts_actual_positives(self) -> None:
        truth = np.array([0.1, 0.9, 0.9])
        self.assertEqual(decision_report(truth, truth, thresholds=(0.5,))["n_positive"][0], 2)

    def test_degenerate_prediction_gives_nan_mcc_not_a_crash(self) -> None:
        truth = np.array([0.1, 0.9])
        table = decision_report(truth, np.zeros(2), thresholds=(0.5,))
        self.assertTrue(np.isnan(table["mcc"][0]))


class TestRanking(unittest.TestCase):
    def test_perfect_ranking_has_no_regret(self) -> None:
        target = np.array([0.1, 0.5, 0.9, 0.2, 0.7, 0.3])
        groups = np.array(["a", "a", "a", "b", "b", "b"])
        report = ranking_report(target, target.copy(), groups)
        self.assertEqual(report.height, 2)
        np.testing.assert_allclose(report["regret"].to_numpy(), [0.0, 0.0])
        np.testing.assert_allclose(report["spearman"].to_numpy(), [1.0, 1.0])

    def test_regret_is_the_gap_to_the_best(self) -> None:
        target = np.array([0.1, 0.9])
        report = ranking_report(target, np.array([1.0, 0.0]), np.array(["a", "a"]))
        self.assertAlmostEqual(float(report["regret"][0]), 0.8)


class TestPairedComparison(unittest.TestCase):
    """A difference of two means over twenty folds is not yet a measurement."""

    def test_a_consistent_win_is_significant(self) -> None:
        first = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.85, 0.75, 0.65])
        result = paired_comparison(first, first - 0.2)
        self.assertEqual(result.wins, 8)
        self.assertEqual(result.losses, 0)
        self.assertTrue(result.significant)
        self.assertAlmostEqual(result.mean, 0.2)

    def test_a_favourable_mean_from_a_few_large_wins_is_not(self) -> None:
        # The case this exists for: the mean says one side leads, the folds say otherwise.
        first = np.array([0.9, 0.9, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
        second = np.array([0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = paired_comparison(first, second)
        self.assertGreater(result.mean, 0.0)
        self.assertLess(result.wins, result.losses)
        self.assertFalse(result.significant)

    def test_lower_is_better_flips_the_sign(self) -> None:
        errors_a = np.array([0.1, 0.2, 0.3, 0.15])
        errors_b = errors_a + 0.05
        result = paired_comparison(errors_a, errors_b, lower_is_better=True)
        self.assertGreater(result.mean, 0.0)
        self.assertEqual(result.wins, 4)

    def test_ties_are_dropped_rather_than_counted(self) -> None:
        first = np.array([0.5, 0.5, 0.7, 0.3])
        second = np.array([0.5, 0.5, 0.6, 0.4])
        result = paired_comparison(first, second)
        self.assertEqual(result.n, 2)
        self.assertEqual((result.wins, result.losses), (1, 1))

    def test_identical_inputs_are_inconclusive(self) -> None:
        values = np.array([0.4, 0.5, 0.6])
        result = paired_comparison(values, values)
        self.assertEqual(result.n, 0)
        self.assertEqual(result.p_value, 1.0)
        self.assertFalse(result.significant)

    def test_mismatched_lengths_are_an_error(self) -> None:
        with self.assertRaises(ValueError):
            paired_comparison(np.array([0.1, 0.2]), np.array([0.1]))

    def test_the_result_is_reproducible(self) -> None:
        first = np.array([0.9, 0.4, 0.7, 0.2, 0.6])
        second = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        self.assertEqual(paired_comparison(first, second), paired_comparison(first, second))


class TestSignTest(unittest.TestCase):
    def test_all_wins_gives_the_smallest_available_p(self) -> None:
        wins, n, p = sign_test(np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
        self.assertEqual((wins, n), (5, 5))
        self.assertAlmostEqual(p, 2.0 / 32.0)

    def test_an_even_split_cannot_be_distinguished(self) -> None:
        _, _, p = sign_test(np.array([1.0, 1.0, -1.0, -1.0]))
        self.assertEqual(p, 1.0)

    def test_it_is_symmetric_in_direction(self) -> None:
        values = np.array([1.0, 1.0, 1.0, -1.0])
        self.assertAlmostEqual(sign_test(values)[2], sign_test(-values)[2])


if __name__ == "__main__":
    unittest.main()
