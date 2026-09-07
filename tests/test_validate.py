"""Validation protocols, baselines and the leakage they are designed to avoid."""

import itertools
import unittest

import numpy as np
import polars as pl

from ml_meta_perf.analysis import feature_reach, grammar_ceiling, redundancy_groups, saturated_fit, screen
from ml_meta_perf.fit import fit
from ml_meta_perf.stats import mae, r2_score
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import (
    CrossValidation,
    additive_oracle,
    baseline_group_centre,
    baseline_group_mean,
    cross_validate_fixed_form,
    decision_report,
    fold_selections,
    interaction_oracle,
    leave_one_group_out,
    oracle_ladder,
    paired_comparison,
    random_kfold_groups,
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

    def test_random_folds_have_the_right_shape(self) -> None:
        folds = random_kfold_groups(50, folds=5, seed=1)
        self.assertEqual(folds.shape, (50,))
        self.assertLessEqual(len(set(folds.tolist())), 5)

    def test_random_folds_are_reproducible(self) -> None:
        np.testing.assert_array_equal(random_kfold_groups(30, seed=9), random_kfold_groups(30, seed=9))


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
        result = fit(self.library, self.target, max_terms=4, penalty=1.0, pool_size=30)
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

    def test_records_one_fold_score_per_group(self) -> None:
        result = self.path()[2]
        self.assertEqual(set(result.per_fold), set(np.unique(self.outer).tolist()))

    def test_the_form_is_identical_in_every_fold(self) -> None:
        """That is what "fixed form" means; only the weights may move."""
        result = self.path()[3]
        self.assertEqual(len({tuple(names) for names in result.selected}), 1)

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


class TestBaselines(unittest.TestCase):
    def setUp(self) -> None:
        _, self.target, self.outer, self.inner = grid()

    def test_global_mean_is_constant_within_a_fold(self) -> None:
        predictions = baseline_group_mean(self.target, self.outer)
        for label in np.unique(self.outer):
            self.assertEqual(len(set(np.round(predictions[self.outer == label], 12))), 1)

    def test_conditioning_on_the_inner_group_helps(self) -> None:
        plain = baseline_group_mean(self.target, self.outer)
        conditioned = baseline_group_mean(self.target, self.outer, self.inner)
        self.assertLess(float(np.abs(self.target - conditioned).mean()), float(np.abs(self.target - plain).mean()))

    def test_additive_oracle_beats_either_group_alone(self) -> None:
        oracle = additive_oracle(self.target, self.outer, self.inner)
        self.assertGreater(
            float(np.corrcoef(oracle, self.target)[0, 1]),
            float(np.corrcoef(baseline_group_mean(self.target, self.outer), self.target)[0, 1]),
        )

    def test_baselines_stay_inside_the_mcc_range(self) -> None:
        for predictions in (
            baseline_group_mean(self.target, self.outer),
            additive_oracle(self.target, self.outer, self.inner),
        ):
            self.assertGreaterEqual(predictions.min(), -1.0)
            self.assertLessEqual(predictions.max(), 1.0)


class TestInteractionOracle(unittest.TestCase):
    def setUp(self) -> None:
        _, self.target, self.outer, self.inner = grid()

    def test_rank_zero_equals_the_additive_oracle(self) -> None:
        np.testing.assert_allclose(
            interaction_oracle(self.target, self.outer, self.inner, 0),
            additive_oracle(self.target, self.outer, self.inner),
            atol=1e-9,
        )

    def test_more_components_never_fit_worse(self) -> None:
        from ml_meta_perf.stats import r2_score

        scores = [
            r2_score(self.target, interaction_oracle(self.target, self.outer, self.inner, rank)) for rank in range(5)
        ]
        for earlier, later in itertools.pairwise(scores):
            self.assertGreaterEqual(later, earlier - 1e-9)

    def test_full_rank_reproduces_every_cell(self) -> None:
        from ml_meta_perf.stats import r2_score

        full = min(np.unique(self.outer).shape[0], np.unique(self.inner).shape[0])
        value = r2_score(self.target, interaction_oracle(self.target, self.outer, self.inner, full))
        self.assertGreater(value, 0.999)

    def test_stays_inside_the_mcc_range(self) -> None:
        prediction = interaction_oracle(self.target, self.outer, self.inner, 2)
        self.assertGreaterEqual(prediction.min(), -1.0)
        self.assertLessEqual(prediction.max(), 1.0)

    def test_handles_missing_cells(self) -> None:
        # 24 of the 500 dataset-by-model cells are absent in the real meta-dataset.
        mask = np.ones(self.target.shape[0], dtype=bool)
        mask[3] = False
        prediction = interaction_oracle(self.target[mask], self.outer[mask], self.inner[mask], 2)
        self.assertTrue(np.all(np.isfinite(prediction)))


class TestOracleLadder(unittest.TestCase):
    def setUp(self) -> None:
        _, self.target, self.outer, self.inner = grid()

    def test_one_row_per_rank(self) -> None:
        table = oracle_ladder(self.target, self.outer, self.inner, ranks=(0, 1, 2))
        self.assertEqual(table["interaction_rank"].to_list(), [0, 1, 2])

    def test_gain_is_undefined_for_the_first_rung(self) -> None:
        table = oracle_ladder(self.target, self.outer, self.inner, ranks=(0, 1))
        self.assertNotEqual(table["gain"][0], table["gain"][0])  # NaN

    def test_r2_is_non_decreasing(self) -> None:
        scores = oracle_ladder(self.target, self.outer, self.inner, ranks=(0, 1, 2, 3))["r2"].to_numpy()
        self.assertTrue((np.diff(scores) >= -1e-9).all())


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


class TestAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.columns, self.target, self.outer, self.inner = grid()
        self.library = build_library(("f1", "f2"), ("g1", "g2"), self.columns)

    def test_screen_returns_one_row_per_term(self) -> None:
        table = screen(self.library, self.target)
        self.assertEqual(table.height, len(self.library))
        self.assertIn("pearson", table.columns)
        self.assertIn("spearman", table.columns)

    def test_screen_is_sorted_by_strength(self) -> None:
        strengths = screen(self.library, self.target)["strength"].to_numpy()
        self.assertTrue(np.all(np.diff(strengths) <= 1e-12))

    def test_within_group_columns_appear_when_groups_are_given(self) -> None:
        table = screen(self.library, self.target, self.outer)
        self.assertIn("within_pearson", table.columns)

    def test_within_group_screening_demotes_a_pure_group_feature(self) -> None:
        # f1 is constant inside a dataset, so once the group mean is removed it has no
        # variance left to correlate with -- which is the whole point of the column.
        table = screen(self.library, self.target, self.outer)
        row = table.filter(pl.col("term") == "f1")
        self.assertLess(abs(float(row["within_pearson"][0])), 1e-9)

    def test_redundancy_groups_cluster_duplicates(self) -> None:
        clusters = redundancy_groups(self.library, threshold=0.999)
        for cluster in clusters:
            self.assertGreater(len(cluster), 1)


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


class TestGroupCentreBaselines(unittest.TestCase):
    """Both centres, because the metrics disagree about which baseline is honest."""

    def setUp(self) -> None:
        rng = np.random.default_rng(11)
        self.outer = np.repeat([f"d{i}" for i in range(6)], 8)
        self.inner = np.tile([f"m{i}" for i in range(8)], 6)
        self.truth = rng.uniform(0.0, 1.0, size=48)

    def test_median_beats_mean_on_absolute_error(self) -> None:
        """The reason the median rows exist: MAE is minimised by the median.

        A skewed target makes the gap visible. Reporting the equation's MAE only against a
        mean baseline compares it with a predictor not minimising the metric being reported.
        """
        skewed = np.concatenate([np.full(40, 0.9), np.linspace(0.0, 0.2, 8)])
        mean = baseline_group_centre(skewed, self.outer, self.inner, centre="mean")
        median = baseline_group_centre(skewed, self.outer, self.inner, centre="median")
        self.assertLess(mae(skewed, median), mae(skewed, mean))

    def test_mean_beats_median_on_squared_error(self) -> None:
        """And the converse, which is why both are kept rather than one replacing the other."""
        mean = baseline_group_centre(self.truth, self.outer, centre="mean")
        median = baseline_group_centre(self.truth, self.outer, centre="median")
        self.assertGreater(r2_score(self.truth, mean), r2_score(self.truth, median))

    def test_the_mean_wrapper_is_the_centre_at_mean(self) -> None:
        np.testing.assert_allclose(
            baseline_group_mean(self.truth, self.outer, self.inner),
            baseline_group_centre(self.truth, self.outer, self.inner, centre="mean"),
        )

    def test_no_row_sees_its_own_value(self) -> None:
        """The leave-one-group-out loop is what keeps the baseline honest."""
        for centre in ("mean", "median"):
            prediction = baseline_group_centre(self.truth, self.outer, centre=centre)
            for label in np.unique(self.outer):
                mask = self.outer == label
                expected = np.mean if centre == "mean" else np.median
                self.assertAlmostEqual(float(prediction[mask][0]), float(expected(self.truth[~mask])))


class TestGrammarReach(unittest.TestCase):
    """The heuristic ceiling the vocabulary implies, before any search runs."""

    def setUp(self) -> None:
        from ml_meta_perf.data import DATASET_FEATURES, MODEL_FEATURES, columns_as_arrays, load
        from ml_meta_perf.data import target as load_target
        from ml_meta_perf.experiment import DEFAULT_E3

        frame = load()
        self.features = DATASET_FEATURES + MODEL_FEATURES
        columns = columns_as_arrays(frame, self.features)
        self.truth = load_target(frame)
        self.library = build_library(
            DATASET_FEATURES,
            MODEL_FEATURES,
            columns,
            max_arity=DEFAULT_E3.max_arity,
            max_abs_zscore=DEFAULT_E3.max_abs_zscore,
        )

    def test_a_feature_is_never_worse_after_the_grammar_than_before(self) -> None:
        """`gain` measures what the transforms unlock, so it cannot be negative.

        The raw column is itself an admissible term, so the best single-feature term is at
        worst the raw one.
        """
        table = feature_reach(self.library, self.truth, self.features)
        self.assertTrue((table["gain"].to_numpy() >= -1e-12).all())
        self.assertTrue((table["r2_best"].to_numpy() >= table["r2_raw"].to_numpy() - 1e-12).all())

    def test_only_single_feature_terms_are_credited(self) -> None:
        """A product would otherwise be counted twice, once under each of its features."""
        table = feature_reach(self.library, self.truth, self.features)
        by_name = {term.name: term for term in self.library.terms}
        for row in table.to_dicts():
            if row["best_term"]:
                self.assertEqual(set(by_name[row["best_term"]].features), {row["feature"]})

    def test_the_ladder_is_monotone(self) -> None:
        """Each rung is a superset of the one before it, so R2 cannot fall."""
        ladder = grammar_ceiling(self.library, self.truth, self.features)
        self.assertLessEqual(ladder["r2_raw_additive"], ladder["r2_best_per_feature"] + 1e-9)
        self.assertLessEqual(ladder["r2_best_per_feature"], ladder["r2_all_single_feature"] + 1e-9)

    def test_it_is_a_heuristic_and_not_a_bound(self) -> None:
        """The published equation passes it, which is the reading the report gives.

        `r2_all_single_feature` bounds a sum of per-feature functions. E3's cross-feature
        terms are not that, so exceeding it is expected -- and is the independent route to
        the same conclusion the additive oracle reaches.
        """
        from ml_meta_perf.data import load
        from ml_meta_perf.experiment import run_e3

        ladder = grammar_ceiling(self.library, self.truth, self.features)
        fitted = float(run_e3(load()).in_sample["r2"])
        self.assertGreater(fitted, ladder["r2_all_single_feature"])


class TestSaturatedFit(unittest.TestCase):
    """The control for the selection stage: what every term at once actually does.

    Chapter 3 opens on this comparison, and it opened on a stale copy of it for months. The
    tests pin the two properties the argument rests on rather than the values themselves.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from ml_meta_perf.data import DATASET_COLUMN, DATASET_FEATURES, columns_as_arrays, groups, load
        from ml_meta_perf.data import target as load_target
        from ml_meta_perf.experiment import DEFAULT_E3, EQUATION_MODEL_FEATURES

        frame = load()
        columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
        cls.library = build_library(
            DATASET_FEATURES,
            EQUATION_MODEL_FEATURES,
            columns,
            max_arity=DEFAULT_E3.max_arity,
            max_abs_zscore=DEFAULT_E3.max_abs_zscore,
        )
        cls.result = saturated_fit(cls.library, load_target(frame), groups(frame, DATASET_COLUMN))

    def test_counts_the_whole_library(self) -> None:
        self.assertEqual(int(self.result["terms"]), len(self.library))

    def test_it_fits_better_than_it_transfers(self) -> None:
        """The finding: an unconstrained fit over a design this wide describes and does not
        generalise. If this ever inverted, the selection stage would need justifying again."""
        self.assertGreater(self.result["r2_in_sample"], 0.5)
        self.assertLess(self.result["r2_loo_dataset_clipped"], 0.0)

    def test_the_clip_is_what_stops_it_running_away(self) -> None:
        """Both bounds are reported because the gap between them is the point."""
        self.assertLess(self.result["r2_loo_dataset_unclipped"], self.result["r2_loo_dataset_clipped"])

    def test_it_transfers_worse_than_the_published_equation(self) -> None:
        from ml_meta_perf.data import load
        from ml_meta_perf.experiment import run_e3

        published = float(run_e3(load()).cross_validated["loo_dataset"]["r2"])
        self.assertGreater(published, self.result["r2_loo_dataset_clipped"])
