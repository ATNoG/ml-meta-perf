"""The descriptor search: the objective function, both optimizer backends, and
stability selection.

These run against the real candidate corpus (`assets/meta_dataset_all_descriptors.csv`),
at trial counts small enough for a commit hook -- the point is that the wiring is
correct, not that a short run reproduces a real recommendation (`test_experiment.py`
follows the same convention against the shipped corpus).

The two optimizer backends need the `search` extra (optuna, pyBlindOpt), which is not a
base dependency (see `pyproject.toml`) and is not installed by CI's `pip install .`
step -- only by `requirements.txt`, in the dev venv. Those tests skip rather than fail
when the extra is absent, so this file behaves the same way import-time optionality is
handled everywhere else in this module.
"""

import unittest
from pathlib import Path

import polars as pl

from ml_meta_perf.data import DATASET_COLUMN
from ml_meta_perf.descriptors import build_registry
from ml_meta_perf.feature_search import evaluate_subset, recommend_features
from ml_meta_perf.validate import baseline_group_mean, score

try:
    import optuna  # noqa: F401  # pyright: ignore[reportMissingImports]

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import pyBlindOpt  # noqa: F401  # pyright: ignore[reportMissingImports]

    HAS_PYBLINDOPT = True
except ImportError:
    HAS_PYBLINDOPT = False

CANDIDATE_PATH = Path("assets/meta_dataset_all_descriptors.csv")
HAS_CANDIDATE_CSV = CANDIDATE_PATH.is_file()


@unittest.skipUnless(HAS_CANDIDATE_CSV, "assets/meta_dataset_all_descriptors.csv not present")
class TestEvaluateSubset(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pl.read_csv(CANDIDATE_PATH)

    def test_empty_subset_with_no_dataset_features_matches_the_per_dataset_mean_baseline(self) -> None:
        from ml_meta_perf.data import groups, target

        result = evaluate_subset(frozenset(), self.frame, dataset_features=())
        self.assertEqual(result.n_features, 0)

        truth = target(self.frame)
        datasets = groups(self.frame, DATASET_COLUMN)
        expected = score(truth, baseline_group_mean(truth, datasets)).r2
        self.assertAlmostEqual(result.r2_loo_dataset, expected, places=9)

    def test_default_scores_in_the_e3_frame_not_model_alone(self) -> None:
        # With dataset features in the library by default, even an empty model-feature
        # set fits a real (E1-equivalent) equation rather than falling back to the naive
        # per-dataset-mean baseline -- and an E1-equivalent fit clears that baseline.
        from ml_meta_perf.data import groups, target

        truth = target(self.frame)
        datasets = groups(self.frame, DATASET_COLUMN)
        floor = score(truth, baseline_group_mean(truth, datasets)).r2

        result = evaluate_subset(frozenset(), self.frame)
        self.assertEqual(result.n_features, 0)
        self.assertGreater(result.r2_loo_dataset, floor)

    def test_a_small_subset_scores_between_the_target_bounds(self) -> None:
        result = evaluate_subset(frozenset({"Model Capability", "Processing Units Number"}), self.frame)
        self.assertEqual(result.n_features, 2)
        self.assertLessEqual(result.r2_loo_dataset, 1.0)

    def test_feature_order_does_not_affect_the_score(self) -> None:
        a = evaluate_subset(("Model Capability", "Processing Units Number"), self.frame)
        b = evaluate_subset(("Processing Units Number", "Model Capability"), self.frame)
        self.assertEqual(a.r2_loo_dataset, b.r2_loo_dataset)
        self.assertEqual(a.features, b.features)  # both sorted the same way

    def test_a_model_feature_useless_alone_can_still_help_via_a_cross_term(self) -> None:
        # The whole point of scoring in the E3 frame: a model feature with essentially no
        # own-signal can still lift the score through a dataset x model interaction term,
        # which model-features-alone scoring (dataset_features=()) cannot see at all.
        alone = evaluate_subset(frozenset({"Model Capability"}), self.frame, dataset_features=())
        combined = evaluate_subset(frozenset({"Model Capability"}), self.frame)
        self.assertGreaterEqual(combined.r2_loo_dataset, alone.r2_loo_dataset)


#: A handful of admissible candidates, not all 61 -- these tests check wiring (exclusion
#: cleanliness, row shapes, frequency bounds), not search quality, and
#: `dataset_features=()` on top keeps every trial down to the cheap model-alone path
#: (`evaluate_subset`'s docstring) rather than the full E3-frame library real searches
#: use. Real-scale timing lives in `TestEvaluateSubset`, on individually chosen subsets.
_SMOKE_CANDIDATES = (
    "Model Capability", "Processing Units Number", "chatgpt__l1_coefficient",
    "chatgpt__l2_coefficient", "chatgpt__learning_rate", "claude1__ensemble_size",
)


@unittest.skipUnless(HAS_CANDIDATE_CSV, "assets/meta_dataset_all_descriptors.csv not present")
@unittest.skipUnless(HAS_OPTUNA, "optuna not installed (pip install ml-meta-perf[search])")
class TestOptunaSearch(unittest.TestCase):
    def test_front_is_free_of_exclusion_group_duplicates(self) -> None:
        from ml_meta_perf.descriptors import EXCLUSION_GROUPS
        from ml_meta_perf.feature_search import optuna_search

        frame = pl.read_csv(CANDIDATE_PATH)
        registry = build_registry(frame, candidates=_SMOKE_CANDIDATES)
        front = optuna_search(registry, frame, dataset_features=(), n_trials=8, seed=0)
        self.assertGreater(front.height, 0)

        for row in front.iter_rows(named=True):
            selected = {name.strip() for name in row["features"].split(",") if name.strip()}
            for members in EXCLUSION_GROUPS.values():
                self.assertLessEqual(len(selected & set(members)), 1)

    def test_front_sizes_match_feature_counts(self) -> None:
        from ml_meta_perf.feature_search import optuna_search

        frame = pl.read_csv(CANDIDATE_PATH)
        registry = build_registry(frame, candidates=_SMOKE_CANDIDATES)
        front = optuna_search(registry, frame, dataset_features=(), n_trials=8, seed=1)
        for row in front.iter_rows(named=True):
            selected = [name for name in row["features"].split(",") if name.strip()]
            self.assertEqual(len(selected), row["n_features"])


@unittest.skipUnless(HAS_CANDIDATE_CSV, "assets/meta_dataset_all_descriptors.csv not present")
@unittest.skipUnless(HAS_PYBLINDOPT, "pyBlindOpt not installed (pip install ml-meta-perf[search])")
class TestPyBlindOptSearch(unittest.TestCase):
    def test_sweep_returns_one_row_per_lambda(self) -> None:
        from ml_meta_perf.feature_search import pyblindopt_search

        frame = pl.read_csv(CANDIDATE_PATH)
        registry = build_registry(frame, candidates=_SMOKE_CANDIDATES)
        front = pyblindopt_search(
            registry, frame, dataset_features=(), lambdas=(0.0, 0.05), n_pop=6, n_iter=2, seed=0
        )
        self.assertEqual(front.height, 2)
        self.assertTrue(set(front["lambda"].to_list()) == {0.0, 0.05})


@unittest.skipUnless(HAS_CANDIDATE_CSV, "assets/meta_dataset_all_descriptors.csv not present")
@unittest.skipUnless(HAS_OPTUNA, "optuna not installed (pip install ml-meta-perf[search])")
class TestStabilitySelection(unittest.TestCase):
    def test_frequencies_are_bounded_and_recommendation_respects_threshold(self) -> None:
        from ml_meta_perf.feature_search import stability_selection

        frame = pl.read_csv(CANDIDATE_PATH)
        registry = build_registry(frame, candidates=_SMOKE_CANDIDATES)
        frequency = stability_selection(
            registry, frame, dataset_features=(), repeats=3, trials_per_repeat=6, hold_out_fraction=0.3, seed=0
        )
        self.assertEqual(set(frequency["feature"].to_list()), set(registry.admissible))
        self.assertTrue(((frequency["frequency"] >= 0.0) & (frequency["frequency"] <= 1.0)).all())

        recommended = recommend_features(frequency, threshold=0.7)
        survivors = set(frequency.filter(pl.col("frequency") >= 0.7)["feature"].to_list())
        self.assertEqual(set(recommended), survivors)


class TestRecommendFeatures(unittest.TestCase):
    def test_empty_frequency_table_recommends_nothing(self) -> None:
        empty = pl.DataFrame(schema={"feature": pl.Utf8, "selected": pl.Int64, "frequency": pl.Float64})
        self.assertEqual(recommend_features(empty), ())

    def test_threshold_is_inclusive(self) -> None:
        frequency = pl.DataFrame(
            {"feature": ["a", "b"], "selected": [7, 6], "frequency": [0.7, 0.6]}
        )
        self.assertEqual(recommend_features(frequency, threshold=0.7), ("a",))


@unittest.skipUnless(HAS_CANDIDATE_CSV, "assets/meta_dataset_all_descriptors.csv not present")
class TestGreedyForwardSearch(unittest.TestCase):
    def test_history_starts_at_the_dataset_only_equation(self) -> None:
        from dataclasses import replace

        from ml_meta_perf.feature_search import GREEDY_SEARCH_CONFIG, greedy_forward_search

        frame = pl.read_csv(CANDIDATE_PATH)
        registry = build_registry(frame, candidates=("Model Capability", "Training Operations"))
        # A tiny dataset side and a tight step budget: this checks the search's
        # bookkeeping (history shape, monotone score, early stop), not its quality.
        tiny_config = replace(GREEDY_SEARCH_CONFIG, pool_size=20, max_terms=6, headline_terms=6)
        history = greedy_forward_search(
            registry, frame, dataset_features=("nr_attr", "nr_class"), config=tiny_config, max_features=2,
        )
        self.assertEqual(history[0].added, None)
        self.assertEqual(history[0].features, ())

    def test_each_step_adds_exactly_one_new_feature(self) -> None:
        from dataclasses import replace

        from ml_meta_perf.feature_search import GREEDY_SEARCH_CONFIG, greedy_forward_search

        frame = pl.read_csv(CANDIDATE_PATH)
        registry = build_registry(frame, candidates=("Model Capability", "Training Operations"))
        tiny_config = replace(GREEDY_SEARCH_CONFIG, pool_size=20, max_terms=6, headline_terms=6)
        history = greedy_forward_search(
            registry, frame, dataset_features=("nr_attr", "nr_class"), config=tiny_config,
            max_features=2, min_improvement=-1.0,  # force both steps regardless of score
        )
        import itertools

        for previous, current in itertools.pairwise(history):
            added = set(current.features) - set(previous.features)
            self.assertEqual(added, {current.added})

    def test_stops_early_when_nothing_improves_enough(self) -> None:
        from dataclasses import replace

        from ml_meta_perf.feature_search import GREEDY_SEARCH_CONFIG, greedy_forward_search

        frame = pl.read_csv(CANDIDATE_PATH)
        registry = build_registry(frame, candidates=("Model Capability", "Training Operations"))
        tiny_config = replace(GREEDY_SEARCH_CONFIG, pool_size=20, max_terms=6, headline_terms=6)
        history = greedy_forward_search(
            registry, frame, dataset_features=("nr_attr", "nr_class"), config=tiny_config,
            max_features=2, min_improvement=10.0,  # nothing can improve by this much
        )
        self.assertEqual(len(history), 1)


if __name__ == "__main__":
    unittest.main()
