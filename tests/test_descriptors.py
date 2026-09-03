"""The candidate registry: exclusion groups, admissibility, and conflict resolution."""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.descriptors import (
    CANDIDATE_FEATURES,
    EXCLUSION_GROUPS,
    INADMISSIBLE,
    build_registry,
    compute_strengths,
)


class TestRegistryTables(unittest.TestCase):
    """The hand-transcribed tables are internally consistent with each other."""

    def test_every_exclusion_group_member_is_a_known_candidate(self) -> None:
        known = set(CANDIDATE_FEATURES)
        for group, members in EXCLUSION_GROUPS.items():
            for member in members:
                self.assertIn(member, known, f"{member!r} in group {group!r} is not in CANDIDATE_FEATURES")

    def test_every_inadmissible_entry_is_a_known_candidate(self) -> None:
        known = set(CANDIDATE_FEATURES)
        for name in INADMISSIBLE:
            self.assertIn(name, known)

    def test_candidate_features_has_no_duplicates(self) -> None:
        self.assertEqual(len(CANDIDATE_FEATURES), len(set(CANDIDATE_FEATURES)))

    def test_a_group_with_a_real_order_has_no_nominal_sign_encoding(self) -> None:
        # The two descriptors this session's review flagged as nominal-choice-as-sign
        # must not be admissible defaults; they are documented exclusions, not just
        # absent by omission.
        self.assertIn("perplexity__knn_metric_signal", INADMISSIBLE)
        self.assertIn("perplexity__knn_weights_signal", INADMISSIBLE)


def _frame(columns: dict[str, list[float]], datasets: list[str]) -> pl.DataFrame:
    data: dict[str, object] = dict(columns)
    data["Dataset"] = datasets
    return pl.DataFrame(data)


class TestBuildRegistry(unittest.TestCase):
    def test_inadmissible_columns_are_dropped_with_their_documented_reason(self) -> None:
        frame = _frame(
            {name: [1.0, 2.0, 3.0, 4.0] for name in ("Model Capability", "perplexity__knn_metric_signal")},
            datasets=["a", "a", "b", "b"],
        )
        registry = build_registry(frame, candidates=("Model Capability", "perplexity__knn_metric_signal"))
        self.assertIn("Model Capability", registry.admissible)
        self.assertNotIn("perplexity__knn_metric_signal", registry.admissible)
        self.assertEqual(
            registry.dropped["perplexity__knn_metric_signal"], INADMISSIBLE["perplexity__knn_metric_signal"]
        )

    def test_near_constant_columns_are_dropped_even_when_not_in_inadmissible(self) -> None:
        frame = _frame(
            {"Model Capability": [5.0, 5.0, 5.0, 5.0], "Processing Units Number": [1.0, 2.0, 3.0, 4.0]},
            datasets=["a", "a", "b", "b"],
        )
        registry = build_registry(frame, candidates=("Model Capability", "Processing Units Number"))
        self.assertNotIn("Model Capability", registry.admissible)
        self.assertIn("Processing Units Number", registry.admissible)
        self.assertIn("near-constant", registry.dropped["Model Capability"])

    def test_missing_columns_are_dropped_rather_than_raising(self) -> None:
        frame = _frame({"Model Capability": [1.0, 2.0, 3.0, 4.0]}, datasets=["a", "a", "b", "b"])
        registry = build_registry(frame, candidates=("Model Capability", "chatgpt__component_count"))
        self.assertIn("Model Capability", registry.admissible)
        self.assertNotIn("chatgpt__component_count", registry.admissible)


class TestConflictResolution(unittest.TestCase):
    def setUp(self) -> None:
        columns = {name: np.linspace(1.0, 10.0, 8).tolist() for name in (
            "chatgpt__l2_coefficient", "perplexity__reg_l2_log",
            "chatgpt__covariance_shrinkage", "chatgpt__covariance_regularization",
            "chatgpt__sequential_depth", "claude1__capacity", "perplexity__capacity_depth",
        )}
        frame = _frame(columns, datasets=["a", "a", "b", "b", "c", "c", "d", "d"])
        self.registry = build_registry(frame, candidates=tuple(columns))

    def test_two_members_of_the_same_group_never_both_survive(self) -> None:
        resolved = self.registry.resolve_conflicts(
            frozenset({"chatgpt__l2_coefficient", "perplexity__reg_l2_log"})
        )
        self.assertEqual(len(resolved), 1)

    def test_resolution_is_deterministic_by_candidate_order(self) -> None:
        # chatgpt__l2_coefficient precedes perplexity__reg_l2_log in CANDIDATE_FEATURES
        # (ChatGPT's block comes before Perplexity's), so it is the one that survives.
        resolved = self.registry.resolve_conflicts(
            frozenset({"perplexity__reg_l2_log", "chatgpt__l2_coefficient"})
        )
        self.assertEqual(resolved, frozenset({"chatgpt__l2_coefficient"}))

    def test_a_member_of_two_groups_blocks_both_groups_at_once(self) -> None:
        # perplexity__reg_l2_log sits in both E16 (L2) and E18 (covariance); selecting it
        # alongside a covariance-only descriptor must still collapse to one survivor.
        resolved = self.registry.resolve_conflicts(
            frozenset({"perplexity__reg_l2_log", "chatgpt__covariance_shrinkage"})
        )
        self.assertEqual(len(resolved), 1)

    def test_a_three_way_multi_group_member_still_yields_an_independent_set(self) -> None:
        # claude1__capacity spans depth (E09), width (E10) and aggregate capacity (E11).
        resolved = self.registry.resolve_conflicts(
            frozenset({"claude1__capacity", "chatgpt__sequential_depth", "perplexity__capacity_depth"})
        )
        self.assertEqual(len(resolved), 1)

    def test_unrelated_candidates_all_survive(self) -> None:
        resolved = self.registry.resolve_conflicts(
            frozenset({"chatgpt__l2_coefficient", "chatgpt__sequential_depth"})
        )
        self.assertEqual(resolved, frozenset({"chatgpt__l2_coefficient", "chatgpt__sequential_depth"}))


class TestComputeStrengths(unittest.TestCase):
    def test_strength_is_the_stronger_of_pearson_and_spearman(self) -> None:
        # A monotone, curved (non-linear) relationship: Spearman should dominate.
        linear = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        curved = [x**3 for x in linear]
        frame = pl.DataFrame({"a": linear, "b": curved, "MCC": linear})
        strengths = compute_strengths(frame, ("a", "b"))
        self.assertAlmostEqual(strengths["a"], 1.0, places=6)
        self.assertAlmostEqual(strengths["b"], 1.0, places=6)  # perfectly monotone, imperfectly linear


class TestReduceToRepresentatives(unittest.TestCase):
    def setUp(self) -> None:
        columns = {
            name: np.linspace(1.0, 10.0, 8).tolist()
            for name in ("chatgpt__l2_coefficient", "perplexity__reg_l2_log", "chatgpt__sequential_depth")
        }
        self.frame = _frame(columns, datasets=["a"] * 8)
        self.registry = build_registry(self.frame, candidates=tuple(columns))

    def test_the_stronger_correlate_survives_its_group(self) -> None:
        strengths = {
            "chatgpt__l2_coefficient": 0.9,
            "perplexity__reg_l2_log": 0.3,
            "chatgpt__sequential_depth": 0.1,
        }
        reduced = self.registry.reduce_to_representatives(strengths)
        self.assertIn("chatgpt__l2_coefficient", reduced.admissible)
        self.assertNotIn("perplexity__reg_l2_log", reduced.admissible)
        self.assertIn("chatgpt__l2_coefficient", reduced.dropped["perplexity__reg_l2_log"])

    def test_a_weaker_correlate_can_still_flip_the_winner(self) -> None:
        strengths = {
            "chatgpt__l2_coefficient": 0.2,
            "perplexity__reg_l2_log": 0.8,
            "chatgpt__sequential_depth": 0.1,
        }
        reduced = self.registry.reduce_to_representatives(strengths)
        self.assertIn("perplexity__reg_l2_log", reduced.admissible)
        self.assertNotIn("chatgpt__l2_coefficient", reduced.admissible)

    def test_ungrouped_candidates_are_unaffected(self) -> None:
        strengths = {"chatgpt__l2_coefficient": 0.9, "perplexity__reg_l2_log": 0.1, "chatgpt__sequential_depth": 0.5}
        reduced = self.registry.reduce_to_representatives(strengths)
        self.assertIn("chatgpt__sequential_depth", reduced.admissible)

    def test_result_is_still_exclusion_clean(self) -> None:
        strengths = {"chatgpt__l2_coefficient": 0.9, "perplexity__reg_l2_log": 0.1, "chatgpt__sequential_depth": 0.5}
        reduced = self.registry.reduce_to_representatives(strengths)
        for members in EXCLUSION_GROUPS.values():
            self.assertLessEqual(len(set(reduced.admissible) & set(members)), 1)


class TestCorrelationConflicts(unittest.TestCase):
    def test_two_unrelated_but_numerically_identical_columns_become_conflicting(self) -> None:
        # Two descriptors from *different* concepts (no shared EXCLUSION_GROUPS entry)
        # that happen to be the same numbers on this corpus.
        values = np.linspace(1.0, 10.0, 8).tolist()
        frame = _frame(
            {"chatgpt__attention_heads": values, "perplexity__knn_k": values, "chatgpt__sequential_depth": [1.0] * 8},
            datasets=["a"] * 8,
        )
        registry = build_registry(
            frame, candidates=("chatgpt__attention_heads", "perplexity__knn_k", "chatgpt__sequential_depth")
        )
        self.assertNotIn("chatgpt__attention_heads", registry.conflicts.get("perplexity__knn_k", frozenset()))

        extended = registry.with_correlation_conflicts(frame, threshold=0.9)
        self.assertIn("chatgpt__attention_heads", extended.conflicts["perplexity__knn_k"])
        self.assertIn("perplexity__knn_k", extended.conflicts["chatgpt__attention_heads"])
        # Unrelated to both, and not correlated with either -- untouched.
        self.assertNotIn("chatgpt__sequential_depth", extended.conflicts["perplexity__knn_k"])

    def test_named_exclusion_groups_survive_the_extension(self) -> None:
        frame = _frame(
            {
                "chatgpt__l2_coefficient": np.linspace(1.0, 10.0, 8).tolist(),
                "chatgpt__sequential_depth": [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0],
            },
            datasets=["a"] * 8,
        )
        registry = build_registry(frame, candidates=("chatgpt__l2_coefficient", "chatgpt__sequential_depth"))
        extended = registry.with_correlation_conflicts(frame, threshold=0.9)
        # E16 has only one member among these candidates, so no named conflict here --
        # this just checks the named graph isn't lost by the extension.
        self.assertEqual(extended.conflicts.keys(), registry.conflicts.keys())


if __name__ == "__main__":
    unittest.main()
