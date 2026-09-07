"""Beam policies, term seeding, and the guarantee that none of it moves the published result.

The load-bearing test in this file is `TestVanillaIsUnchanged`. Everything else can be wrong
in an interesting way; that one being wrong means a refactor silently changed the equation the
study reports, which is the failure that would be hardest to notice and worst to ship.
"""

import unittest
from typing import ClassVar

import numpy as np

from ml_meta_perf.beam import (
    BY_NAME,
    CATALOGUE,
    VANILLA,
    BeamPolicy,
    cap_per_parent,
    determinantal_select,
    prune,
)


class TestPolicy(unittest.TestCase):
    def test_the_default_is_the_incumbent(self) -> None:
        self.assertTrue(BeamPolicy().is_vanilla())
        self.assertTrue(VANILLA.is_vanilla())

    def test_every_catalogued_variant_says_it_is_not_vanilla(self) -> None:
        """A variant that reports itself vanilla would take the fast path and measure nothing."""
        for policy in CATALOGUE:
            with self.subTest(policy=policy.name):
                self.assertEqual(policy.is_vanilla(), policy.name == "vanilla")

    def test_names_are_unique(self) -> None:
        self.assertEqual(len(BY_NAME), len(CATALOGUE))

    def test_the_tighter_threshold_wins_when_both_are_set(self) -> None:
        """Combining the two pruning forms can only narrow the beam, so a swept grid over both
        cannot contain a point looser than either alone."""
        policy = BeamPolicy(absolute_prune=1.0, relative_prune=0.001)
        self.assertEqual(policy.threshold(10.0, 100.0), 10.1)

    def test_no_threshold_without_pruning(self) -> None:
        self.assertIsNone(VANILLA.threshold(10.0, 100.0))


class TestPrune(unittest.TestCase):
    def test_keeps_everything_when_unconfigured(self) -> None:
        scores = np.array([1.0, 5.0, 9.0])
        self.assertTrue(prune(scores, VANILLA, 100.0).all())

    def test_drops_candidates_beyond_the_threshold(self) -> None:
        scores = np.array([1.0, 1.5, 9.0])
        kept = prune(scores, BeamPolicy(absolute_prune=1.0), 100.0)
        self.assertEqual(kept.tolist(), [True, True, False])

    def test_relative_pruning_scales_with_the_target(self) -> None:
        scores = np.array([1.0, 3.0])
        self.assertEqual(prune(scores, BeamPolicy(relative_prune=0.01), 100.0).tolist(), [True, False])
        self.assertEqual(prune(scores, BeamPolicy(relative_prune=0.05), 100.0).tolist(), [True, True])

    def test_at_least_one_candidate_always_survives(self) -> None:
        """A threshold that excluded everything would end the search a step early and return a
        shorter equation than asked for, silently."""
        scores = np.array([1.0, 2.0])
        self.assertEqual(int(prune(scores, BeamPolicy(absolute_prune=-5.0), 100.0).sum()), 1)

    def test_handles_negative_objectives(self) -> None:
        """`Subset.rss` is penalised and ridge shrinkage can drive it negative, which is why
        both pruning forms are differences rather than ratios."""
        scores = np.array([-3.0, -2.5, 4.0])
        self.assertEqual(prune(scores, BeamPolicy(absolute_prune=1.0), 100.0).tolist(), [True, True, False])

    def test_an_empty_step_returns_an_empty_mask(self) -> None:
        self.assertEqual(prune(np.zeros(0), BeamPolicy(absolute_prune=1.0), 1.0).shape, (0,))


class TestCapPerParent(unittest.TestCase):
    def test_unlimited_is_the_textbook_beam(self) -> None:
        self.assertEqual(cap_per_parent([0, 0, 0, 0], 3, None), [0, 1, 2])

    def test_caps_how_many_slots_one_parent_takes(self) -> None:
        """The failure it exists for: one strong parent filling the whole width with
        variations of itself, so the beam is nominally wide and effectively one."""
        self.assertEqual(cap_per_parent([0, 0, 1, 0, 1], 4, 2), [0, 1, 2, 4])

    def test_backfills_rather_than_starving_the_beam(self) -> None:
        """A cap tight enough to leave the beam short would confound the comparison with the
        incumbent: the search would proceed on fewer parents than the width asks for."""
        self.assertEqual(len(cap_per_parent([0, 0, 0, 0], 3, 1)), 3)

    def test_never_returns_more_than_the_width(self) -> None:
        self.assertEqual(len(cap_per_parent([0, 1, 2, 3, 4], 2, 1)), 2)

    def test_backfilled_positions_stay_in_score_order(self) -> None:
        kept = cap_per_parent([0, 0, 0, 1], 3, 1)
        self.assertEqual(kept, sorted(kept))


class TestDeterminantalSelect(unittest.TestCase):
    FEATURES: ClassVar[list[frozenset[str]]] = [
        frozenset({"a"}),
        frozenset({"a"}),
        frozenset({"b"}),
        frozenset({"c"}),
    ]

    def test_zero_weight_takes_the_best_in_order(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(determinantal_select(scores, self.FEATURES, 2, 0.0), [0, 1])

    def test_weight_prefers_a_new_feature_over_a_repeat(self) -> None:
        """Two subsets over the same features say the same thing twice, which for an equation
        meant to be read is exactly what the beam's width should not be spent on."""
        scores = np.array([1.0, 1.05, 1.1, 5.0])
        self.assertEqual(determinantal_select(scores, self.FEATURES, 2, 1.0), [0, 2])

    def test_returns_at_most_the_width(self) -> None:
        scores = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(len(determinantal_select(scores, self.FEATURES, 3, 0.5)), 3)

    def test_survives_terms_with_no_features(self) -> None:
        scores = np.array([1.0, 2.0])
        self.assertEqual(len(determinantal_select(scores, [frozenset(), frozenset()], 2, 1.0)), 2)


class TestVanillaIsUnchanged(unittest.TestCase):
    """The published equation must be identical with the policy machinery present.

    Fitted on the real corpus at the published configuration, both ways. Anything less than
    the real library and the real length would not exercise the paths a policy touches.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from ml_meta_perf.data import DATASET_FEATURES, columns_as_arrays, load, target
        from ml_meta_perf.experiment import DEFAULT_E3, EQUATION_MODEL_FEATURES
        from ml_meta_perf.fit import fit
        from ml_meta_perf.terms import build_library

        frame = load()
        columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
        library = build_library(
            DATASET_FEATURES,
            EQUATION_MODEL_FEATURES,
            columns,
            max_arity=DEFAULT_E3.max_arity,
            max_abs_zscore=DEFAULT_E3.max_abs_zscore,
        )
        common = {
            "max_terms": DEFAULT_E3.max_terms,
            "penalty": DEFAULT_E3.penalty,
            "pool_size": DEFAULT_E3.pool_size,
            "beam_width": DEFAULT_E3.beam_width,
        }
        truth = target(frame)
        cls.implicit = fit(library, truth, **common)
        cls.explicit = fit(library, truth, policy=VANILLA, **common)

    def test_the_same_terms_at_every_length(self) -> None:
        self.assertEqual(set(self.implicit.equations), set(self.explicit.equations))
        for size, equation in self.implicit.equations.items():
            with self.subTest(size=size):
                self.assertEqual(
                    [term.name for term in equation.terms],
                    [term.name for term in self.explicit.equations[size].terms],
                )

    def test_the_same_weights(self) -> None:
        for size, equation in self.implicit.equations.items():
            with self.subTest(size=size):
                np.testing.assert_allclose(equation.weights, self.explicit.equations[size].weights, rtol=0, atol=0)


class TestPoliciesChangeSomething(unittest.TestCase):
    """A variant that never changes the equation is not being measured, it is being ignored.

    Not every policy has to differ at every configuration -- a beam whose top children are far
    apart is unaffected by pruning, and that is a finding rather than a fault. But if *no*
    catalogued variant moved the equation anywhere, the machinery would be inert.
    """

    def test_at_least_one_variant_finds_a_different_equation(self) -> None:
        from ml_meta_perf.data import DATASET_FEATURES, columns_as_arrays, load, target
        from ml_meta_perf.experiment import DEFAULT_E3, EQUATION_MODEL_FEATURES
        from ml_meta_perf.fit import fit
        from ml_meta_perf.terms import build_library

        frame = load()
        columns = columns_as_arrays(frame, DATASET_FEATURES + EQUATION_MODEL_FEATURES)
        library = build_library(
            DATASET_FEATURES,
            EQUATION_MODEL_FEATURES,
            columns,
            max_arity=DEFAULT_E3.max_arity,
            max_abs_zscore=DEFAULT_E3.max_abs_zscore,
        )
        truth = target(frame)

        def terms(policy: BeamPolicy) -> set[str]:
            result = fit(
                library,
                truth,
                max_terms=DEFAULT_E3.headline_terms,
                penalty=DEFAULT_E3.penalty,
                pool_size=DEFAULT_E3.pool_size,
                beam_width=DEFAULT_E3.beam_width,
                policy=policy,
            )
            return {term.name for term in result.equations[DEFAULT_E3.headline_terms].terms}

        baseline = terms(VANILLA)
        moved = [policy.name for policy in CATALOGUE if policy.name != "vanilla" and terms(policy) != baseline]
        self.assertGreater(len(moved), 0, "no beam policy changed the equation on the real corpus")


if __name__ == "__main__":
    unittest.main()
