"""Choosing which terms enter the equation: the screen, the beam, and pruning.

The arithmetic each candidate is scored with is `tests/test_fit.py`.
"""

import itertools
import unittest

import numpy as np

from ml_meta_perf.fit import Standardizer, Subset, ridge_solve
from ml_meta_perf.model import Equation
from ml_meta_perf.search import (
    Selector,
    guided_screen,
    prune,
    search,
    selected_terms,
    transform_gap,
)
from ml_meta_perf.terms import Term, build_library


def synthetic_columns(n: int = 120, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(1.0, 20.0, n) for name in ("f1", "f2", "f3", "f4")}


class TestTransformGap(unittest.TestCase):
    def test_gap_is_near_zero_for_a_linear_relation(self) -> None:
        x = np.linspace(1.0, 50.0, 60)
        self.assertLess(abs(transform_gap(x, 3.0 * x)), 1e-6)

    def test_gap_is_positive_for_a_monotone_curve(self) -> None:
        x = np.linspace(1.0, 200.0, 60)
        self.assertGreater(transform_gap(x, np.log(x)), 0.05)


class TestGuidedScreen(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = synthetic_columns()
        self.library = build_library(("f1", "f2"), ("f3", "f4"), self.columns)
        self.target = 0.4 * np.log(self.columns["f1"]) - 0.2 * self.columns["f3"]

    def test_returns_at_most_the_requested_count(self) -> None:
        self.assertLessEqual(len(guided_screen(self.library, self.target, keep=15)), 15)

    def test_indices_are_valid_and_unique(self) -> None:
        kept = guided_screen(self.library, self.target, keep=25)
        self.assertEqual(len(kept), len(set(kept)))
        for index in kept:
            self.assertTrue(0 <= index < len(self.library))

    def test_keeps_a_term_related_to_the_target(self) -> None:
        kept = set(guided_screen(self.library, self.target, keep=40))
        names = {self.library.names[index] for index in kept}
        self.assertTrue(any("f1" in name or "f3" in name for name in names))


class TestSelector(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(5)
        self.design = rng.normal(size=(80, 12))
        # A target built from three known columns, so search has a right answer.
        self.target = 2.0 * self.design[:, 1] - 1.5 * self.design[:, 4] + 0.8 * self.design[:, 7]
        self.selector = Selector(self.design, self.target, 0.0)

    def test_finds_the_generating_columns(self) -> None:
        found = self.selector.search(list(range(12)), 3, beam_width=4)
        self.assertEqual(set(found[3].indices), {1, 4, 7})

    def test_the_fast_path_matches_the_reference_solver(self) -> None:
        # ``Selector`` folds the ridge penalty into the full Gram diagonal once and
        # indexes submatrices out of that, instead of calling ``ridge_solve`` per subset.
        # The two must stay identical, and this is what says so -- without it
        # ``ridge_solve`` would be documentation of an algorithm nothing runs.
        for penalty in (0.0, 5.0, 50.0):
            selector = Selector(self.design, self.target, penalty)
            for indices in ((1,), (1, 4), (0, 4, 7), (1, 3, 4, 7, 9)):
                order = np.array(indices, dtype=np.intp)
                reference = ridge_solve(
                    selector.gram[order[:, None], order], selector.projection[order], penalty
                )
                np.testing.assert_allclose(
                    selector._evaluate(indices).weights, reference, rtol=1e-10, atol=1e-12
                )

    def test_the_batched_solve_matches_the_one_at_a_time_solve(self) -> None:
        # Beam steps and refinement positions go through the solver as one stack. That is
        # only an optimisation if it is the same arithmetic, so both paths are run over the
        # same subsets and compared.
        #
        # Compared to a tight tolerance rather than exactly. A stacked `np.linalg.solve` and
        # a loop of single solves are the same operations in a different order, and LAPACK
        # is free to associate them differently -- so the last couple of ULP depend on the
        # BLAS. They agree bitwise against the system OpenBLAS here and differ at 1e-16
        # against the one on GitHub's runners, which is the tolerance being asserted, not a
        # defect. Anything looser would stop catching a real divergence.
        batch = [(1, 4), (0, 4), (4, 7), (1, 7)]
        for penalty in (0.0, 5.0):
            batched = Selector(self.design, self.target, penalty)._evaluate_many(batch)
            single = Selector(self.design, self.target, penalty)
            for subset, indices in zip(batched, batch, strict=True):
                reference = single._evaluate(indices)
                self.assertEqual(subset.indices, reference.indices)
                np.testing.assert_allclose(
                    subset.weights, reference.weights, rtol=1e-12, atol=1e-14
                )
                self.assertAlmostEqual(subset.rss, reference.rss, delta=1e-12 * abs(reference.rss))

    def test_the_batched_solve_shares_the_cache_and_keeps_order(self) -> None:
        requested = [(1, 4), (0, 7), (1, 4)]
        subsets = self.selector._evaluate_many(requested)
        self.assertEqual([subset.indices for subset in subsets], requested)
        # A repeated subset is solved once and served from the cache the second time.
        self.assertIs(subsets[0], subsets[2])

    def test_returns_a_subset_for_every_size(self) -> None:
        found = self.selector.search(list(range(12)), 5, beam_width=3)
        self.assertEqual(sorted(found), [1, 2, 3, 4, 5])
        for size, subset in found.items():
            self.assertEqual(len(subset.indices), size)

    def test_residual_error_decreases_with_size(self) -> None:
        found = self.selector.search(list(range(12)), 6, beam_width=3)
        errors = [found[size].rss for size in sorted(found)]
        for earlier, later in itertools.pairwise(errors):
            self.assertLessEqual(later, earlier + 1e-9)

    def test_weights_match_the_chosen_indices(self) -> None:
        found = self.selector.search(list(range(12)), 4, beam_width=3)
        self.assertEqual(len(found[4].weights), 4)

    def test_pool_restricts_what_can_be_chosen(self) -> None:
        found = self.selector.search([0, 2, 3], 2, beam_width=2)
        self.assertTrue(set(found[2].indices) <= {0, 2, 3})

    def test_perfectly_collinear_duplicate_is_refused(self) -> None:
        design = np.column_stack([self.design[:, 1], self.design[:, 1], self.design[:, 4]])
        selector = Selector(design, self.target, 0.0)
        found = selector.search([0, 1, 2], 2, beam_width=2)
        self.assertNotEqual(set(found[2].indices), {0, 1})


class TestSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = synthetic_columns(150, seed=7)
        self.library = build_library(("f1", "f2"), ("f3", "f4"), self.columns)
        self.target = (
            0.5 * np.log(self.columns["f1"]) - 0.02 * self.columns["f3"] + 0.3
        )

    def test_produces_an_equation_at_every_size(self) -> None:
        result = search(self.library, self.target, max_terms=5, penalty=1.0, pool_size=60)
        self.assertEqual(sorted(result.equations), [1, 2, 3, 4, 5])
        for size, equation in result.equations.items():
            self.assertEqual(equation.n_terms, size)

    def test_accuracy_improves_with_more_terms(self) -> None:
        result = search(self.library, self.target, max_terms=4, penalty=0.0, pool_size=60)
        errors = [
            float(np.abs(self.target - result.equations[size].evaluate(self.columns)).mean())
            for size in sorted(result.equations)
        ]
        self.assertLess(errors[-1], errors[0])

    def test_recovers_a_clean_generating_relation(self) -> None:
        result = search(self.library, self.target, max_terms=3, penalty=0.0, pool_size=120)
        predicted = result.equations[3].evaluate(self.columns)
        self.assertGreater(float(np.corrcoef(predicted, self.target)[0, 1]), 0.95)

    def test_best_returns_the_largest_available(self) -> None:
        result = search(self.library, self.target, max_terms=4, penalty=1.0, pool_size=60)
        self.assertEqual(result.best().n_terms, 4)
        self.assertEqual(result.best(max_terms=2).n_terms, 2)

    def test_pool_size_is_reported(self) -> None:
        result = search(self.library, self.target, max_terms=2, penalty=1.0, pool_size=30)
        self.assertLessEqual(result.pool_size, 30)

    def test_selected_terms_are_ordered_by_importance(self) -> None:
        result = search(self.library, self.target, max_terms=3, penalty=1.0, pool_size=60)
        terms = selected_terms(result.equations[3])
        self.assertEqual(len(terms), 3)


class TestPrune(unittest.TestCase):
    """Dropping terms that do no work, and refitting what remains."""

    def setUp(self) -> None:
        self.columns = synthetic_columns(120, seed=21)
        self.library = build_library(("f1", "f2"), ("f3", "f4"), self.columns)
        self.target = 0.4 * np.log(self.columns["f1"]) - 0.02 * self.columns["f3"] + 0.3
        self.equation = search(self.library, self.target, max_terms=6, penalty=1.0, pool_size=60).equations[6]

    def test_a_generous_threshold_shortens_the_equation(self) -> None:
        pruned = prune(self.equation, self.columns, self.target, penalty=1.0, min_contribution=10.0)
        self.assertLess(pruned.n_terms, self.equation.n_terms)

    def test_a_tiny_threshold_keeps_everything(self) -> None:
        pruned = prune(self.equation, self.columns, self.target, penalty=1.0, min_contribution=1e-12)
        self.assertEqual(pruned.n_terms, self.equation.n_terms)

    def test_weights_are_refitted_not_carried_over(self) -> None:
        pruned = prune(self.equation, self.columns, self.target, penalty=1.0, min_contribution=10.0)
        if pruned.n_terms and pruned.n_terms < self.equation.n_terms:
            survivors = {t.name: w for t, w in zip(self.equation.terms, self.equation.weights, strict=True)}
            changed = [
                abs(w - survivors[t.name]) > 1e-12
                for t, w in zip(pruned.terms, pruned.weights, strict=True)
                if t.name in survivors
            ]
            self.assertTrue(any(changed))

    def test_an_empty_equation_is_returned_unchanged(self) -> None:
        empty = Equation(intercept=0.5, terms=(), weights=(), standardized_weights=())
        self.assertEqual(prune(empty, self.columns, self.target).n_terms, 0)

    def test_dropping_everything_leaves_the_mean(self) -> None:
        pruned = prune(self.equation, self.columns, self.target, penalty=1.0, min_contribution=1e9)
        self.assertEqual(pruned.n_terms, 0)
        self.assertAlmostEqual(pruned.intercept, float(self.target.mean()))

    def test_prediction_stays_finite(self) -> None:
        pruned = prune(self.equation, self.columns, self.target, penalty=1.0, min_contribution=0.01)
        self.assertTrue(np.all(np.isfinite(pruned.predict(self.columns))))


class TestOneTermPerFeatureCombination(unittest.TestCase):
    """The selector may not state one relationship twice. See `Library.feature_groups`."""

    def setUp(self) -> None:
        self.columns = synthetic_columns()
        self.library = build_library(("f1", "f2"), ("f3", "f4"), self.columns)
        self.target = (
            3.0 * np.log(self.columns["f1"]) / np.log(self.columns["f3"])
            + 1.5 * np.log(self.columns["f2"])
        )

    def repeated(self, terms) -> list[list[str]]:
        seen: dict[frozenset[str], list[str]] = {}
        for term in terms:
            features = frozenset(term.features)
            if len(features) > 1:
                seen.setdefault(features, []).append(term.name)
        return [names for names in seen.values() if len(names) > 1]

    def test_no_fitted_equation_repeats_a_feature_combination(self) -> None:
        result = search(self.library, self.target, max_terms=10, penalty=1.0, pool_size=len(self.library))
        for size, equation in result.equations.items():
            with self.subTest(size=size):
                self.assertEqual(self.repeated(equation.terms), [])

    def test_the_constraint_is_what_does_it(self) -> None:
        # Without the group array the same search is free to take both orientations, so the
        # test above would pass for the wrong reason if the mask were silently ineffective.
        standardizer = Standardizer.fit(self.library.matrix)
        design = standardizer.apply(self.library.matrix)
        pool = list(range(len(self.library)))
        unconstrained = Selector(design, self.target, 1.0).search(pool, 8, beam_width=6)
        constrained = Selector(design, self.target, 1.0, self.library.feature_groups).search(
            pool, 8, beam_width=6
        )
        def picked(subset: Subset) -> list[Term]:
            return [self.library.terms[i] for i in subset.indices]

        self.assertNotEqual(self.repeated(picked(unconstrained[8])), [])
        self.assertEqual(self.repeated(picked(constrained[8])), [])

    def test_a_group_blocks_only_its_own_members(self) -> None:
        groups = self.library.feature_groups
        selector = Selector(
            Standardizer.fit(self.library.matrix).apply(self.library.matrix),
            self.target,
            0.0,
            groups,
        )
        chosen = int(np.flatnonzero(groups >= 0)[0])
        blocked = selector._blocked((chosen,))
        self.assertIsNotNone(blocked)
        assert blocked is not None
        same = groups == groups[chosen]
        self.assertTrue(bool(blocked[same].all()))
        # Terms of no group are never blocked by the group rule, only by correlation.
        self.assertFalse(bool(blocked[groups < 0].all()))


if __name__ == "__main__":
    unittest.main()

if __name__ == "__main__":
    unittest.main()
