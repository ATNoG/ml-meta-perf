"""The term library, and in particular the filters that keep unstable terms out."""

import unittest

import numpy as np

from metafit.terms import (
    Atom,
    Library,
    Term,
    build_library,
    composition_atom,
    denominator_is_safe,
    is_admissible,
    pairwise_terms,
    sum_ratio_terms,
    unary_terms,
)


def columns(**kwargs: list[float]) -> dict[str, np.ndarray]:
    return {name: np.array(values, dtype=np.float64) for name, values in kwargs.items()}


class TestAtom(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = columns(a=[1.0, 2.0, 4.0], b=[0.0, 1.0, 2.0])

    def test_names(self) -> None:
        self.assertEqual(Atom("a", "id").name, "a")
        self.assertEqual(Atom("a", "log").name, "log(a)")
        self.assertEqual(Atom("a", "inv").name, "1/a")
        self.assertEqual(Atom("a", "sq").name, "a^2")
        self.assertEqual(Atom("a", "sqrt").name, "sqrt(a)")

    def test_evaluation(self) -> None:
        np.testing.assert_allclose(Atom("a", "id").evaluate(self.columns), [1.0, 2.0, 4.0])
        np.testing.assert_allclose(Atom("a", "sq").evaluate(self.columns), [1.0, 4.0, 16.0])
        np.testing.assert_allclose(Atom("a", "inv").evaluate(self.columns), [1.0, 0.5, 0.25])
        np.testing.assert_allclose(Atom("a", "log").evaluate(self.columns), np.log([1.0, 2.0, 4.0]))

    def test_transforms_needing_positivity_are_rejected_on_zero(self) -> None:
        self.assertTrue(Atom("a", "log").is_defined_on(self.columns))
        self.assertFalse(Atom("b", "log").is_defined_on(self.columns))
        self.assertFalse(Atom("b", "inv").is_defined_on(self.columns))
        self.assertTrue(Atom("b", "sq").is_defined_on(self.columns))

    def test_unknown_feature_is_not_defined(self) -> None:
        self.assertFalse(Atom("missing", "id").is_defined_on(self.columns))


class TestTerm(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = columns(a=[2.0, 4.0, 8.0], b=[1.0, 2.0, 4.0], c=[1.0, 1.0, 2.0])

    def test_names_show_the_structure(self) -> None:
        a, b, c = Atom("a"), Atom("b"), Atom("c")
        self.assertEqual(Term("atom", (a,)).name, "a")
        self.assertEqual(Term("ratio", (a, b)).name, "[a] / [b]")
        self.assertEqual(Term("product", (a, b)).name, "[a] * [b]")
        self.assertEqual(Term("sum_ratio", (a, b, c)).name, "([a] + [b]) / [c]")

    def test_arithmetic(self) -> None:
        a, b, c = Atom("a"), Atom("b"), Atom("c")
        np.testing.assert_allclose(Term("ratio", (a, b)).evaluate(self.columns), [2.0, 2.0, 2.0], rtol=1e-6)
        np.testing.assert_allclose(Term("product", (a, b)).evaluate(self.columns), [2.0, 8.0, 32.0])
        np.testing.assert_allclose(
            Term("sum_ratio", (a, b, c)).evaluate(self.columns), [3.0, 6.0, 6.0], rtol=1e-6
        )

    def test_features_reports_every_operand(self) -> None:
        term = Term("sum_ratio", (Atom("a"), Atom("b"), Atom("c")))
        self.assertEqual(term.features, ("a", "b", "c"))

    def test_negative_denominator_keeps_its_sign(self) -> None:
        data = columns(x=[1.0], y=[-2.0])
        result = Term("ratio", (Atom("x"), Atom("y"))).evaluate(data)
        self.assertLess(float(result[0]), 0.0)

    def test_round_trips_through_a_dict(self) -> None:
        term = Term("sum_ratio", (Atom("a", "log"), Atom("b"), Atom("c", "sq")))
        restored = Term.from_dict(term.to_dict())
        self.assertEqual(restored, term)
        self.assertEqual(restored.name, term.name)

    def test_malformed_payload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Term.from_dict({"operation": "atom", "operands": "not a list"})


class TestAdmissibility(unittest.TestCase):
    def test_rejects_non_finite_and_constant(self) -> None:
        self.assertFalse(is_admissible(np.array([1.0, np.inf, 2.0])))
        self.assertFalse(is_admissible(np.array([np.nan, 1.0])))
        self.assertFalse(is_admissible(np.array([2.0, 2.0, 2.0])))

    def test_rejects_effectively_constant_terms(self) -> None:
        # Large magnitude, negligible relative spread. De-standardising a weight for
        # such a term is what produced a 1.5e9 coefficient against a 1.5e9 intercept.
        self.assertFalse(is_admissible(np.array([1e9, 1e9 + 1e-3, 1e9 - 1e-3])))

    def test_rejects_terms_driven_by_a_single_row(self) -> None:
        # A lone spike can only reach a z-score of about sqrt(n), so the sample has to
        # be big enough for the default cap of 8 to be able to fire at all.
        spike = np.concatenate([np.ones(200), np.array([1e6])])
        self.assertFalse(is_admissible(spike))

    def test_cap_cannot_fire_below_its_own_sample_size_limit(self) -> None:
        # Documents the scale dependence: with 51 rows no single outlier can exceed
        # sqrt(51) = 7.1, so a cap of 8 admits even an extreme spike. This is why the
        # 20-row E1 fit uses a tighter cap than the 476-row E2 fit.
        spike = np.concatenate([np.ones(50), np.array([1e6])])
        self.assertTrue(is_admissible(spike, max_abs_zscore=8.0))
        self.assertFalse(is_admissible(spike, max_abs_zscore=3.0))

    def test_accepts_a_well_behaved_term(self) -> None:
        self.assertTrue(is_admissible(np.linspace(1.0, 10.0, 40)))

    def test_cap_is_configurable(self) -> None:
        values = np.concatenate([np.zeros(30), np.array([6.0])])
        self.assertFalse(is_admissible(values, max_abs_zscore=3.0))
        self.assertTrue(is_admissible(values, max_abs_zscore=100.0))


class TestDenominatorSafety(unittest.TestCase):
    def test_non_ratio_terms_are_always_safe(self) -> None:
        self.assertTrue(denominator_is_safe(Term("atom", (Atom("a"),)), columns(a=[0.0, 1.0])))
        self.assertTrue(
            denominator_is_safe(Term("product", (Atom("a"), Atom("b"))), columns(a=[0.0], b=[0.0]))
        )

    def test_denominator_reaching_zero_is_rejected(self) -> None:
        data = columns(a=[1.0, 2.0, 3.0], b=[0.0, 1.0, 2.0])
        self.assertFalse(denominator_is_safe(Term("ratio", (Atom("a"), Atom("b"))), data))

    def test_denominator_clear_of_zero_is_accepted(self) -> None:
        data = columns(a=[1.0, 2.0, 3.0], b=[10.0, 11.0, 12.0])
        self.assertTrue(denominator_is_safe(Term("ratio", (Atom("a"), Atom("b"))), data))

    def test_log_denominator_touching_one_is_rejected(self) -> None:
        # log(1) == 0, an easy way to divide by zero without any zero in the data.
        data = columns(a=[5.0, 6.0, 7.0], b=[1.0, 10.0, 100.0])
        self.assertFalse(denominator_is_safe(Term("ratio", (Atom("a"), Atom("b", "log"))), data))


class TestBuilders(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = columns(p=[1.0, 2.0, 3.0, 4.0], q=[0.0, 1.0, 2.0, 3.0])

    def test_composition_atom_compresses_only_positive_features(self) -> None:
        self.assertEqual(composition_atom("p", self.columns).transform, "log")
        self.assertEqual(composition_atom("q", self.columns).transform, "id")

    def test_unary_terms_skip_undefined_transforms(self) -> None:
        names = {term.name for term in unary_terms(("q",), self.columns)}
        self.assertIn("q", names)
        self.assertIn("q^2", names)
        self.assertNotIn("log(q)", names)
        self.assertNotIn("1/q", names)

    def test_pairwise_within_a_group_is_one_directional(self) -> None:
        terms = pairwise_terms(("p", "q"), ("p", "q"), self.columns, both_directions=False)
        names = {term.name for term in terms}
        self.assertEqual(sum(1 for name in names if " / " in name), 1)

    def test_pairwise_across_groups_is_bidirectional(self) -> None:
        terms = pairwise_terms(("p",), ("q",), self.columns, both_directions=True)
        names = {term.name for term in terms}
        self.assertEqual(sum(1 for name in names if " / " in name), 2)

    def test_pairwise_never_pairs_a_feature_with_itself(self) -> None:
        terms = pairwise_terms(("p",), ("p",), self.columns, both_directions=True)
        self.assertEqual(terms, [])

    def test_sum_ratio_uses_three_distinct_features(self) -> None:
        data = columns(a=[1.0, 2.0], b=[3.0, 4.0], c=[5.0, 6.0])
        for term in sum_ratio_terms(("a", "b", "c"), data):
            self.assertEqual(len(set(term.features)), 3)


class TestLibrary(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.columns = {name: rng.uniform(1.0, 9.0, 40) for name in ("f1", "f2", "f3")}

    def test_matrix_matches_terms(self) -> None:
        library = build_library(("f1", "f2", "f3"), (), self.columns)
        self.assertEqual(library.matrix.shape, (40, len(library)))
        self.assertEqual(len(library.names), len(library.terms))

    def test_names_are_unique(self) -> None:
        library = build_library(("f1", "f2", "f3"), (), self.columns)
        self.assertEqual(len(set(library.names)), len(library.names))

    def test_every_column_is_finite(self) -> None:
        library = build_library(("f1", "f2"), ("f3",), self.columns)
        self.assertTrue(np.all(np.isfinite(library.matrix)))

    def test_model_features_add_cross_terms(self) -> None:
        without = build_library(("f1", "f2"), (), self.columns)
        with_model = build_library(("f1", "f2"), ("f3",), self.columns)
        self.assertGreater(len(with_model), len(without))

    def test_design_reevaluates_a_subset_on_fresh_columns(self) -> None:
        library = build_library(("f1", "f2"), (), self.columns)
        indices = [2, 0, 1]
        design = library.design(self.columns, indices)
        np.testing.assert_allclose(design, library.matrix[:, indices])

    def test_empty_library_is_an_error(self) -> None:
        constant = {"z": np.ones(10)}
        with self.assertRaises(ValueError):
            Library([Term("atom", (Atom("z"),))], constant)


if __name__ == "__main__":
    unittest.main()
