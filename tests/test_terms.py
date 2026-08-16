"""The term library, and in particular the filters that keep unstable terms out."""

import unittest

import numpy as np

from metafit.terms import (
    Atom,
    Library,
    Term,
    build_library,
    composition_atom,
    denominator_atom,
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


class TestNesting(unittest.TestCase):
    """A term may take another term as an operand, so one term can be a deep expression."""

    def setUp(self) -> None:
        self.columns = columns(a=[2.0, 4.0, 8.0], b=[1.0, 2.0, 4.0], c=[3.0, 6.0, 12.0])
        self.inner = Term("ratio", (Atom("a"), Atom("b")))
        self.outer = Term("product", (self.inner, Atom("c")))

    def test_depth_counts_nesting(self) -> None:
        self.assertEqual(Term("atom", (Atom("a"),)).depth, 1)
        self.assertEqual(self.inner.depth, 1)
        self.assertEqual(self.outer.depth, 2)
        self.assertEqual(Term("product", (self.outer, Atom("b"))).depth, 3)

    def test_features_are_collected_recursively(self) -> None:
        self.assertEqual(set(self.outer.features), {"a", "b", "c"})

    def test_evaluation_recurses(self) -> None:
        expected = (self.columns["a"] / self.columns["b"]) * self.columns["c"]
        np.testing.assert_allclose(self.outer.evaluate(self.columns), expected, rtol=1e-6)

    def test_name_shows_the_structure(self) -> None:
        self.assertEqual(self.outer.name, "[[a] / [b]] * [c]")

    def test_nested_terms_round_trip(self) -> None:
        restored = Term.from_dict(self.outer.to_dict())
        self.assertEqual(restored, self.outer)
        self.assertEqual(restored.depth, 2)
        np.testing.assert_allclose(
            restored.evaluate(self.columns), self.outer.evaluate(self.columns)
        )

    def test_deeply_nested_round_trip(self) -> None:
        deep = Term("ratio", (Term("product", (self.outer, Atom("b"))), Atom("c")))
        self.assertEqual(Term.from_dict(deep.to_dict()), deep)

    def test_malformed_nested_operand_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Term.from_dict({"operation": "product", "operands": ["not an object"]})


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


class TestDenominatorEligibility(unittest.TestCase):
    """Unsafe divisors are never generated, rather than generated and then screened."""

    def test_feature_reaching_zero_is_never_a_denominator(self) -> None:
        self.assertIsNone(denominator_atom("b", columns(a=[1.0, 2.0, 3.0], b=[0.0, 1.0, 2.0])))

    def test_feature_clear_of_zero_uses_its_log(self) -> None:
        atom = denominator_atom("b", columns(b=[10.0, 11.0, 12.0]))
        self.assertIsNotNone(atom)
        assert atom is not None
        self.assertEqual(atom.transform, "log")

    def test_falls_back_to_the_raw_feature_when_the_log_reaches_zero(self) -> None:
        # log(1) == 0, so log is ineligible, but the feature itself is tight around 1.
        atom = denominator_atom("b", columns(b=[0.9, 1.0, 1.1]))
        self.assertIsNotNone(atom)
        assert atom is not None
        self.assertEqual(atom.transform, "id")

    def test_wide_range_divisor_is_refused_even_though_it_never_reaches_zero(self) -> None:
        # min is 1, so a "non-zero denominator" rule would admit this. Dividing by it
        # spans two orders of magnitude and hands one row the term's whole variance.
        self.assertIsNone(denominator_atom("b", columns(b=[1.0, 10.0, 100.0])))

    def test_log_compression_rescues_a_wide_range_feature(self) -> None:
        # nr_inst-like: hopeless raw, comfortable once compressed.
        atom = denominator_atom("b", columns(b=[165.0, 20000.0, 7_062_606.0]))
        self.assertIsNotNone(atom)
        assert atom is not None
        self.assertEqual(atom.transform, "log")

    def test_negative_feature_can_still_divide_when_clear_of_zero(self) -> None:
        atom = denominator_atom("b", columns(b=[-5.0, -6.0, -7.0]))
        self.assertIsNotNone(atom)
        assert atom is not None
        self.assertEqual(atom.transform, "id")

    def test_no_generated_ratio_has_a_denominator_near_zero(self) -> None:
        # The structural guarantee: over the real meta-dataset, every ratio the library
        # emits has a divisor that stays clear of zero, so none of them can spike.
        from metafit.data import DATASET_FEATURES, MODEL_FEATURES, columns_as_arrays, load

        data = columns_as_arrays(load(), DATASET_FEATURES + MODEL_FEATURES)
        library = build_library(DATASET_FEATURES, MODEL_FEATURES, data)
        checked = 0
        for term in library.terms:
            if term.operation not in ("ratio", "sum_ratio"):
                continue
            checked += 1
            divisor = term.operands[-1].evaluate(data)
            self.assertGreater(float(np.min(np.abs(divisor))), 0.0, term.name)
            self.assertGreater(
                float(np.min(np.abs(divisor))), 0.05 * float(np.std(divisor)), term.name
            )
        self.assertGreater(checked, 0)


class TestBuilders(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = columns(p=[1.0, 2.0, 3.0, 4.0], q=[0.0, 1.0, 2.0, 3.0])
        # Both eligible as denominators, so the direction tests measure direction and
        # not eligibility.
        self.divisible = columns(p=[10.0, 12.0, 14.0, 16.0], q=[20.0, 24.0, 28.0, 32.0])

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
        terms = pairwise_terms(("p", "q"), ("p", "q"), self.divisible, both_directions=False)
        names = {term.name for term in terms}
        self.assertEqual(sum(1 for name in names if " / " in name), 1)

    def test_pairwise_across_groups_is_bidirectional(self) -> None:
        terms = pairwise_terms(("p",), ("q",), self.divisible, both_directions=True)
        names = {term.name for term in terms}
        self.assertEqual(sum(1 for name in names if " / " in name), 2)

    def test_products_survive_even_when_no_ratio_can_be_formed(self) -> None:
        # q reaches zero, so it cannot be divided by -- but multiplying by it is fine,
        # and refusing the ratio must not cost the product.
        terms = pairwise_terms(("p",), ("q",), self.columns, both_directions=True)
        names = {term.name for term in terms}
        self.assertEqual(sum(1 for name in names if " * " in name), 1)
        self.assertNotIn("[log(p)] / [q]", names)

    def test_pairwise_never_pairs_a_feature_with_itself(self) -> None:
        terms = pairwise_terms(("p",), ("p",), self.divisible, both_directions=True)
        self.assertEqual(terms, [])

    def test_sum_ratio_skips_ineligible_divisors(self) -> None:
        data = columns(a=[1.0, 2.0], b=[3.0, 4.0], zero=[0.0, 1.0])
        for term in sum_ratio_terms(("a", "b", "zero"), data):
            divisor = term.operands[-1]
            # build_library only ever composes atoms; nesting comes from metafit.construct.
            self.assertIsInstance(divisor, Atom)
            assert isinstance(divisor, Atom)
            self.assertNotEqual(divisor.feature, "zero")

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
