"""Agglomerative term construction."""

import unittest

import numpy as np

from metafit.construct import (
    agglomerate,
    constructed_library,
    curvature,
    linearity,
    straighten,
    structural_terms,
)
from metafit.terms import Atom, Term


def columns(**kwargs: list[float]) -> dict[str, np.ndarray]:
    return {name: np.array(values, dtype=np.float64) for name, values in kwargs.items()}


class TestLinearity(unittest.TestCase):
    def test_perfect_line_scores_one(self) -> None:
        x = np.linspace(1.0, 10.0, 30)
        self.assertAlmostEqual(linearity(x, 2.0 * x + 3.0), 1.0)

    def test_sign_is_ignored(self) -> None:
        x = np.linspace(1.0, 10.0, 30)
        self.assertAlmostEqual(linearity(x, -x), 1.0)

    def test_curvature_is_zero_for_a_line(self) -> None:
        x = np.linspace(1.0, 10.0, 30)
        self.assertAlmostEqual(curvature(x, 3.0 * x), 0.0, places=9)

    def test_curvature_is_positive_for_a_monotone_curve(self) -> None:
        x = np.linspace(1.0, 200.0, 60)
        self.assertGreater(curvature(x, np.log(x)), 0.05)


class TestStraighten(unittest.TestCase):
    def test_picks_the_transform_that_linearises(self) -> None:
        # y is linear in log(x), so log is the transform that straightens x.
        x = np.linspace(1.0, 500.0, 80)
        atom = straighten("x", {"x": x}, np.log(x))
        self.assertEqual(atom.transform, "log")

    def test_leaves_an_already_linear_feature_alone(self) -> None:
        x = np.linspace(1.0, 50.0, 60)
        self.assertEqual(straighten("x", {"x": x}, 2.0 * x).transform, "id")

    def test_never_returns_an_undefined_transform(self) -> None:
        # A feature reaching zero admits neither log nor inverse.
        x = np.linspace(0.0, 10.0, 40)
        atom = straighten("x", {"x": x}, x**2)
        self.assertIn(atom.transform, ("id", "sq"))


class TestAgglomerate(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.columns = {name: rng.uniform(2.0, 20.0, 80) for name in ("a", "b", "c")}
        self.target = self.columns["a"] / self.columns["b"]

    def test_returns_at_least_one_term_per_feature(self) -> None:
        terms = agglomerate(("a", "b", "c"), self.columns, self.target)
        used = {feature for term in terms for feature in term.features}
        self.assertEqual(used, {"a", "b", "c"})

    def test_terms_are_unique(self) -> None:
        names = [term.name for term in agglomerate(("a", "b", "c"), self.columns, self.target)]
        self.assertEqual(len(names), len(set(names)))

    def test_finds_the_generating_ratio(self) -> None:
        names = {term.name for term in agglomerate(("a", "b", "c"), self.columns, self.target)}
        self.assertTrue(any("a" in name and "b" in name and "/" in name for name in names))

    def test_strict_clustering_yields_fewer_terms(self) -> None:
        # Consuming parents is real hierarchical clustering but produces a pool too small
        # to build a long equation from, which is why it is not the default.
        rounds = agglomerate(("a", "b", "c"), self.columns, self.target)
        strict = agglomerate(("a", "b", "c"), self.columns, self.target, per_round=False)
        self.assertLessEqual(len(strict), len(rounds) + 1)

    def test_a_high_gain_threshold_suppresses_merging(self) -> None:
        terms = agglomerate(("a", "b", "c"), self.columns, self.target, min_gain=10.0)
        self.assertTrue(all(term.operation == "atom" for term in terms))

    def test_every_term_evaluates_finitely(self) -> None:
        for term in agglomerate(("a", "b", "c"), self.columns, self.target):
            self.assertTrue(np.all(np.isfinite(term.evaluate(self.columns))))


class TestStructuralTerms(unittest.TestCase):
    def test_uncorrelated_features_are_not_paired(self) -> None:
        rng = np.random.default_rng(1)
        data = {"a": rng.uniform(1.0, 9.0, 200), "b": rng.uniform(1.0, 9.0, 200)}
        self.assertEqual(structural_terms(("a", "b"), data, min_correlation=0.9), [])

    def test_correlated_features_are_paired(self) -> None:
        base = np.linspace(1.0, 20.0, 100)
        data = {"a": base, "b": base * 2.0 + 0.5}
        terms = structural_terms(("a", "b"), data, min_correlation=0.5)
        self.assertGreater(len(terms), 0)

    def test_uses_no_target(self) -> None:
        # The signature takes no target at all: that independence is why these terms cost
        # nothing at validation time.
        base = np.linspace(1.0, 20.0, 100)
        data = {"a": base, "b": base * 2.0}
        self.assertEqual(
            [t.name for t in structural_terms(("a", "b"), data)],
            [t.name for t in structural_terms(("a", "b"), data)],
        )


class TestConstructedLibrary(unittest.TestCase):
    def test_builds_a_usable_library(self) -> None:
        rng = np.random.default_rng(2)
        data = {name: rng.uniform(2.0, 20.0, 60) for name in ("d1", "d2", "m1")}
        target = data["d1"] / data["m1"]
        library = constructed_library(("d1", "d2"), ("m1",), data, target)
        self.assertGreater(len(library), 0)
        self.assertEqual(library.matrix.shape[0], 60)
        self.assertTrue(np.all(np.isfinite(library.matrix)))

    def test_works_without_model_features(self) -> None:
        rng = np.random.default_rng(3)
        data = {name: rng.uniform(2.0, 20.0, 60) for name in ("d1", "d2")}
        library = constructed_library(("d1", "d2"), (), data, data["d1"] * 0.1)
        self.assertGreater(len(library), 0)

    def test_terms_round_trip(self) -> None:
        rng = np.random.default_rng(4)
        data = {name: rng.uniform(2.0, 20.0, 40) for name in ("d1", "m1")}
        for term in constructed_library(("d1",), ("m1",), data, data["d1"]).terms:
            self.assertEqual(Term.from_dict(term.to_dict()), term)

    def test_operands_are_atoms(self) -> None:
        rng = np.random.default_rng(5)
        data = {name: rng.uniform(2.0, 20.0, 40) for name in ("d1", "m1")}
        for term in constructed_library(("d1",), ("m1",), data, data["d1"]).terms:
            for operand in term.operands:
                self.assertIsInstance(operand, Atom)


if __name__ == "__main__":
    unittest.main()
