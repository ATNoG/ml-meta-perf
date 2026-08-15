"""The Equation object: evaluation, clipping, rendering and serialisation."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from metafit.model import Equation
from metafit.terms import Atom, Term


def equation(intercept: float = 0.5) -> Equation:
    return Equation(
        intercept=intercept,
        terms=(Term("atom", (Atom("a"),)), Term("ratio", (Atom("a"), Atom("b")))),
        weights=(0.25, -0.5),
        standardized_weights=(0.1, -0.9),
        name="test",
    )


class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = {"a": np.array([1.0, 2.0]), "b": np.array([1.0, 4.0])}

    def test_evaluate_applies_the_weights(self) -> None:
        # 0.5 + 0.25*a - 0.5*(a/b)
        np.testing.assert_allclose(
            equation().evaluate(self.columns), [0.5 + 0.25 - 0.5, 0.5 + 0.5 - 0.25], rtol=1e-6
        )

    def test_predict_clips_into_the_mcc_range(self) -> None:
        wild = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("a"),)),),
            weights=(100.0,),
            standardized_weights=(1.0,),
        )
        columns = {"a": np.array([5.0, -5.0])}
        np.testing.assert_allclose(wild.predict(columns), [1.0, -1.0])
        # evaluate is deliberately unclipped, so the two differ
        self.assertGreater(float(wild.evaluate(columns)[0]), 1.0)

    def test_an_equation_with_no_terms_is_a_constant(self) -> None:
        constant = Equation(intercept=0.3, terms=(), weights=(), standardized_weights=())
        np.testing.assert_allclose(constant.evaluate({"a": np.zeros(4)}), np.full(4, 0.3))

    def test_n_terms(self) -> None:
        self.assertEqual(equation().n_terms, 2)


class TestConsistency(unittest.TestCase):
    def test_mismatched_weights_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Equation(
                intercept=0.0,
                terms=(Term("atom", (Atom("a"),)),),
                weights=(1.0, 2.0),
                standardized_weights=(1.0, 2.0),
            )


class TestRendering(unittest.TestCase):
    def test_ranked_terms_order_by_standardised_magnitude(self) -> None:
        ranked = equation().ranked_terms()
        self.assertEqual(ranked[0][0].name, "[a] / [b]")
        self.assertAlmostEqual(ranked[0][2], -0.9)

    def test_str_shows_the_equation_and_the_betas(self) -> None:
        rendered = str(equation())
        self.assertIn("MCC =", rendered)
        self.assertIn("[a] / [b]", rendered)
        self.assertIn("beta=", rendered)

    def test_latex_escapes_underscores(self) -> None:
        latex = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("nr_attr"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        ).to_latex()
        self.assertIn(r"nr\_attr", latex)
        self.assertIn(r"\mathrm{MCC}", latex)


class TestSerialisation(unittest.TestCase):
    def test_round_trips_through_a_dict(self) -> None:
        original = equation()
        restored = Equation.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_round_trips_through_a_file(self) -> None:
        original = equation()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eq.json"
            original.save(path)
            self.assertEqual(Equation.load(path), original)

    def test_saved_file_is_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eq.json"
            equation().save(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["name"], "test")
            self.assertEqual(len(payload["terms"]), 2)

    def test_restored_equation_predicts_identically(self) -> None:
        original = equation()
        columns = {"a": np.array([1.0, 3.0, 7.0]), "b": np.array([2.0, 5.0, 9.0])}
        np.testing.assert_allclose(
            Equation.from_dict(original.to_dict()).predict(columns), original.predict(columns)
        )

    def test_malformed_payload_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Equation.from_dict({"intercept": 0.0, "terms": None, "weights": [], "standardized_weights": []})


if __name__ == "__main__":
    unittest.main()
