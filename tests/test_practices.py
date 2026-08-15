"""Extraction of written practices, and the filters that keep unsupported ones out."""

import unittest

import numpy as np
import polars as pl

from metafit.model import Equation
from metafit.practices import best_practices, feature_practices, render
from metafit.terms import Atom, Term


def rising_equation() -> Equation:
    """MCC increases with ``up`` and decreases with ``down``."""
    return Equation(
        intercept=0.5,
        terms=(Term("atom", (Atom("up"),)), Term("atom", (Atom("down"),))),
        weights=(0.10, -0.08),
        standardized_weights=(0.6, -0.4),
    )


def columns(n: int = 100) -> dict[str, np.ndarray]:
    return {"up": np.linspace(1.0, 5.0, n), "down": np.linspace(1.0, 5.0, n)[::-1].copy()}


def stability_table(**frequencies: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "term": list(frequencies),
            "folds": [round(20 * value) for value in frequencies.values()],
            "frequency": list(frequencies.values()),
        }
    )


class TestFeaturePractices(unittest.TestCase):
    def setUp(self) -> None:
        self.table = feature_practices(rising_equation(), columns())

    def test_one_row_per_feature(self) -> None:
        self.assertEqual(sorted(self.table["feature"].to_list()), ["down", "up"])

    def test_direction_matches_the_relationship(self) -> None:
        rows = {row["feature"]: row for row in self.table.iter_rows(named=True)}
        self.assertGreater(rows["up"]["direction"], 0.99)
        self.assertLess(rows["down"]["direction"], -0.99)

    def test_effect_is_signed_and_follows_the_feature(self) -> None:
        rows = {row["feature"]: row for row in self.table.iter_rows(named=True)}
        # up rises from 1 to 5 with weight 0.10, so its top decile contributes more.
        self.assertGreater(rows["up"]["effect"], 0.0)
        self.assertLess(rows["down"]["effect"], 0.0)

    def test_effect_measures_response_not_spread(self) -> None:
        # A quantile of the contributions alone would be order-invariant and identical
        # for both features; ordering by the feature is what makes the sign meaningful.
        effects = self.table["effect"].to_numpy()
        self.assertNotAlmostEqual(float(effects[0]), float(effects[1]))

    def test_glossary_is_used_when_available(self) -> None:
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("nr_class"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        table = feature_practices(equation, {"nr_class": np.linspace(2.0, 9.0, 30)})
        self.assertIn("classes", table["meaning"][0])

    def test_stability_is_nan_without_a_table(self) -> None:
        self.assertTrue(np.isnan(self.table["stability"].to_numpy()).all())

    def test_stability_is_read_from_the_table(self) -> None:
        table = feature_practices(rising_equation(), columns(), stability_table(up=0.9, down=0.4))
        rows = {row["feature"]: row for row in table.iter_rows(named=True)}
        self.assertAlmostEqual(rows["up"]["stability"], 0.9)
        self.assertAlmostEqual(rows["down"]["stability"], 0.4)


class TestBestPractices(unittest.TestCase):
    def test_statements_state_the_direction(self) -> None:
        table = best_practices(rising_equation(), columns())
        statements = {row["feature"]: row["practice"] for row in table.iter_rows(named=True)}
        self.assertIn("higher MCC", statements["up"])
        self.assertIn("lower MCC", statements["down"])

    def test_unstable_terms_are_dropped(self) -> None:
        table = best_practices(rising_equation(), columns(), stability_table(up=0.95, down=0.10))
        self.assertEqual(table["feature"].to_list(), ["up"])

    def test_tiny_effects_are_dropped(self) -> None:
        tiny = Equation(
            intercept=0.5,
            terms=(Term("atom", (Atom("up"),)),),
            weights=(1e-6,),
            standardized_weights=(0.9,),
        )
        self.assertEqual(best_practices(tiny, columns()).height, 0)

    def test_non_monotone_features_are_dropped(self) -> None:
        # A symmetric V: large contribution range, no direction anyone could state.
        values = np.concatenate([np.linspace(-3.0, 3.0, 60)])
        equation = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("v", "sq"),)),),
            weights=(0.2,),
            standardized_weights=(0.8,),
        )
        table = best_practices(equation, {"v": values})
        self.assertEqual(table.height, 0)

    def test_confidence_rises_with_evidence(self) -> None:
        strong = best_practices(rising_equation(), columns(), stability_table(up=0.95, down=0.95))
        levels = set(strong["confidence"].to_list())
        self.assertTrue(levels <= {"strong", "moderate", "weak"})
        self.assertIn("strong", levels)

    def test_confidence_is_unrated_without_stability(self) -> None:
        table = best_practices(rising_equation(), columns())
        self.assertEqual(set(table["confidence"].to_list()), {"unrated"})

    def test_empty_equation_yields_no_practices(self) -> None:
        empty = Equation(intercept=0.5, terms=(), weights=(), standardized_weights=())
        table = best_practices(empty, columns())
        self.assertEqual(table.height, 0)
        self.assertIn("practice", table.columns)


class TestRender(unittest.TestCase):
    def test_numbered_output(self) -> None:
        text = render(best_practices(rising_equation(), columns()))
        self.assertIn(" 1. ", text)
        self.assertEqual(len(text.splitlines()), 2)

    def test_empty_table_says_so(self) -> None:
        empty = Equation(intercept=0.5, terms=(), weights=(), standardized_weights=())
        self.assertIn("No practice", render(best_practices(empty, columns())))


if __name__ == "__main__":
    unittest.main()
