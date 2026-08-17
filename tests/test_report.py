"""The programmatic report.

The claim these tests defend is that the written analysis is *derived*, not authored:
the same equation must always produce the same sentences, and every sentence must be
traceable to a number in the importance table.
"""

import unittest

import numpy as np
import polars as pl

from metafit.model import Equation
from metafit.report import (
    MAJOR_MASS,
    coverage,
    glossary,
    group_sentences,
    marginal_versus_conditional,
    term_importance,
    term_sentences,
    unstable_majors,
)
from metafit.terms import Atom, Term

DATASET = ("a", "b")
MODEL = ("m",)


def _columns(rows: int = 40) -> dict[str, np.ndarray]:
    """Three features on deliberately unrelated scales.

    ``m`` runs a hundred thousand times larger than ``a``, which is the situation the
    real meta-dataset is in (``gravity`` reaches 1e16 while ``nr_cor_attr`` stays under
    1). It is what makes raw weights incomparable and standardised ones necessary.
    """
    generator = np.random.default_rng(0)
    return {
        "a": generator.uniform(1.0, 5.0, rows),
        "b": generator.uniform(1.0, 5.0, rows),
        "m": generator.uniform(1e5, 5e5, rows),
    }


_WEIGHTS = (0.2, -3e-7, 1e-6)


def _equation(columns: dict[str, np.ndarray] | None = None) -> Equation:
    """The fixture equation, with standardised weights derived rather than invented.

    ``beta = weight * sd(term)`` is the relationship the fitter maintains, so building
    the fixture any other way would test the report against an equation the fitter could
    never produce.
    """
    columns = columns if columns is not None else _columns()
    terms = (
        Term("atom", (Atom("a"),)),
        Term("product", (Atom("b"), Atom("m"))),
        Term("atom", (Atom("m"),)),
    )
    betas = tuple(
        weight * float(term.evaluate(columns).std())
        for term, weight in zip(terms, _WEIGHTS, strict=True)
    )
    return Equation(
        intercept=0.5,
        terms=terms,
        weights=_WEIGHTS,
        standardized_weights=betas,
        name="test",
    )


class TestTermImportance(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = _columns()
        self.table = term_importance(_equation(), self.columns, DATASET, MODEL)

    def test_one_row_per_term_ranked_by_standardised_weight(self) -> None:
        self.assertEqual(self.table.height, 3)
        self.assertEqual(self.table["rank"].to_list(), [1, 2, 3])
        magnitudes = self.table["beta"].abs().to_list()
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_shares_sum_to_one_and_cumulate(self) -> None:
        self.assertAlmostEqual(float(self.table["share"].sum()), 1.0, places=10)
        self.assertAlmostEqual(float(self.table["cumulative"][-1]), 1.0, places=10)

    def test_raw_weight_ranking_would_disagree(self) -> None:
        # The reason the ranking uses beta rather than the raw weight: raw weights carry
        # the units of whatever the term computes. Here the largest raw weight and the
        # largest beta happen to coincide, but the *second* place does not.
        by_weight = self.table.sort(pl.col("weight").abs(), descending=True)["term"].to_list()
        self.assertNotEqual(by_weight, self.table["term"].to_list())

    def test_major_terms_reach_the_mass_threshold(self) -> None:
        major = self.table.filter(pl.col("major"))
        self.assertGreaterEqual(float(major["share"].sum()), MAJOR_MASS)
        # And dropping the last one would fall short, so nothing is flagged needlessly.
        self.assertLess(float(major["share"].sum()) - float(major["share"][-1]), MAJOR_MASS)

    def test_groups_are_labelled_from_the_features_used(self) -> None:
        labels = dict(zip(self.table["term"].to_list(), self.table["group"].to_list(), strict=True))
        self.assertEqual(labels["a"], "dataset")
        self.assertEqual(labels["m"], "model")
        self.assertEqual(labels["[b] * [m]"], "mixed")

    def test_stability_is_carried_through_when_supplied(self) -> None:
        stability = pl.DataFrame({"term": ["a"], "frequency": [0.75]})
        table = term_importance(_equation(), self.columns, DATASET, MODEL, stability)
        row = table.filter(pl.col("term") == "a")
        self.assertAlmostEqual(float(row["stability"][0]), 0.75)
        # A term the stability table never mentions is unrated rather than zero -- zero
        # would read as "never selected", which is a different and false claim.
        self.assertTrue(np.isnan(float(table.filter(pl.col("term") == "m")["stability"][0])))

    def test_empty_equation_yields_a_schema_not_a_crash(self) -> None:
        empty = Equation(intercept=0.0, terms=(), weights=(), standardized_weights=())
        table = term_importance(empty, self.columns, DATASET, MODEL)
        self.assertEqual(table.height, 0)
        self.assertIn("major", table.columns)
        self.assertEqual(coverage(table)["n_terms"], 0)


class TestSentences(unittest.TestCase):
    def setUp(self) -> None:
        self.table = term_importance(_equation(), _columns(), DATASET, MODEL)

    def test_one_sentence_per_major_term(self) -> None:
        sentences = term_sentences(self.table)
        self.assertEqual(len(sentences), int(self.table["major"].sum()))

    def test_sentences_are_deterministic(self) -> None:
        self.assertEqual(term_sentences(self.table), term_sentences(self.table))

    def test_direction_follows_the_sign_of_the_weight(self) -> None:
        sentences = term_sentences(self.table)
        self.assertIn("raises MCC", sentences[0])
        self.assertIn("lowers MCC", sentences[1])

    def test_unmeasured_stability_is_said_so_rather_than_invented(self) -> None:
        self.assertIn("fold agreement not measured", term_sentences(self.table)[0])

    def test_group_sentences_pluralise_and_cover_every_group(self) -> None:
        shares = pl.DataFrame(
            {
                "group": ["dataset", "model", "mixed"],
                "n_terms": [1, 0, 2],
                "share": [0.5, 0.0, 0.5],
                "effect_sum": [0.1, 0.0, 0.2],
            }
        )
        sentences = group_sentences(shares)
        self.assertEqual(len(sentences), 3)
        self.assertIn("1 term,", sentences[0])
        self.assertIn("2 terms,", sentences[2])

    def test_empty_equation_produces_no_sentences(self) -> None:
        empty = Equation(intercept=0.0, terms=(), weights=(), standardized_weights=())
        self.assertEqual(term_sentences(term_importance(empty, _columns(), DATASET, MODEL)), [])


class TestCoverage(unittest.TestCase):
    def test_reports_concentration(self) -> None:
        table = term_importance(_equation(), _columns(), DATASET, MODEL)
        summary = coverage(table)
        self.assertEqual(summary["n_terms"], 3)
        self.assertLessEqual(summary["n_major"], 3)
        self.assertGreaterEqual(summary["major_share"], MAJOR_MASS)
        self.assertAlmostEqual(float(summary["top_share"]), float(table["share"][0]))

    def test_effective_terms_equals_the_count_when_weights_are_equal(self) -> None:
        equal = pl.DataFrame({"share": [0.25] * 4, "major": [True] * 4})
        self.assertAlmostEqual(float(coverage(equal)["effective_terms"]), 4.0)

    def test_effective_terms_falls_toward_one_when_a_term_dominates(self) -> None:
        skewed = pl.DataFrame({"share": [0.97, 0.01, 0.01, 0.01], "major": [True] * 4})
        self.assertLess(float(coverage(skewed)["effective_terms"]), 1.1)

    def test_effective_terms_never_exceeds_the_term_count(self) -> None:
        table = term_importance(_equation(), _columns(), DATASET, MODEL)
        summary = coverage(table)
        self.assertLessEqual(float(summary["effective_terms"]), float(summary["n_terms"]) + 1e-9)


class TestUnstableMajors(unittest.TestCase):
    def _table(self, frequencies: list[float]) -> pl.DataFrame:
        columns = _columns()
        stability = pl.DataFrame(
            {"term": ["a", "[b] * [m]", "m"], "frequency": frequencies}
        )
        return term_importance(_equation(columns), columns, DATASET, MODEL, stability)

    def test_flags_terms_with_a_large_weight_and_few_folds(self) -> None:
        flagged = unstable_majors(self._table([0.2, 0.9, 0.9]))
        self.assertEqual(flagged["term"].to_list(), ["a"])

    def test_reports_nothing_when_every_major_term_is_stable(self) -> None:
        self.assertEqual(unstable_majors(self._table([0.9, 0.9, 0.9])).height, 0)

    def test_unrated_terms_are_not_flagged(self) -> None:
        # NaN means "not measured", which is not evidence of instability.
        table = term_importance(_equation(), _columns(), DATASET, MODEL)
        self.assertEqual(unstable_majors(table).height, 0)

    def test_empty_equation_is_handled(self) -> None:
        empty = Equation(intercept=0.0, terms=(), weights=(), standardized_weights=())
        self.assertEqual(unstable_majors(term_importance(empty, _columns(), DATASET, MODEL)).height, 0)


class TestMarginalVersusConditional(unittest.TestCase):
    def _practices(self, direction: float) -> pl.DataFrame:
        return pl.DataFrame(
            {"feature": ["a"], "meaning": ["feature a"], "direction": [direction]}
        )

    def test_flags_agreement_when_the_signs_match(self) -> None:
        columns = _columns()
        truth = columns["a"] * 2.0
        table = marginal_versus_conditional(self._practices(0.9), columns, truth)
        self.assertGreater(float(table["marginal"][0]), 0.99)
        self.assertTrue(bool(table["agrees"][0]))

    def test_flags_disagreement_when_conditioning_flips_the_sign(self) -> None:
        # The case the section exists for: a feature positively correlated with MCC on
        # its own, whose equation contribution falls as it rises.
        columns = _columns()
        truth = columns["a"] * 2.0
        table = marginal_versus_conditional(self._practices(-0.9), columns, truth)
        self.assertFalse(bool(table["agrees"][0]))
        self.assertEqual(table["practice_says"][0], "lower")

    def test_empty_practices_yield_a_schema(self) -> None:
        table = marginal_versus_conditional(pl.DataFrame(), _columns(), np.zeros(40))
        self.assertEqual(table.height, 0)
        self.assertIn("agrees", table.columns)


class TestGlossary(unittest.TestCase):
    def test_every_feature_is_explained(self) -> None:
        from metafit.data import ALL_FEATURES

        table = glossary()
        self.assertEqual(set(table["feature"].to_list()), set(ALL_FEATURES))


if __name__ == "__main__":
    unittest.main()
