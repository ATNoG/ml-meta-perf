"""Term effects, group shares and the identity-based variance decomposition."""

import unittest

import numpy as np

from ml_meta_perf.attribution import (
    DATASET_ONLY,
    MIXED,
    MODEL_ONLY,
    classify,
    contributions,
    group_shares,
    term_effects,
    variance_decomposition,
)
from ml_meta_perf.model import Equation, LOWERS, RAISES
from ml_meta_perf.terms import Atom, Term

DATASET = ("d1", "d2")
MODEL = ("m1", "m2")


def equation() -> Equation:
    return Equation(
        intercept=0.5,
        terms=(
            Term("atom", (Atom("d1"),)),
            Term("atom", (Atom("m1"),)),
            Term("product", (Atom("d2"), Atom("m2"))),
        ),
        weights=(0.10, -0.20, 0.05),
        standardized_weights=(0.3, -0.5, 0.2),
    )


def columns(n: int = 60, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(1.0, 5.0, n) for name in (*DATASET, *MODEL)}


class TestClassify(unittest.TestCase):
    def test_labels_each_group(self) -> None:
        self.assertEqual(classify(("d1",), DATASET, MODEL), DATASET_ONLY)
        self.assertEqual(classify(("m1",), DATASET, MODEL), MODEL_ONLY)
        self.assertEqual(classify(("d1", "m2"), DATASET, MODEL), MIXED)

    def test_unknown_feature_defaults_to_dataset(self) -> None:
        self.assertEqual(classify(("other",), DATASET, MODEL), DATASET_ONLY)


class TestContributions(unittest.TestCase):
    def test_shape_matches_terms(self) -> None:
        data = columns()
        self.assertEqual(contributions(equation(), data).shape, (60, 3))

    def test_rows_sum_to_the_prediction_minus_intercept(self) -> None:
        data = columns()
        eq = equation()
        np.testing.assert_allclose(
            contributions(eq, data).sum(axis=1) + eq.intercept, eq.evaluate(data), rtol=1e-9
        )

    def test_empty_equation_gives_an_empty_matrix(self) -> None:
        empty = Equation(intercept=0.2, terms=(), weights=(), standardized_weights=())
        self.assertEqual(contributions(empty, columns()).shape, (60, 0))


class TestTermEffects(unittest.TestCase):
    def setUp(self) -> None:
        self.table = term_effects(equation(), columns(), DATASET, MODEL)

    def test_one_row_per_term(self) -> None:
        self.assertEqual(self.table.height, 3)

    def test_sorted_by_absolute_effect(self) -> None:
        effects = np.abs(self.table["effect"].to_numpy())
        self.assertTrue((np.diff(effects) <= 1e-12).all())

    def test_effects_are_non_negative_spans(self) -> None:
        self.assertTrue((self.table["effect"].to_numpy() >= 0.0).all())

    def test_direction_follows_the_standardised_weight(self) -> None:
        for row in self.table.iter_rows(named=True):
            expected = RAISES if row["beta"] > 0 else LOWERS
            self.assertEqual(row["direction"], expected)

    def test_groups_are_labelled(self) -> None:
        self.assertEqual(set(self.table["group"].to_list()), {DATASET_ONLY, MODEL_ONLY, MIXED})


class TestGroupShares(unittest.TestCase):
    def test_shares_sum_to_one(self) -> None:
        table = group_shares(equation(), columns(), DATASET, MODEL)
        self.assertAlmostEqual(float(table["share"].sum()), 1.0, places=6)

    def test_every_group_is_present_even_when_unused(self) -> None:
        only_dataset = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("d1"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        table = group_shares(only_dataset, columns(), DATASET, MODEL)
        self.assertEqual(table.height, 3)
        counts = dict(zip(table["group"].to_list(), table["n_terms"].to_list(), strict=True))
        self.assertEqual(counts[MODEL_ONLY], 0)
        self.assertEqual(counts[DATASET_ONLY], 1)

    def test_a_single_group_takes_the_whole_share(self) -> None:
        only_dataset = Equation(
            intercept=0.0,
            terms=(Term("atom", (Atom("d1"),)),),
            weights=(1.0,),
            standardized_weights=(1.0,),
        )
        table = group_shares(only_dataset, columns(), DATASET, MODEL)
        shares = dict(zip(table["group"].to_list(), table["share"].to_list(), strict=True))
        self.assertAlmostEqual(shares[DATASET_ONLY], 1.0, places=6)


class TestVarianceDecomposition(unittest.TestCase):
    def test_perfectly_separated_groups_explain_everything(self) -> None:
        target = np.array([0.1, 0.1, 0.9, 0.9])
        datasets = np.array(["a", "a", "b", "b"])
        models = np.array(["x", "y", "x", "y"])
        table = variance_decomposition(target, datasets, models)
        scores = dict(zip(table["knowing only"].to_list(), table["variance_explained"].to_list(), strict=True))
        self.assertAlmostEqual(scores["dataset identity"], 1.0)
        self.assertAlmostEqual(scores["model identity"], 0.0)

    def test_counts_the_groups(self) -> None:
        table = variance_decomposition(
            np.array([0.1, 0.2, 0.3]), np.array(["a", "b", "c"]), np.array(["x", "x", "y"])
        )
        counts = dict(zip(table["knowing only"].to_list(), table["n_groups"].to_list(), strict=True))
        self.assertEqual(counts["dataset identity"], 3)
        self.assertEqual(counts["model identity"], 2)

    def test_constant_target_is_zero_not_nan(self) -> None:
        table = variance_decomposition(np.ones(4), np.array(["a", "a", "b", "b"]), np.array(["x", "y", "x", "y"]))
        self.assertTrue(np.all(np.isfinite(table["variance_explained"].to_numpy())))

    def test_real_data_puts_dataset_above_model(self) -> None:
        # The empirical answer to "does model choice matter more than the data".
        from ml_meta_perf.data import DATASET_COLUMN, MODEL_COLUMN, groups, load, target

        frame = load()
        table = variance_decomposition(
            target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)
        )
        scores = dict(zip(table["knowing only"].to_list(), table["variance_explained"].to_list(), strict=True))
        self.assertGreater(scores["dataset identity"], scores["model identity"])


if __name__ == "__main__":
    unittest.main()
