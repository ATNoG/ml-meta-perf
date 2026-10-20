"""The mathematical statements of chapter 7, checked numerically.

A statement in the documentation that no test can fail is an assertion, not a result. Each test
below names the section it backs.
"""

import unittest

import numpy as np

from ml_meta_perf.data import flexfl_schema
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import _clip_to_training  # pyright: ignore[reportPrivateUsage]
from tests.study import SCHEMA


def indicator_columns(n: int = 40, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"x": rng.uniform(1.0, 5.0, n), "d": np.tile([0.0, 1.0], n // 2)}


class TestBoundsTable(unittest.TestCase):
    """Section 7.1: the bounds of each target."""

    def test_each_row_of_the_table(self) -> None:
        self.assertEqual(SCHEMA.bounds, (-1.0, 1.0))
        self.assertEqual(flexfl_schema("performance", "classification").bounds, (-1.0, 1.0))
        self.assertEqual(flexfl_schema("performance", "regression").bounds, (0.0, 2.0))
        for cost in ("total_time_s", "comm_bytes_total", "compute_time_total_s", "n_epochs"):
            self.assertIsNone(flexfl_schema(cost).bounds, cost)


class TestFoldRange(unittest.TestCase):
    """Section 7.2: the training-range clip, and its absence for an unbounded target."""

    def test_a_clipped_prediction_cannot_reach_a_value_above_the_training_maximum(self) -> None:
        rng = np.random.default_rng(1)
        for _ in range(50):
            training = rng.uniform(0.0, 10.0, 12)
            held_out = training.max() + rng.uniform(0.1, 5.0, 4)
            predicted = held_out + rng.normal(0.0, 3.0, 4)
            clipped = _clip_to_training(predicted, training, (-1.0, 1.0))
            self.assertTrue(np.all(clipped <= training.max()))
            # error >= y - M_g, whatever the equation said
            self.assertTrue(np.all(np.abs(clipped - held_out) >= held_out - training.max() - 1e-12))

    def test_without_bounds_the_prediction_is_returned_as_fitted(self) -> None:
        predicted = np.array([-3.0, 0.5, 40.0])
        np.testing.assert_array_equal(_clip_to_training(predicted, np.array([0.0, 1.0]), None), predicted)

    def test_the_fold_range_uses_the_training_rows_only(self) -> None:
        clipped = _clip_to_training(np.array([-9.0, 9.0]), np.array([2.0, 3.0, 4.0]), (-1.0, 1.0))
        np.testing.assert_array_equal(clipped, [2.0, 4.0])


class TestIndicatorColumns(unittest.TestCase):
    """Section 7.3: why a 0/1 column is an offset and not an independent feature."""

    def setUp(self) -> None:
        self.columns = indicator_columns()
        self.library = build_library(("x",), ("d",), self.columns, max_arity=2)

    def test_d_squared_and_sqrt_d_are_d(self) -> None:
        d = self.columns["d"]
        np.testing.assert_array_equal(d**2, d)
        np.testing.assert_array_equal(np.sqrt(d), d)

    def test_the_grammar_keeps_one_column_for_an_indicator(self) -> None:
        alone = [name for name in self.library.names if name.count("[") == 0 and "d" in name.replace("x", "")]
        self.assertEqual(alone, ["d"])
        for name in self.library.names:
            self.assertNotIn("log(d)", name)
            self.assertNotIn("1/d", name)

    def test_inside_a_product_the_indicator_switches_the_term_off(self) -> None:
        products = [term for term in self.library.terms if "d" in term.features and "x" in term.features]
        self.assertTrue(products)
        for term in products:
            values = term.evaluate(self.columns)
            np.testing.assert_array_equal(values[self.columns["d"] == 0.0], 0.0)

    def test_a_complete_one_hot_set_is_collinear_with_the_intercept(self) -> None:
        group = np.tile([0, 1, 2, 3], 10)
        one_hot = np.stack([(group == k).astype(float) for k in range(4)], axis=1)
        np.testing.assert_array_equal(one_hot.sum(axis=1), 1.0)
        full = np.column_stack([np.ones(len(group)), one_hot])
        self.assertEqual(np.linalg.matrix_rank(full), 4)
        self.assertEqual(full.shape[1], 5)

    def test_which_indicator_is_dropped_does_not_change_the_predictions(self) -> None:
        group = np.tile([0, 1, 2, 3], 10)
        rng = np.random.default_rng(3)
        response = np.array([1.0, 4.0, 2.0, 8.0])[group] + rng.normal(0.0, 0.1, len(group))
        one_hot = np.stack([(group == k).astype(float) for k in range(4)], axis=1)
        predictions = []
        for dropped in range(4):
            design = np.column_stack([np.ones(len(group)), np.delete(one_hot, dropped, axis=1)])
            weights = np.linalg.lstsq(design, response, rcond=None)[0]
            predictions.append(design @ weights)
        for other in predictions[1:]:
            np.testing.assert_allclose(other, predictions[0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
