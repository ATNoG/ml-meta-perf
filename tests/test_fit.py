"""Fitting weights to a form already chosen: standardising, solving, and reading back out.

The search that chooses the form is `tests/test_search.py`.
"""

import unittest

import numpy as np

from ml_meta_perf.fit import Standardizer, Subset, ridge_solve, to_equation
from ml_meta_perf.terms import build_library


def synthetic_columns(n: int = 120, seed: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.uniform(1.0, 20.0, n) for name in ("f1", "f2", "f3", "f4")}


class TestStandardizer(unittest.TestCase):
    def test_produces_zero_mean_unit_scale(self) -> None:
        matrix = np.random.default_rng(0).normal(5.0, 3.0, (50, 4))
        applied = Standardizer.fit(matrix).apply(matrix)
        np.testing.assert_allclose(applied.mean(axis=0), np.zeros(4), atol=1e-12)
        np.testing.assert_allclose(applied.std(axis=0), np.ones(4), atol=1e-12)

    def test_constant_column_does_not_divide_by_zero(self) -> None:
        matrix = np.column_stack([np.ones(10), np.arange(10.0)])
        applied = Standardizer.fit(matrix).apply(matrix)
        self.assertTrue(np.all(np.isfinite(applied)))

    def test_scale_learned_on_train_is_reused_on_test(self) -> None:
        train = np.arange(20.0).reshape(10, 2)
        standardizer = Standardizer.fit(train)
        test = np.array([[100.0, 100.0]])
        np.testing.assert_allclose(
            standardizer.apply(test), (test - standardizer.mean) / standardizer.scale
        )


class TestRidgeSolve(unittest.TestCase):
    def test_zero_penalty_reproduces_least_squares(self) -> None:
        rng = np.random.default_rng(1)
        design = rng.normal(size=(40, 3))
        truth = rng.normal(size=40)
        expected = np.linalg.lstsq(design, truth, rcond=None)[0]
        np.testing.assert_allclose(ridge_solve(design.T @ design, design.T @ truth, 0.0), expected, atol=1e-8)

    def test_penalty_shrinks_the_weights(self) -> None:
        rng = np.random.default_rng(2)
        design = rng.normal(size=(40, 3))
        truth = rng.normal(size=40)
        gram, projection = design.T @ design, design.T @ truth
        small = np.abs(ridge_solve(gram, projection, 0.0)).sum()
        large = np.abs(ridge_solve(gram, projection, 500.0)).sum()
        self.assertLess(large, small)

    def test_singular_system_still_returns_a_solution(self) -> None:
        column = np.arange(10.0)
        design = np.column_stack([column, column])
        weights = ridge_solve(design.T @ design, design.T @ column, 0.0)
        self.assertTrue(np.all(np.isfinite(weights)))


class TestToEquation(unittest.TestCase):
    def test_raw_weights_reproduce_the_standardised_prediction(self) -> None:
        """The round trip the published equation depends on: fitted on standardised columns,
        printed in raw units, and the two must predict the same thing.

        The subset is solved here rather than searched for. What `to_equation` folds back out
        is a set of indices and the weights beside them; where they came from is not its
        business, and reaching into `search` for a realistic-looking one only coupled this
        test to a beam it never asserts about.
        """
        columns = synthetic_columns(90, seed=11)
        library = build_library(("f1", "f2"), ("f3",), columns)
        target = 0.3 * np.log(columns["f1"]) + 0.1 * columns["f3"]

        standardizer = Standardizer.fit(library.matrix)
        design = standardizer.apply(library.matrix)
        offset = float(target.mean())
        indices = (0, 3, 7)
        block = design[:, list(indices)]
        weights = ridge_solve(block.T @ block, block.T @ (target - offset), 1.0)
        subset = Subset(indices, weights, 0.0)

        equation = to_equation(library, subset, standardizer, offset, "check")
        np.testing.assert_allclose(equation.evaluate(columns), offset + block @ weights, rtol=1e-6, atol=1e-9)

    def test_subset_dataclass_carries_its_fit(self) -> None:
        subset = Subset((1, 2), np.array([0.5, -0.5]), 3.0)
        self.assertEqual(subset.indices, (1, 2))
        self.assertEqual(subset.rss, 3.0)


if __name__ == "__main__":
    unittest.main()
