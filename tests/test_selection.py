"""Tests for the retained E3-Valid plateau rule and the independent E3-MAX bound."""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.selection import (
    arity_candidates,
    complexity,
    consensus_curve,
    floor_curve,
    most_capable,
    plateau_configuration,
    plateau_index,
    protocol_spread,
)


def curve(
    sizes: list[int],
    in_sample: list[float],
    loo: list[float] | None = None,
    loo_model: list[float] | None = None,
    loo_cell: list[float] | None = None,
) -> pl.DataFrame:
    data: dict[str, list[float] | list[int]] = {"n_terms": sizes, "r2_in_sample": in_sample}
    if loo is not None:
        data["r2_loo_dataset"] = loo
    if loo_model is not None:
        data["r2_loo_model"] = loo_model
    if loo_cell is not None:
        data["r2_loo_cell"] = loo_cell
    return pl.DataFrame(data)


class TestConsensusCurve(unittest.TestCase):
    """One score per length across the protocols, so no single curve chooses the length."""

    def setUp(self) -> None:
        self.table = curve(
            [2, 4, 6, 8],
            [0.40, 0.50, 0.60, 0.66],
            [0.30, 0.45, 0.10, 0.62],  # 0.10 is a crater, as loo-dataset genuinely has
            [0.35, 0.47, 0.58, 0.60],
        )

    def test_the_median_ignores_a_single_protocol_crater(self) -> None:
        """Why median and not mean.

        At the cratered length the three protocols read 0.60 / 0.10 / 0.58. The median takes
        0.58 -- one fold extrapolating outside the training hull is a property of that fold,
        not of the length -- while the mean is dragged down by it.
        """
        median = consensus_curve(self.table, "median")
        mean = consensus_curve(self.table, "mean")
        self.assertAlmostEqual(float(median[2]), 0.58)
        self.assertAlmostEqual(float(mean[2]), (0.60 + 0.10 + 0.58) / 3)
        self.assertGreater(float(median[2]), float(mean[2]))

    def test_min_is_the_conservative_reading(self) -> None:
        self.assertAlmostEqual(float(consensus_curve(self.table, "min")[2]), 0.10)

    def test_a_single_protocol_is_its_own_consensus(self) -> None:
        plain = curve([2, 4, 6], [0.2, 0.5, 0.6])
        np.testing.assert_allclose(consensus_curve(plain), [0.2, 0.5, 0.6])

    def test_it_refuses_a_curve_with_no_protocol_columns(self) -> None:
        with self.assertRaises(ValueError):
            consensus_curve(pl.DataFrame({"n_terms": [2, 4]}))


class TestFloorCurve(unittest.TestCase):
    """A length scores as its worst protocol, over all four including the cell protocol."""

    def test_it_takes_the_minimum_over_every_protocol_present(self) -> None:
        table = curve([4], [0.70], [0.65], [0.60], [0.55])
        self.assertAlmostEqual(float(floor_curve(table)[0]), 0.55)

    def test_the_cell_protocol_is_read_and_the_median_would_have_hidden_it(self) -> None:
        """The point of the minimum. Three protocols agree at 0.66 and the cell reads 0.50;
        a median of the four sits at 0.66 and reports a number no protocol achieved."""
        table = curve([4], [0.66], [0.66], [0.66], [0.50])
        self.assertAlmostEqual(float(floor_curve(table)[0]), 0.50)
        self.assertGreater(float(consensus_curve(table)[0]), 0.60)

    def test_it_works_on_a_curve_that_has_not_been_scored_on_every_protocol(self) -> None:
        np.testing.assert_allclose(floor_curve(curve([2, 4], [0.5, 0.6], [0.4, 0.55])), [0.4, 0.55])

    def test_it_refuses_a_curve_with_no_protocol_columns(self) -> None:
        with self.assertRaises(ValueError):
            floor_curve(pl.DataFrame({"n_terms": [2, 4]}))


class TestProtocolSpread(unittest.TestCase):
    """How far a length falls from its fit to its worst protocol."""

    def test_it_measures_the_drop_from_the_fit_to_the_floor(self) -> None:
        table = curve([4], [0.70], [0.65], [0.63], [0.60])
        self.assertAlmostEqual(float(protocol_spread(table)[0]), 0.10)

    def test_the_floor_alone_cannot_tell_two_equations_apart_and_this_can(self) -> None:
        table = curve([4, 8], [0.66, 0.80], [0.64, 0.70], [0.63, 0.68], [0.60, 0.60])
        np.testing.assert_allclose(floor_curve(table), [0.60, 0.60])
        np.testing.assert_allclose(protocol_spread(table), [0.06, 0.20])

    def test_it_refuses_a_curve_with_nothing_to_measure_the_drop_from(self) -> None:
        with self.assertRaises(ValueError):
            protocol_spread(pl.DataFrame({"n_terms": [2], "r2_loo_dataset": [0.5]}))


class TestE3Selection(unittest.TestCase):
    """The sole E3-Valid plateau rule and the independent E3-MAX bound."""

    def test_plateau_index_stops_before_a_sustained_stall(self) -> None:
        scores = np.array([0.2, 0.4, 0.6, 0.6002, 0.6004, 0.7])
        self.assertEqual(plateau_index(scores, tolerance=0.001, window=2), 2)

    def test_plateau_selection_compares_both_arities(self) -> None:
        arity_two = [0.30, 0.50, 0.60, 0.6002, 0.6003]
        arity_three = [0.31, 0.49, 0.59, 0.6001, 0.6003]
        curves = {
            2: curve([1, 2, 3, 4, 5], arity_two, arity_two, arity_two),
            3: curve([1, 2, 3, 4, 5], arity_three, arity_three, arity_three),
        }
        self.assertEqual(plateau_configuration(curves, tolerance=0.001, window=2), (2, 3))

    def test_most_capable_uses_the_four_protocol_floor(self) -> None:
        curves = {
            2: curve([5, 10], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60], [0.50, 0.60]),
            3: curve([5, 28], [0.50, 0.65], [0.50, 0.65], [0.50, 0.65], [0.50, 0.65]),
        }
        self.assertEqual(arity_candidates(curves), {2: 10, 3: 28})
        self.assertEqual(most_capable(curves), (3, 28))

    def test_complexity_charges_for_the_grammar(self) -> None:
        self.assertEqual(complexity(2, 15), 30)
        self.assertEqual(complexity(3, 15), 45)


if __name__ == "__main__":
    unittest.main()
