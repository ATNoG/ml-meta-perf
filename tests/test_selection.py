"""Tests for the length rule (`plateau_knee`) and the E3-MAX bound (`floor_argmax`)."""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.selection import (
    complexity,
    floor_argmax,
    floor_curve,
    knee_lengths,
    plateau_knee,
    plateau_start,
    protocol_spread,
    smoothed,
)

RULE = {"delta": 0.01, "window": 4, "smoothing": 3}


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


def saturating(horizon: int = 30, scale: float = 4.0) -> np.ndarray:
    """A curve that climbs fast and levels off, the shape every equation's floor has."""
    return 0.65 * (1.0 - np.exp(-np.arange(1, horizon + 1) / scale))


def as_curve(floor: np.ndarray) -> pl.DataFrame:
    """A curve whose four protocols are all ``floor``, so the floor is exactly ``floor``."""
    values = [float(value) for value in floor]
    return curve(list(range(1, len(values) + 1)), values, values, values, values)


class TestFloorCurve(unittest.TestCase):
    """A length scores as its worst protocol, over all four including DHO."""

    def test_it_takes_the_minimum_over_every_protocol_present(self) -> None:
        table = curve([4], [0.70], [0.65], [0.60], [0.55])
        self.assertAlmostEqual(float(floor_curve(table)[0]), 0.55)

    def test_the_cell_protocol_is_read_where_a_median_would_hide_it(self) -> None:
        """Three protocols agree at 0.66 and the cell reads 0.50: a median of the four reports
        a number no protocol achieved, the minimum reports the cell."""
        table = curve([4], [0.66], [0.66], [0.66], [0.50])
        self.assertAlmostEqual(float(floor_curve(table)[0]), 0.50)
        self.assertGreater(float(np.median([0.66, 0.66, 0.66, 0.50])), 0.60)

    def test_it_works_on_a_curve_that_has_not_been_scored_on_every_protocol(self) -> None:
        np.testing.assert_allclose(floor_curve(curve([2, 4], [0.5, 0.6], [0.4, 0.55])), [0.4, 0.55])

    def test_it_refuses_a_curve_with_no_protocol_columns(self) -> None:
        with self.assertRaises(ValueError):
            floor_curve(pl.DataFrame({"n_terms": [2, 4]}))


class TestSmoothing(unittest.TestCase):
    """A running median: no single length can move the curve the rule reads."""

    def test_a_single_length_crater_is_ignored(self) -> None:
        values = np.array([0.50, 0.55, 0.20, 0.60, 0.62])
        self.assertAlmostEqual(float(smoothed(values, 3)[2]), 0.55)

    def test_a_single_lucky_length_is_ignored(self) -> None:
        values = np.array([0.60, 0.61, 0.70, 0.61, 0.62])
        self.assertAlmostEqual(float(smoothed(values, 3)[2]), 0.61)

    def test_the_window_shrinks_at_the_ends_and_width_one_is_the_identity(self) -> None:
        values = np.array([0.1, 0.4, 0.2])
        self.assertAlmostEqual(float(smoothed(values, 3)[0]), 0.25)
        np.testing.assert_allclose(smoothed(values, 1), values)

    def test_it_refuses_a_non_positive_width(self) -> None:
        with self.assertRaises(ValueError):
            smoothed(np.array([0.1]), 0)


class TestKneeLengths(unittest.TestCase):
    """multi-Kneedle proposes lengths where the smoothed front bends."""

    def test_a_saturating_curve_has_knees_on_its_rising_part(self) -> None:
        knees = knee_lengths(as_curve(saturating()), 3)
        self.assertTrue(knees, "a saturating curve bends somewhere")
        self.assertEqual(knees, sorted(knees))
        self.assertTrue(all(1 <= knee <= 30 for knee in knees))

    def test_a_flat_curve_has_no_front_and_no_knee(self) -> None:
        self.assertEqual(knee_lengths(as_curve(np.full(12, 0.5)), 3), [])


class TestPlateauKnee(unittest.TestCase):
    """The length rule: the first knee (or the maximum) after which the curve stops improving."""

    def test_it_stops_where_a_saturating_curve_levels_off(self) -> None:
        floor = saturating()
        size = plateau_knee(as_curve(floor), **RULE)
        smooth = smoothed(floor, 3)
        ahead = smooth[size : size + RULE["window"]]
        self.assertLessEqual(float(ahead.max() - smooth[size - 1]), RULE["delta"] + 1e-12)
        self.assertLess(size, 30, "a curve that has levelled off is not read to its horizon")
        shorter = smooth[: size - 1]
        self.assertTrue(
            any(
                float(smooth[i + 1 : i + 1 + RULE["window"]].max() - smooth[i]) > RULE["delta"]
                for i in range(len(shorter))
            )
            or size <= RULE["window"],
            "every shorter length is still followed by a real gain",
        )

    def test_a_lucky_length_near_the_horizon_does_not_choose_the_length(self) -> None:
        """The spike is never chosen. It can still nudge the knees by one length: Kneedle
        normalises over the whole Pareto front, and the spike moves the front's far end."""
        floor = saturating()
        spiked = floor.copy()
        spiked[27] += 0.05
        chosen = plateau_knee(as_curve(spiked), **RULE)
        self.assertNotEqual(chosen, 28)
        self.assertLessEqual(abs(chosen - plateau_knee(as_curve(floor), **RULE)), 1)

    def test_a_crater_does_not_block_a_plateau(self) -> None:
        floor = saturating()
        size = plateau_knee(as_curve(floor), **RULE)
        cratered = floor.copy()
        cratered[size + 1] -= 0.2
        self.assertEqual(plateau_knee(as_curve(cratered), **RULE), size)

    def test_a_curve_still_climbing_returns_its_maximum(self) -> None:
        floor = np.linspace(0.2, 0.7, 20)
        size, how = plateau_start(as_curve(floor), **RULE)
        self.assertEqual((size, how), (20, "maximum"))

    def test_a_peak_before_a_decline_is_the_plateau_start(self) -> None:
        floor = np.concatenate([np.linspace(0.3, 0.6, 12), np.linspace(0.59, 0.5, 10)])
        size, how = plateau_start(as_curve(floor), **RULE)
        self.assertEqual(how, "maximum")
        self.assertEqual(size, 12)

    def test_it_is_deterministic(self) -> None:
        table = as_curve(saturating())
        self.assertEqual({plateau_knee(table, **RULE) for _ in range(5)}, {plateau_knee(table, **RULE)})

    def test_it_refuses_a_negative_delta_or_an_empty_window(self) -> None:
        table = as_curve(saturating())
        with self.assertRaises(ValueError):
            plateau_knee(table, delta=-0.1, window=4, smoothing=3)
        with self.assertRaises(ValueError):
            plateau_knee(table, delta=0.01, window=0, smoothing=3)


class TestFloorArgmax(unittest.TestCase):
    """E3-MAX: the raw maximum, unsmoothed -- a bound, not an equation to read."""

    def test_it_takes_the_raw_maximum_even_on_a_single_length(self) -> None:
        floor = saturating()
        floor[27] += 0.05
        self.assertEqual(floor_argmax(as_curve(floor)), 28)

    def test_ties_go_to_the_shorter_equation(self) -> None:
        self.assertEqual(floor_argmax(as_curve(np.array([0.5, 0.6, 0.6]))), 2)


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


class TestComplexity(unittest.TestCase):
    def test_complexity_charges_for_the_grammar(self) -> None:
        self.assertEqual(complexity(2, 15), 30)
        self.assertEqual(complexity(3, 15), 45)


if __name__ == "__main__":
    unittest.main()
