"""Figures. These check that each plot writes a valid PNG, not that it looks right."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from ml_meta_perf.plots import (
    contribution_shares,
    count_below_floor,
    equation_comparison,
    error_curve,
    per_group_quality,
    practice_effects,
    predicted_versus_actual,
    scatter_limits,
    term_count_curve,
    term_effects,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def curve() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "n_terms": [2, 4, 6, 8],
            "r2_in_sample": [0.30, 0.42, 0.50, 0.55],
            "r2_loo_dataset": [0.10, 0.25, 0.28, 0.33],
            "mae_loo_dataset": [0.28, 0.24, 0.22, 0.21],
            "r2_loo_model": [0.20, 0.35, 0.40, 0.44],
            "mae_loo_model": [0.26, 0.22, 0.20, 0.19],
        }
    )


def effects() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "term": ["[log(a)] * [log(b)]", "c", "[d] / [log(e)]"],
            "group": ["dataset", "model", "mixed"],
            "weight": [-0.13, 0.02, 0.09],
            "beta": [-0.17, 0.09, 0.06],
            "effect": [0.31, 0.22, 0.14],
            "direction": ["decreases MCC", "increases MCC", "increases MCC"],
        }
    )


def practices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["gravity", "nr_attr", "Robust to Outliers"],
            "effect": [-0.42, 0.23, 0.13],
            "confidence": ["strong", "moderate", "strong"],
        }
    )


class PlotTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.folder = Path(self._directory.name)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def assertIsPng(self, path: Path) -> None:
        """Both files, because every figure is written as a raster and a vector."""
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)
        with path.open("rb") as handle:
            self.assertEqual(handle.read(8), PNG_MAGIC)

        vector = path.with_suffix(".pdf")
        self.assertTrue(vector.is_file(), f"no PDF beside {path.name}")
        with vector.open("rb") as handle:
            self.assertEqual(handle.read(4), b"%PDF")

        # Transparent background, checked at a corner rather than asserted from the
        # savefig argument -- the argument could be right and the axes patch still opaque.
        import matplotlib.image as mpimg

        pixels = mpimg.imread(path)
        self.assertEqual(pixels.shape[2], 4, "no alpha channel")
        self.assertEqual(float(pixels[0, 0, 3]), 0.0, "corner pixel is not transparent")


class TestPlots(PlotTestCase):
    def test_term_count_curve(self) -> None:
        self.assertIsPng(term_count_curve(curve(), self.folder / "curve.png", oracle=0.661))

    def test_term_count_curve_without_an_oracle(self) -> None:
        self.assertIsPng(term_count_curve(curve(), self.folder / "plain.png"))

    def test_term_count_curve_with_only_one_protocol(self) -> None:
        partial = curve().drop("r2_loo_model", "mae_loo_model")
        self.assertIsPng(term_count_curve(partial, self.folder / "partial.png"))

    def test_predicted_versus_actual(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.uniform(-1.0, 1.0, 80)
        self.assertIsPng(
            predicted_versus_actual(truth, truth + rng.normal(0, 0.1, 80), self.folder / "scatter.png")
        )

    def test_scatter_with_a_rug(self) -> None:
        truth = np.linspace(0.0, 1.0, 40)
        self.assertIsPng(
            predicted_versus_actual(truth, truth * 0.9, self.folder / "rug.png", groups=truth)
        )

    def test_scatter_of_realistic_data_is_still_a_png(self) -> None:
        truth = np.linspace(0.0, 1.0, 40)
        self.assertIsPng(
            predicted_versus_actual(truth, truth * 0.9 + 0.05, self.folder / "tight.png")
        )

    def test_scatter_renders_with_points_below_the_floor(self) -> None:
        # The out-of-range point falls outside the axes; rendering must not crash.
        truth = np.concatenate([np.array([-0.29]), np.linspace(0.0, 1.0, 30)])
        predicted = np.concatenate([np.array([0.5]), np.linspace(0.2, 1.0, 30)])
        self.assertIsPng(predicted_versus_actual(truth, predicted, self.folder / "clipped.png"))


    def test_error_curve_mae(self) -> None:
        self.assertIsPng(error_curve(curve(), self.folder / "mae.png", metric="mae"))

    def test_error_curve_smape_with_a_knee_marker(self) -> None:
        table = curve().with_columns(
            pl.Series("smape_loo_dataset", [60.0, 50.0, 45.0, 42.0]),
            pl.Series("smape_loo_model", [55.0, 48.0, 44.0, 41.0]),
        )
        self.assertIsPng(error_curve(table, self.folder / "smape.png", metric="smape", marker=6))

    def test_term_effects(self) -> None:
        self.assertIsPng(term_effects(effects(), self.folder / "terms.png"))

    def test_term_effects_respects_the_top_limit(self) -> None:
        self.assertIsPng(term_effects(effects(), self.folder / "top.png", top=2))

    def test_practice_effects(self) -> None:
        self.assertIsPng(practice_effects(practices(), self.folder / "practices.png"))

    def test_contribution_shares(self) -> None:
        shares = pl.DataFrame(
            {"group": ["dataset", "model", "mixed"], "n_terms": [5, 4, 3], "share": [0.52, 0.28, 0.20]}
        )
        self.assertIsPng(contribution_shares(shares, self.folder / "shares.png"))

    def test_equation_comparison(self) -> None:
        table = pl.DataFrame(
            {
                "equation": ["E1 (dataset only)", "E1 ceiling (true dataset means)", "E2 (dataset + model)"],
                "r2": [0.337, 0.354, 0.556],
            }
        )
        self.assertIsPng(equation_comparison(table, self.folder / "comparison.png"))

    def test_per_group_quality(self) -> None:
        table = pl.DataFrame(
            {"group": ["alpha", "beta", "gamma"], "spearman": [0.8, 0.2, 0.6], "regret": [0.0, 0.1, 0.02]}
        )
        self.assertIsPng(per_group_quality(table, self.folder / "quality.png"))

    def test_practice_effects_without_confidence(self) -> None:
        plain = practices().drop("confidence")
        self.assertIsPng(practice_effects(plain, self.folder / "plain_practices.png"))

    def test_creates_missing_directories(self) -> None:
        nested = self.folder / "a" / "b" / "curve.png"
        self.assertIsPng(term_count_curve(curve(), nested))


class TestScatterLimits(unittest.TestCase):
    """Axis bounds follow the data, not MCC's theoretical range."""

    def test_upper_bound_covers_both_series(self) -> None:
        low, high = scatter_limits(np.array([0.0, 0.8]), np.array([0.3, 1.0]), margin=0.05)
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 1.05)

    def test_floor_stops_the_axis_at_zero(self) -> None:
        # The sub-zero region holds only degenerate results with no linear structure;
        # including it compresses the range where the relationship actually lives.
        truth = np.array([-0.29, 0.0, 1.0])
        predicted = np.array([0.17, 0.5, 1.0])
        low, _ = scatter_limits(truth, predicted)
        self.assertEqual(low, 0.0)

    def test_floor_is_configurable(self) -> None:
        truth = np.array([-0.29, 1.0])
        predicted = np.array([0.17, 1.0])
        low, _ = scatter_limits(truth, predicted, floor=-1.0)
        self.assertAlmostEqual(low, -0.34)

    def test_margin_applies_when_it_does_not_cross_the_floor(self) -> None:
        values = np.array([0.2, 0.6])
        low, high = scatter_limits(values, values, margin=0.1)
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(high, 0.7)

    def test_zero_margin_is_exact(self) -> None:
        values = np.array([0.0, 1.0])
        self.assertEqual(scatter_limits(values, values, margin=0.0), (0.0, 1.0))

    def test_counts_points_below_the_floor(self) -> None:
        truth = np.array([-0.29, 0.0, 0.5, 1.0])
        predicted = np.array([0.5, 0.2, 0.6, 1.0])
        self.assertEqual(count_below_floor(truth, predicted), 1)

    def test_counts_zero_when_everything_is_in_range(self) -> None:
        values = np.array([0.0, 0.5, 1.0])
        self.assertEqual(count_below_floor(values, values), 0)

    def test_count_respects_a_custom_floor(self) -> None:
        values = np.array([0.1, 0.5, 1.0])
        self.assertEqual(count_below_floor(values, values, floor=0.4), 1)

    def test_never_returns_an_inverted_range(self) -> None:
        values = np.array([0.1, 0.9])
        low, high = scatter_limits(values, values)
        self.assertLess(low, high)


class TestOracleLookup(unittest.TestCase):
    """The ceiling drawn on the curve must be the one the run computed."""

    def test_reads_the_oracle_from_the_comparison_table(self) -> None:
        from ml_meta_perf.figures import ORACLE_ROW, _oracle

        report = SimpleNamespace(
            comparison=pl.DataFrame({"equation": ["E2 (dataset + model)", ORACLE_ROW], "r2": [0.556, 0.6605]})
        )
        value = _oracle(report)  # pyright: ignore[reportArgumentType]
        assert value is not None
        self.assertAlmostEqual(value, 0.6605)

    def test_returns_none_when_the_row_is_absent(self) -> None:
        from ml_meta_perf.figures import _oracle

        report = SimpleNamespace(comparison=pl.DataFrame({"equation": ["E2"], "r2": [0.5]}))
        self.assertIsNone(_oracle(report))  # pyright: ignore[reportArgumentType]


class TestFigureSet(PlotTestCase):
    def test_generate_writes_the_whole_set(self) -> None:
        from ml_meta_perf.experiment import run
        from ml_meta_perf.figures import generate

        written = generate(run(quick=True), self.folder)
        self.assertEqual(len(written), len(set(written)))
        for path in written:
            self.assertIsPng(path)


if __name__ == "__main__":
    unittest.main()
