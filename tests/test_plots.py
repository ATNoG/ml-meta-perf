"""Figures. These check that each plot writes a valid PNG, not that it looks right."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl

from ml_meta_perf.model import LOWERS, RAISES
from ml_meta_perf.plots import (
    CONFIDENCE_ALPHA,
    _confidence_levels,
    count_below_floor,
    equation_comparison,
    practice_effects,
    predicted_versus_actual,
    scatter_limits,
    term_count_curve,
    term_effects,
    term_to_math,
)
from tests import corpus

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
            "direction": [LOWERS, RAISES, RAISES],
        }
    )


def practices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": ["gravity", "nr_attr", "Model Capability"],
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
        self.assertIsPng(term_count_curve(curve(), self.folder / "curve.png", reference=0.661))

    def test_term_count_curve_without_a_reference(self) -> None:
        self.assertIsPng(term_count_curve(curve(), self.folder / "plain.png"))

    def test_term_count_curve_with_only_one_protocol(self) -> None:
        partial = curve().drop("r2_loo_model", "mae_loo_model")
        self.assertIsPng(term_count_curve(partial, self.folder / "partial.png"))

    def test_predicted_versus_actual(self) -> None:
        rng = np.random.default_rng(0)
        truth = rng.uniform(-1.0, 1.0, 80)
        self.assertIsPng(predicted_versus_actual(truth, truth + rng.normal(0, 0.1, 80), self.folder / "scatter.png"))

    def test_scatter_with_a_rug(self) -> None:
        truth = np.linspace(0.0, 1.0, 40)
        self.assertIsPng(predicted_versus_actual(truth, truth * 0.9, self.folder / "rug.png", groups=truth))

    def test_scatter_of_realistic_data_is_still_a_png(self) -> None:
        truth = np.linspace(0.0, 1.0, 40)
        self.assertIsPng(predicted_versus_actual(truth, truth * 0.9 + 0.05, self.folder / "tight.png"))

    def test_scatter_renders_with_points_below_the_floor(self) -> None:
        # The out-of-range point falls outside the axes; rendering must not crash.
        truth = np.concatenate([np.array([-0.29]), np.linspace(0.0, 1.0, 30)])
        predicted = np.concatenate([np.array([0.5]), np.linspace(0.2, 1.0, 30)])
        self.assertIsPng(predicted_versus_actual(truth, predicted, self.folder / "clipped.png"))

    def test_term_effects(self) -> None:
        self.assertIsPng(term_effects(effects(), self.folder / "terms.png"))

    def test_term_effects_respects_the_top_limit(self) -> None:
        self.assertIsPng(term_effects(effects(), self.folder / "top.png", top=2))

    def test_term_effects_draws_every_term_by_default(self) -> None:
        """No implicit truncation of the equation.

        The default was 12, which silently dropped four of the published sixteen terms -- and
        the dropped ones are the small-effect terms the brevity argument is about.
        """
        table = pl.concat([effects()] * 7)  # 21 terms, comfortably past the old default
        figure = term_effects(table, self.folder / "all_terms.png")
        self.assertIsPng(figure)
        # 0.52 inches per bar plus margin: taller than the old fixed 12-bar figure would be.
        self.assertGreater(figure.stat().st_size, 0)

    def test_practice_effects(self) -> None:
        self.assertIsPng(practice_effects(practices(), self.folder / "practices.png"))

    def test_equation_comparison(self) -> None:
        table = pl.DataFrame(
            {
                "equation": [
                    "E1, dataset only (7 terms)",
                    "E1 reference: true dataset means",
                    "E3-Valid, dataset + model (15 terms)",
                ],
                "r2": [0.337, 0.354, 0.556],
            }
        )
        self.assertIsPng(equation_comparison(table, self.folder / "comparison.png"))

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


class TestAdditiveReferenceLookup(unittest.TestCase):
    """The ceiling drawn on the curve must be the one the run computed."""

    def test_reads_the_reference_from_the_comparison_table(self) -> None:
        from ml_meta_perf.figures import ADDITIVE_REFERENCE_ROW, _additive_reference

        report = SimpleNamespace(
            comparison=pl.DataFrame(
                {"equation": ["E2 (dataset + model)", ADDITIVE_REFERENCE_ROW], "r2": [0.556, 0.6605]}
            )
        )
        value = _additive_reference(report)  # pyright: ignore[reportArgumentType]
        assert value is not None
        self.assertAlmostEqual(value, 0.6605)

    def test_returns_none_when_the_row_is_absent(self) -> None:
        from ml_meta_perf.figures import _additive_reference

        report = SimpleNamespace(comparison=pl.DataFrame({"equation": ["E2"], "r2": [0.5]}))
        self.assertIsNone(_additive_reference(report))  # pyright: ignore[reportArgumentType]


class TestFigureSet(PlotTestCase):
    def test_generate_writes_the_whole_set(self) -> None:
        from ml_meta_perf.figures import generate

        written = generate(corpus.report(), self.folder, corpus.sample_path())
        self.assertEqual(len(written), len(set(written)))
        for path in written:
            self.assertIsPng(path)


class TestFigureNaming(unittest.TestCase):
    """Figures are numbered by their position in the set, so they can be named by number.

    The number is derived from `FIGURE_ORDER` rather than written beside each call, which is
    what stops a file called `04_term_effects.png` from being the fifth figure in the
    chapters. `generate` asserts the same thing at runtime.
    """

    def test_numbers_run_from_one_in_order(self) -> None:
        from ml_meta_perf.figures import FIGURE_ORDER, figure_name

        self.assertEqual(
            [figure_name(stem) for stem in FIGURE_ORDER],
            [f"{index:02d}_{stem}.png" for index, stem in enumerate(FIGURE_ORDER, start=1)],
        )

    def test_every_figure_has_a_caption_under_its_published_name(self) -> None:
        """A caption keyed by the unnumbered stem would silently go missing on rename."""
        from ml_meta_perf.figures import FIGURE_ORDER, captions, figure_name

        available = captions(corpus.report(), corpus.sample_path())
        self.assertEqual(set(available), {figure_name(stem) for stem in FIGURE_ORDER})

    def test_an_unknown_stem_is_refused(self) -> None:
        from ml_meta_perf.figures import figure_name

        with self.assertRaises(ValueError):
            figure_name("not_a_figure")

    def test_the_chapters_reference_the_published_names(self) -> None:
        """The markdown embeds are hand-written, so nothing else checks they were renamed."""
        import re
        from pathlib import Path

        from ml_meta_perf.figures import FIGURE_ORDER, figure_name

        published = {figure_name(stem) for stem in FIGURE_ORDER}
        docs = Path(__file__).resolve().parent.parent / "assets" / "docs"
        if not docs.is_dir():
            self.skipTest("chapters are not installed beside the package")
        for page in docs.glob("*.md"):
            for referenced in re.findall(r"figures/([\w.]+\.png)", page.read_text(encoding="utf-8")):
                self.assertIn(referenced, published, f"{page.name} references {referenced}")


class TestTermMath(unittest.TestCase):
    """Term names render as mathematics, set inline so every label is one height."""

    def test_a_ratio_is_inline_rather_than_built_up(self) -> None:
        """A built-up `\\frac` is set smaller than the line around it, and the figure mixes
        ratios with products -- so half the labels came out at two thirds the size of the
        other half. An inline slash keeps them comparable."""
        rendered = term_to_math("[log(eq_num_attr)] / [log(Processing Units Number)]")
        self.assertTrue(rendered.startswith("$") and rendered.endswith("$"))
        self.assertNotIn(r"\frac", rendered)
        self.assertIn("/", rendered)

    def test_a_product_uses_times(self) -> None:
        self.assertIn(r"\times", term_to_math("[log(gravity)] * [log(Model Capability)]"))
        self.assertNotIn(r"\cdot", term_to_math("[log(gravity)] * [log(Model Capability)]"))

    def test_a_logarithm_is_parenthesised(self) -> None:
        """`\\log a \\times \\log b` does not say where the first logarithm stops."""
        self.assertIn(r"\log(\mathrm{gravity})", term_to_math("[log(gravity)] * [log(Model Capability)]"))

    def test_a_nested_reciprocal_is_parenthesised(self) -> None:
        """Inline division is only unambiguous if `1/f` inside a ratio gets brackets:
        `1 / f / g` is a different expression from `(1 / f) / g`."""
        rendered = term_to_math("[1/gravity] / [log(nr_class)]")
        self.assertIn(r"\left(", rendered)
        self.assertIn(r"\right)", rendered)

    def test_a_top_level_reciprocal_needs_no_brackets(self) -> None:
        self.assertNotIn(r"\left(", term_to_math("1/Fitting Regime"))

    def test_model_features_are_abbreviated(self) -> None:
        """Five-syllable names do not fit fifteen to a figure; chapter 1 expands them."""
        self.assertIn("PUN", term_to_math("[nr_cor_attr] / [log(Processing Units Number)]"))

    def test_underscores_are_escaped(self) -> None:
        """An unescaped underscore is a subscript in mathtext, and renders as gibberish."""
        self.assertIn(r"\_", term_to_math("class_ent^2"))

    def test_a_leading_numeral_stays_a_numeral(self) -> None:
        self.assertTrue(term_to_math("1/Fitting Regime").startswith("$1"))

    def test_a_sum_over_a_ratio_keeps_both_operands(self) -> None:
        rendered = term_to_math("([log(inst_to_attr)] + [nr_norm]) / [log(nr_attr)]")
        self.assertIn("inst", rendered)
        self.assertIn("nr", rendered)
        # The sum is bracketed, or inline division would read as `a + b / c`.
        self.assertIn(r"\left(", rendered)

    def test_every_published_shape_renders(self) -> None:
        """mathtext raises on malformed input, so a bad term would break the whole figure."""
        import matplotlib.pyplot as plt

        shapes = [
            "[a] * [b]",
            "[a] / [b]",
            "1/a",
            "a^2",
            "sqrt(a)",
            "log(a)",
            "([a] + [b]) / [c]",
            "[log(a)] / [log(b)]",
            "[1/a] / [b]",
            "[1/a] * [b]",
            "[sqrt(a)] / [1/b]",
        ]
        figure, axes = plt.subplots()
        for index, shape in enumerate(shapes):
            axes.text(0.1, 0.05 * index, term_to_math(shape))
        figure.canvas.draw()  # raises if any expression is malformed
        plt.close(figure)


class TestConfidenceShading(unittest.TestCase):
    """The legend lists the levels on the plot, and every level has its own opacity."""

    def test_unrated_is_distinct_from_moderate(self) -> None:
        """They shared an alpha, so a table of moderate and unrated rows rendered flat."""
        self.assertNotEqual(CONFIDENCE_ALPHA["unrated"], CONFIDENCE_ALPHA["moderate"])

    def test_levels_are_the_ones_present_strongest_first(self) -> None:
        table = pl.DataFrame({"confidence": ["unrated", "moderate", "moderate"]})
        self.assertEqual(_confidence_levels(table), ["moderate", "unrated"])

    def test_no_levels_without_the_column(self) -> None:
        self.assertEqual(_confidence_levels(pl.DataFrame({"effect": [0.1]})), [])


if __name__ == "__main__":
    unittest.main()
