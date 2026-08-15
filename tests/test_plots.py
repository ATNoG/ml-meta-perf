"""Figures. These check that each plot writes a valid PNG, not that it looks right."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from metafit.model import Equation
from metafit.plots import (
    contribution_sources,
    equation_summary,
    practice_effects,
    predicted_versus_actual,
    protocol_comparison,
    term_count_curve,
    term_effects,
)
from metafit.terms import Atom, Term

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
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 1000)
        with path.open("rb") as handle:
            self.assertEqual(handle.read(8), PNG_MAGIC)


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

    def test_term_effects(self) -> None:
        self.assertIsPng(term_effects(effects(), self.folder / "terms.png"))

    def test_term_effects_respects_the_top_limit(self) -> None:
        self.assertIsPng(term_effects(effects(), self.folder / "top.png", top=2))

    def test_practice_effects(self) -> None:
        self.assertIsPng(practice_effects(practices(), self.folder / "practices.png"))

    def test_protocol_comparison(self) -> None:
        leakage = pl.DataFrame(
            {"protocol": ["random 10-fold (leaky)", "leave-one-dataset-out"], "r2": [0.51, 0.37]}
        )
        self.assertIsPng(protocol_comparison(leakage, self.folder / "protocols.png"))

    def test_contribution_sources(self) -> None:
        shares = pl.DataFrame(
            {"group": ["dataset", "model", "mixed"], "n_terms": [5, 4, 3], "share": [0.52, 0.28, 0.20]}
        )
        decomposition = pl.DataFrame(
            {"knowing only": ["dataset identity", "model identity"], "variance_explained": [0.354, 0.282]}
        )
        self.assertIsPng(contribution_sources(shares, decomposition, self.folder / "sources.png"))

    def test_equation_summary(self) -> None:
        equation = Equation(
            intercept=0.83,
            terms=(Term("atom", (Atom("a"),)), Term("ratio", (Atom("b"), Atom("c")))),
            weights=(0.1, -0.2),
            standardized_weights=(0.3, -0.5),
        )
        self.assertIsPng(equation_summary(equation, self.folder / "equation.png"))

    def test_creates_missing_directories(self) -> None:
        nested = self.folder / "a" / "b" / "curve.png"
        self.assertIsPng(term_count_curve(curve(), nested))


class TestFigureSet(PlotTestCase):
    def test_generate_writes_the_whole_set(self) -> None:
        from metafit.experiment import run
        from metafit.figures import generate

        written = generate(run(quick=True), self.folder)
        self.assertEqual(len(written), len(set(written)))
        for path in written:
            self.assertIsPng(path)


if __name__ == "__main__":
    unittest.main()
