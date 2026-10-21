"""Which protocols a study is scored under: the fit runs only those, and a missing group is an error."""

import dataclasses
import unittest

import numpy as np
import polars as pl

from ml_meta_perf.config import PROTOCOLS, Schema, default_configuration
from ml_meta_perf.data import columns_as_arrays, groups, target
from ml_meta_perf.experiment import fit_path
from ml_meta_perf.search import search
from ml_meta_perf.selection import floor_curve
from ml_meta_perf.terms import build_library
from ml_meta_perf.validate import cross_validate_fixed_form

TINY = dataclasses.replace(default_configuration(), max_terms=2, pool_size=20, beam_width=1, max_arity=1)


def corpus(datasets: int = 6, models: int = 3, seed: int = 4) -> pl.DataFrame:
    """A dataset-by-model corpus whose target is linear in one feature of each side, plus noise."""
    rng = np.random.default_rng(seed)
    size, weight = rng.uniform(1.0, 9.0, datasets), rng.uniform(1.0, 5.0, models)
    rows = [
        {"d": f"d{i}", "m": f"m{j}", "size": size[i], "depth": weight[j], "y": 2.0 + 0.5 * size[i] + weight[j]}
        for i in range(datasets)
        for j in range(models)
    ]
    return pl.DataFrame(rows).with_columns((pl.col("y") + pl.Series(rng.normal(0, 0.01, len(rows)))).alias("y"))


def schema(protocols: tuple[str, ...] = PROTOCOLS, *, model_column: str | None = "m") -> Schema:
    return Schema("d", model_column, ("size",), ("depth",), "y", None, protocols=protocols)


class TestFitPathProtocols(unittest.TestCase):
    def fit(self, protocols: tuple[str, ...], model_column: str | None = "m"):
        return fit_path(
            corpus(), ("size",), ("depth",), TINY, "E3", schema=schema(protocols, model_column=model_column)
        )

    def test_every_protocol_runs_by_default(self) -> None:
        fitted = self.fit(PROTOCOLS)
        self.assertEqual(set(fitted.paths), {"loo_dataset", "loo_model", "loo_cell"})
        for column in ("r2_in_sample", "r2_loo_dataset", "r2_loo_model", "r2_loo_cell"):
            self.assertIn(column, fitted.curve.columns)

    def test_only_the_asked_protocols_are_computed(self) -> None:
        fitted = self.fit(("IS", "LODO"))
        self.assertEqual(set(fitted.paths), {"loo_dataset"})
        self.assertNotIn("r2_loo_model", fitted.curve.columns)
        self.assertNotIn("r2_loo_cell", fitted.curve.columns)

    def test_the_floor_is_the_worst_of_the_protocols_that_ran(self) -> None:
        full, partial = self.fit(PROTOCOLS), self.fit(("IS", "LODO", "LOMO"))
        curve = partial.curve
        expected = np.minimum(
            curve["r2_in_sample"].to_numpy(),
            np.minimum(*(curve[f"r2_{p}"].to_numpy() for p in ("loo_dataset", "loo_model"))),
        )
        np.testing.assert_allclose(floor_curve(curve), expected)
        self.assertTrue(np.all(floor_curve(full.curve) <= floor_curve(curve) + 1e-12))

    def test_a_protocol_shared_by_two_runs_gives_the_same_numbers(self) -> None:
        full, partial = self.fit(PROTOCOLS), self.fit(("IS", "LODO"))
        np.testing.assert_array_equal(
            full.curve["r2_loo_dataset"].to_numpy(), partial.curve["r2_loo_dataset"].to_numpy()
        )

    def test_a_corpus_with_one_group_column_can_be_fitted(self) -> None:
        fitted = self.fit(("IS", "LODO"), model_column=None)
        self.assertEqual(set(fitted.paths), {"loo_dataset"})
        self.assertTrue(np.isfinite(fitted.curve["r2_loo_dataset"].to_numpy()).all())


class TestAFeatureConstantInTraining(unittest.TestCase):
    """A fold whose training rows never vary in a feature must not let that feature predict.

    The standardiser leaves a constant column at zero, so its weight is zero and the prediction
    cannot move with a value it never saw. This pins that, since it is what lets LODO run on a
    corpus where one dataset alone carries a feature value.
    """

    def test_the_weight_of_a_term_that_is_constant_in_the_training_rows_is_zero(self) -> None:
        frame = corpus()
        # `rare` is 0 everywhere except the held-out dataset, where it carries the whole signal.
        frame = frame.with_columns(rare=pl.when(pl.col("d") == "d0").then(7.0).otherwise(0.0))
        frame = frame.with_columns((pl.col("y") + 3.0 * pl.col("rare")).alias("y"))
        features = ("size", "rare")
        columns = columns_as_arrays(frame, (*features, "depth"))
        truth, datasets = target(frame, schema()), groups(frame, "d")
        library = build_library(features, ("depth",), columns, max_arity=1, max_abs_zscore=10.0)
        result = search(library, truth, max_terms=3, penalty=0.1, pool_size=20, beam_width=1)
        equation = result.equations[3]
        self.assertIn(
            "rare", {name for term in equation.terms for name in term.name.replace("(", " ").replace(")", " ").split()}
        )
        outcome = cross_validate_fixed_form(library, columns, truth, datasets, {3: equation}, penalty=0.1)[3]
        fold = outcome.equations["d0"]
        for term, weight in zip(fold.terms, fold.standardized_weights, strict=True):
            if "rare" in term.name:
                self.assertEqual(weight, 0.0)
        self.assertTrue(np.isfinite(outcome.predictions).all())


if __name__ == "__main__":
    unittest.main()
