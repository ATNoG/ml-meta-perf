"""Tests for the per-model correction: what it fits, and where it must refuse to."""

from __future__ import annotations

import unittest

import numpy as np

from metafit.identity import (
    ModelEffects,
    carrier_stability,
    carrying_atoms,
    correct_out_of_fold,
    fit_effects,
    predict,
)
from metafit.model import Equation
from metafit.terms import Atom, Library, Term
from metafit.validate import CrossValidation, cross_validate_path


def _grid(n_datasets: int = 6, n_models: int = 5) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """A small (dataset x model) grid with a known per-model level and slope."""
    rng = np.random.default_rng(11)
    difficulty = np.exp(np.linspace(0.5, 6.0, n_datasets))
    datasets = np.repeat([f"d{i}" for i in range(n_datasets)], n_models)
    models = np.tile([f"m{j}" for j in range(n_models)], n_datasets)
    columns = {
        "difficulty": np.repeat(difficulty, n_models),
        "size": np.repeat(np.linspace(100.0, 900.0, n_datasets), n_models),
    }
    level = np.tile(np.linspace(-0.2, 0.2, n_models), n_datasets)
    slope = np.tile(np.linspace(-0.05, 0.05, n_models), n_datasets)
    residual = level + slope * np.log(columns["difficulty"]) + rng.normal(scale=1e-3, size=datasets.shape[0])
    return columns, residual, models


class TestFitEffects(unittest.TestCase):
    def setUp(self) -> None:
        self.columns, self.residual, self.models = _grid()

    def test_recovers_a_planted_level_and_slope(self) -> None:
        effects = fit_effects(
            self.residual,
            self.columns,
            self.models,
            ("difficulty", "size"),
            intercept_shrinkage=0.0,
            slope_shrinkage=0.0,
        )
        assert effects.atom is not None
        self.assertEqual(effects.atom.name, "log(difficulty)")
        fitted = effects.apply(self.columns, self.models)
        np.testing.assert_allclose(fitted, self.residual, atol=0.02)

    def test_no_features_gives_levels_only(self) -> None:
        effects = fit_effects(self.residual, self.columns, self.models)
        self.assertIsNone(effects.atom)
        self.assertEqual(effects.slopes, {})
        self.assertEqual(effects.n_models, 5)

    def test_shrinkage_pulls_levels_toward_zero(self) -> None:
        loose = fit_effects(self.residual, self.columns, self.models, intercept_shrinkage=0.0)
        tight = fit_effects(self.residual, self.columns, self.models, intercept_shrinkage=1e6)
        for name, value in tight.intercepts.items():
            self.assertLess(abs(value), abs(loose.intercepts[name]) + 1e-12)
            self.assertAlmostEqual(value, 0.0, places=5)

    def test_an_unseen_model_gets_no_correction(self) -> None:
        effects = fit_effects(self.residual, self.columns, self.models, ("difficulty",))
        unseen = np.array(["m0", "brand_new"])
        columns = {name: values[:2] for name, values in self.columns.items()}
        correction = effects.apply(columns, unseen)
        self.assertNotAlmostEqual(float(correction[0]), 0.0)
        self.assertEqual(float(correction[1]), 0.0)

    def test_the_carrier_choice_is_by_residual_not_by_order(self) -> None:
        # ``size`` is offered first and explains nothing; the fit must still take
        # ``difficulty``, which is what the residual was built from.
        effects = fit_effects(self.residual, self.columns, self.models, ("size", "difficulty"))
        assert effects.atom is not None
        self.assertEqual(effects.atom.feature, "difficulty")

    def test_a_constant_carrier_is_refused(self) -> None:
        columns = dict(self.columns)
        columns["flat"] = np.ones_like(columns["size"])
        effects = fit_effects(self.residual, columns, self.models, ("flat",))
        self.assertIsNone(effects.atom)

    def test_table_is_one_row_per_model_ordered_by_level(self) -> None:
        table = fit_effects(self.residual, self.columns, self.models, ("difficulty",)).table()
        self.assertEqual(table.height, 5)
        self.assertEqual(table.columns, ["model", "level", "slope"])
        levels = table["level"].to_list()
        self.assertEqual(levels, sorted(levels, reverse=True))

    def test_str_names_the_carrier(self) -> None:
        rendered = str(fit_effects(self.residual, self.columns, self.models, ("difficulty",)))
        self.assertIn("log(difficulty)", rendered)
        self.assertIn("b_model", rendered)


class TestCarryingAtoms(unittest.TestCase):
    def test_positive_features_are_log_compressed(self) -> None:
        columns = {"positive": np.array([1.0, 2.0, 3.0]), "signed": np.array([-1.0, 0.0, 1.0])}
        atoms = carrying_atoms(("positive", "signed"), columns)
        self.assertEqual([atom.transform for atom in atoms], ["log", "id"])


class TestPredict(unittest.TestCase):
    def test_prediction_is_clipped_into_the_mcc_range(self) -> None:
        columns = {"f": np.array([1.0, 2.0])}
        equation = Equation(
            intercept=0.9,
            terms=(Term("atom", (Atom("f"),)),),
            weights=(0.5,),
            standardized_weights=(0.5,),
        )
        effects = ModelEffects(atom=None, intercepts={"m": 5.0}, slopes={})
        values = predict(equation, effects, columns, np.array(["m", "m"]))
        np.testing.assert_array_equal(values, np.array([1.0, 1.0]))


class TestCorrectOutOfFold(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        n_datasets, n_models = 8, 6
        self.datasets = np.repeat([f"d{i}" for i in range(n_datasets)], n_models)
        self.models = np.tile([f"m{j}" for j in range(n_models)], n_datasets)
        self.columns = {
            "difficulty": np.repeat(np.exp(np.linspace(0.5, 5.0, n_datasets)), n_models),
            "capacity": np.tile(np.linspace(1.0, 6.0, n_models), n_datasets),
        }
        # The level must not be a function of ``capacity``, or the equation would already
        # be able to reach it and the correction would have nothing left to explain --
        # which is the situation the meta-dataset is *not* in.
        per_model = np.array([0.3, -0.25, 0.05, -0.1, 0.2, -0.2])
        level = np.tile(per_model, n_datasets)
        self.target = (
            0.4
            + 0.05 * np.log(self.columns["difficulty"])
            + level
            + rng.normal(scale=0.01, size=self.datasets.shape[0])
        )
        terms = [
            Term("atom", (Atom("difficulty", "log"),)),
            Term("atom", (Atom("capacity"),)),
        ]
        self.library = Library(terms, self.columns)
        self.path = cross_validate_path(
            self.library,
            self.columns,
            self.target,
            self.datasets,
            max_terms=2,
            penalty=0.0,
            pool_size=2,
            beam_width=2,
        )

    def test_the_path_records_an_equation_per_fold(self) -> None:
        self.assertEqual(len(self.path[2].equations), 8)

    def test_the_correction_improves_a_planted_model_effect(self) -> None:
        from metafit.stats import r2_score

        before = r2_score(self.target, self.path[2].predictions)
        after = r2_score(
            self.target,
            correct_out_of_fold(self.path[2], self.columns, self.target, self.datasets, self.models),
        )
        self.assertGreater(after, before)

    def test_holding_out_the_model_leaves_predictions_untouched(self) -> None:
        # Under leave-one-model-out the held-out model has no training row, so there is no
        # effect to apply and the correction must be exactly the identity. This is the
        # boundary of the method and it is asserted rather than described.
        path = cross_validate_path(
            self.library,
            self.columns,
            self.target,
            self.models,
            max_terms=2,
            penalty=0.0,
            pool_size=2,
            beam_width=2,
        )
        corrected = correct_out_of_fold(
            path[2], self.columns, self.target, self.models, self.models, ("difficulty",)
        )
        np.testing.assert_array_equal(corrected, path[2].predictions)

    def test_a_fold_without_an_equation_keeps_its_prediction(self) -> None:
        empty = CrossValidation(predictions=np.full_like(self.target, 0.25))
        corrected = correct_out_of_fold(
            empty, self.columns, self.target, self.datasets, self.models
        )
        np.testing.assert_array_equal(corrected, empty.predictions)

    def test_carrier_stability_counts_folds(self) -> None:
        table = carrier_stability(
            self.columns,
            self.target,
            self.datasets,
            self.models,
            self.path[2],
            ("difficulty", "capacity"),
        )
        self.assertEqual(int(table["folds"].sum()), 8)
        self.assertAlmostEqual(float(table["frequency"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
