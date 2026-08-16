"""The end-to-end study.

These run against the real meta-dataset but with deliberately small configurations:
the point is that the wiring is correct and the reported relationships hold, not to
reproduce the published numbers inside a commit hook.
"""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from metafit.attribution import group_shares, term_effects, variance_decomposition
from metafit.cli import main, render
from metafit.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FEATURES,
    columns_as_arrays,
    groups,
    load,
    target,
)
from metafit.experiment import (
    Configuration,
    Report,
    baselines,
    comparison,
    correlation_analysis,
    leakage_demonstration,
    model_selection,
    run_e1,
    run_e2,
    run_model_only,
)
from metafit.model import Equation
from metafit.practices import best_practices
from metafit.selection import pareto_table, recommend
from metafit.validate import oracle_ladder

FAST_E1 = Configuration(max_abs_zscore=3.0, penalty=1.0, pool_size=40, max_terms=3, headline_terms=3)
FAST_E2 = Configuration(max_abs_zscore=3.0, penalty=20.0, pool_size=40, max_terms=3, headline_terms=3)


class TestEquationReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()
        cls.e1 = run_e1(cls.frame, FAST_E1)
        cls.e2 = run_e2(cls.frame, FAST_E2)

    def test_e1_reports_a_readable_equation(self) -> None:
        self.assertEqual(self.e1.equation.n_terms, 3)
        for weight in self.e1.equation.weights:
            # The stability filters exist to keep coefficients on a human scale.
            self.assertLess(abs(weight), 1e6)

    def test_e1_only_uses_dataset_features(self) -> None:
        from metafit.data import MODEL_FEATURES

        used = {feature for term in self.e1.equation.terms for feature in term.features}
        self.assertFalse(used & set(MODEL_FEATURES))

    def test_e2_uses_at_least_one_model_feature(self) -> None:
        from metafit.data import MODEL_FEATURES

        used = {feature for term in self.e2.equation.terms for feature in term.features}
        self.assertTrue(used & set(MODEL_FEATURES))

    def test_curves_cover_the_requested_sizes(self) -> None:
        self.assertGreater(self.e1.curve.height, 0)
        self.assertIn("r2_in_sample", self.e1.curve.columns)
        self.assertIn("r2_loo_dataset", self.e2.curve.columns)
        self.assertIn("r2_loo_model", self.e2.curve.columns)

    def test_cross_validated_scores_are_reported_for_both_protocols(self) -> None:
        self.assertEqual(set(self.e2.cross_validated), {"loo_dataset", "loo_model"})

    def test_cross_validation_never_beats_the_in_sample_fit(self) -> None:
        for scores in self.e2.cross_validated.values():
            self.assertLessEqual(scores["r2"], self.e2.in_sample["r2"] + 1e-9)

    def test_stability_table_is_populated(self) -> None:
        self.assertIsNotNone(self.e2.stability)
        assert self.e2.stability is not None
        self.assertIn("frequency", self.e2.stability.columns)


class TestStudyTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()
        cls.e1 = run_e1(cls.frame, FAST_E1)
        cls.e2 = run_e2(cls.frame, FAST_E2)

    def test_e2_beats_e1_on_the_common_scale(self) -> None:
        # The central claim of the study: model features carry information that dataset
        # features cannot express, because E1 can only predict a per-dataset constant.
        table = comparison(self.frame, self.e1, self.e2)
        scores = dict(zip(table["equation"].to_list(), table["r2"].to_list(), strict=True))
        self.assertGreater(scores["E2 (dataset + model)"], scores["E1 (dataset only)"])

    def test_e1_cannot_exceed_the_dataset_mean_ceiling(self) -> None:
        table = comparison(self.frame, self.e1, self.e2)
        scores = dict(zip(table["equation"].to_list(), table["r2"].to_list(), strict=True))
        self.assertLessEqual(scores["E1 (dataset only)"], scores["E1 ceiling (true dataset means)"] + 1e-9)

    def test_nothing_additive_passes_the_oracle(self) -> None:
        table = comparison(self.frame, self.e1, self.e2)
        scores = dict(zip(table["equation"].to_list(), table["r2"].to_list(), strict=True))
        self.assertLessEqual(scores["E2 (dataset + model)"], scores["additive oracle (ceiling)"] + 1e-9)

    def test_baselines_table_is_complete(self) -> None:
        table = baselines(self.frame)
        self.assertEqual(table.height, 5)
        self.assertIn("r2", table.columns)

    def test_per_model_mean_beats_the_global_mean(self) -> None:
        table = baselines(self.frame)
        scores = dict(zip(table["baseline"].to_list(), table["r2"].to_list(), strict=True))
        self.assertGreater(
            scores["per-model mean (loo-dataset)"], scores["global mean (loo-dataset)"]
        )

    def test_random_folds_look_better_than_grouped_ones(self) -> None:
        # The leakage this project exists to warn about: a random split scores the same
        # equation far higher because dataset identity is visible on both sides.
        table = leakage_demonstration(self.frame, FAST_E2)
        scores = dict(zip(table["protocol"].to_list(), table["r2"].to_list(), strict=True))
        self.assertGreater(scores["random 10-fold (leaky)"], scores["leave-one-dataset-out"])

    def test_model_only_equation_uses_no_dataset_feature(self) -> None:
        report = run_model_only(self.frame, FAST_E2)
        used = {feature for term in report.equation.terms for feature in term.features}
        self.assertTrue(used)
        self.assertFalse(used & set(DATASET_FEATURES))

    def test_dataset_features_explain_more_than_model_features(self) -> None:
        # The direct test of "model choice outweighs the data": on this meta-dataset it
        # does not. Dataset identity explains more variance, and the dataset-only
        # equation gets far closer to its own ceiling than the model-only one does.
        table = comparison(self.frame, self.e1, self.e2)
        scores = dict(zip(table["equation"].to_list(), table["r2"].to_list(), strict=True))
        self.assertGreater(scores["E1 (dataset only)"], scores["EM (model only)"])
        self.assertGreater(
            scores["E1 ceiling (true dataset means)"], scores["EM ceiling (true model means)"]
        )

    def test_model_selection_reports_every_dataset(self) -> None:
        table = model_selection(self.frame, self.e2)
        self.assertEqual(table.height, 20)
        self.assertTrue((table["regret"] >= 0.0).all())

    def test_correlation_analysis_reports_both_correlations(self) -> None:
        table = correlation_analysis(self.frame, FAST_E2, top=10)
        self.assertEqual(table.height, 10)
        for column in ("term", "pearson", "spearman", "within_pearson", "monotone_gap"):
            self.assertIn(column, table.columns)

    def test_correlation_analysis_is_ranked_by_within_group_strength(self) -> None:
        strengths = correlation_analysis(self.frame, FAST_E2, top=10)["strength"].to_numpy()
        self.assertTrue((np.diff(strengths) <= 1e-12).all())


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = load()
        e1, e2 = run_e1(frame, FAST_E1), run_e2(frame, FAST_E2)
        cols = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
        cls.report = Report(
            e1=e1,
            e2=e2,
            e2_accurate=e2,
            model_only=run_model_only(frame, FAST_E2),
            practices=best_practices(e2.equation, cols, e2.stability),
            effects=term_effects(e2.equation, cols, DATASET_FEATURES, MODEL_FEATURES),
            shares=group_shares(e2.equation, cols, DATASET_FEATURES, MODEL_FEATURES),
            decomposition=variance_decomposition(
                target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)
            ),
            correlations=correlation_analysis(frame, FAST_E2, top=5),
            baselines=baselines(frame),
            comparison=comparison(frame, e1, e2),
            leakage=leakage_demonstration(frame, FAST_E2),
            selection=model_selection(frame, e2),
            term_choice=recommend(e2.curve),
            pareto=pareto_table(e2.curve),
            oracles=oracle_ladder(
                target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN), ranks=(0, 1)
            ),
        )

    def test_render_prints_every_section(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            render(self.report)
        printed = buffer.getvalue()
        for expected in ("Correlation screening", "E1 --", "E2 --", "E2 accurate --", "EM --", "Where the signal lives",
                         "Extracted practices", "Baselines", "Model selection"):
            self.assertIn(expected, printed)

    def test_main_writes_equations_that_load_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(["--quick", "--save", directory])
            self.assertEqual(code, 0)
            for name in ("e1.json", "e2.json"):
                path = Path(directory) / name
                self.assertTrue(path.is_file())
                self.assertGreater(Equation.load(path).n_terms, 0)


if __name__ == "__main__":
    unittest.main()
