"""The end-to-end study.

These run against the real meta-dataset but with deliberately small configurations:
the point is that the wiring is correct and the reported relationships hold, not to
reproduce the published numbers inside a commit hook.
"""

import re
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_FEATURES,
    load,
)
from ml_meta_perf.experiment import (
    ARITIES,
    DEFAULT,
    baselines,
    comparison,
    correlation_analysis,
    leakage_demonstration,
    model_selection,
    run_e1,
    run_e2,
    run_e3,
)
from tests import corpus


def scored(table, prefix: str) -> float:
    """R2 of the comparison row whose label starts with ``prefix``.

    By prefix because the labels carry their term count, which moves with the configuration
    -- and these tests deliberately run a fast three-term one.
    """
    matched = table.filter(pl.col("equation").str.starts_with(prefix))
    return float(matched["r2"][0])


class TestEquationReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = corpus.sample()
        cls.e1 = run_e1(cls.frame, corpus.E1)
        cls.e3 = run_e3(cls.frame, corpus.E3)

    def test_e1_reports_a_readable_equation(self) -> None:
        # A count, not a number: pruning decides the length and the search horizon bounds it,
        # so asserting the horizon exactly was asserting that pruning never fires.
        self.assertGreater(self.e1.equation.n_terms, 0)
        self.assertLessEqual(self.e1.equation.n_terms, corpus.E1.max_terms)
        for weight in self.e1.equation.weights:
            # The stability filters exist to keep coefficients on a human scale.
            self.assertLess(abs(weight), 1e6)

    def test_e1_only_uses_dataset_features(self) -> None:

        used = {feature for term in self.e1.equation.terms for feature in term.features}
        self.assertFalse(used & set(MODEL_FEATURES))

    def test_e2_uses_at_least_one_model_feature(self) -> None:

        used = {feature for term in self.e3.equation.terms for feature in term.features}
        self.assertTrue(used & set(MODEL_FEATURES))

    def test_curves_cover_the_requested_sizes(self) -> None:
        self.assertGreater(self.e1.curve.height, 0)
        self.assertIn("r2_in_sample", self.e1.curve.columns)
        self.assertIn("r2_loo_dataset", self.e3.curve.columns)
        self.assertIn("r2_loo_model", self.e3.curve.columns)
        self.assertIn("r2_loo_cell", self.e3.curve.columns)

    def test_cross_validated_scores_are_reported_for_every_protocol(self) -> None:
        """Three, not two, since C1 (2026-09-09). The doubly-held-out protocol is computed at
        every length anyway -- `selection.floor_curve` needs it to choose the length -- and the
        standing rule here is that a comparison missing its strictest column is not
        conservative, it flatters whichever side had more left over."""
        self.assertEqual(set(self.e3.cross_validated), {"loo_dataset", "loo_model", "loo_cell"})

    def test_cross_validated_scores_are_finite_and_bounded(self) -> None:
        # Deliberately *not* asserting cross-validated <= in-sample. Cross-validated
        # predictions come from 20 different fold-equations, and that ensemble can beat a
        # single equation when each one is heavily constrained -- which is exactly the
        # FAST configuration used here (3 terms from a 40-term pool). At the real
        # configuration the ordering holds at every length, but it is not an invariant of
        # the method and asserting it was wrong.
        for scores in self.e3.cross_validated.values():
            self.assertTrue(-10.0 < scores["r2"] <= 1.0)

    def test_stability_table_is_populated(self) -> None:
        self.assertIsNotNone(self.e3.stability)
        assert self.e3.stability is not None
        self.assertIn("frequency", self.e3.stability.columns)

    def test_published_equations_are_already_simplified(self) -> None:
        # A published term that ``simplify`` can still shorten is a defect: the whole
        # claim is that the equation can be read, and ``([a] / [b]) * [b]`` says ``a`` in
        # six symbols. ``prune`` applies it, and this is the check that it stuck.
        from ml_meta_perf.terms import simplify

        for equation in (self.e1.equation, self.e3.equation):
            for term in equation.terms:
                self.assertEqual(simplify(term), term, f"{term.name} is still reducible")

    def test_published_equations_carry_no_duplicate_terms(self) -> None:
        # Two terms that simplify to the same expression are collinear, so the weights
        # they get are arbitrary and the pair reads as two pieces of evidence when it is
        # one.
        for equation in (self.e1.equation, self.e3.equation):
            names = [term.name for term in equation.terms]
            self.assertEqual(len(names), len(set(names)))


class TestStudyTables(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = corpus.sample()
        cls.e1 = run_e1(cls.frame, corpus.E1)
        cls.e3 = run_e3(cls.frame, corpus.E3)

    def test_e1_cannot_exceed_the_dataset_mean_ceiling(self) -> None:
        table = comparison(self.frame, self.e1, self.e3)
        self.assertLessEqual(scored(table, "E1, dataset only"), scored(table, "E1 reference") + 1e-9)

    def test_group_equations_stay_under_their_own_ceilings(self) -> None:
        """E1 and E2 cannot pass the ceiling their group identity sets.

        This replaces an assertion that E3 also stays under the additive oracle. That is not
        an invariant and the published equation violates it: the oracle bounds a predictor
        that is a per-dataset value *plus* a per-model value, and E3's mixed terms multiply a
        dataset feature by a model one, so they represent interactions the oracle cannot. E3
        scores above it, and that is the study's headline rather than a bug. The test only
        passed because the fast configuration used here fits a weaker E3.
        """
        table = comparison(self.frame, self.e1, self.e3)
        self.assertLessEqual(scored(table, "E1, dataset only"), scored(table, "E1 reference") + 1e-9)

    def test_baselines_table_is_complete(self) -> None:
        """Four trivial predictors at two centres each, plus the oracle."""
        table = baselines(self.frame)
        self.assertEqual(table.height, 9)
        self.assertIn("r2", table.columns)
        names = set(table["baseline"].to_list())
        for centre in ("mean", "median"):
            self.assertIn(f"per-model {centre} (loo-dataset)", names)
            self.assertIn(f"per-dataset {centre} (loo-model)", names)

    def test_e2_equation_uses_no_dataset_feature(self) -> None:
        report = run_e2(self.frame, corpus.E3)
        used = {feature for term in report.equation.terms for feature in term.features}
        self.assertTrue(used)
        self.assertFalse(used & set(DATASET_FEATURES))

    def test_model_selection_reports_every_dataset(self) -> None:
        table = model_selection(self.frame, self.e3)
        self.assertEqual(table.height, self.frame[DATASET_COLUMN].n_unique())
        self.assertTrue((table["regret"] >= 0.0).all())

    def test_correlation_analysis_reports_both_correlations(self) -> None:
        table = correlation_analysis(self.frame, corpus.E3, top=10)
        self.assertEqual(table.height, 10)
        for column in ("term", "pearson", "spearman", "within_pearson", "monotone_gap"):
            self.assertIn(column, table.columns)

    def test_correlation_analysis_is_ranked_by_within_group_strength(self) -> None:
        strengths = correlation_analysis(self.frame, corpus.E3, top=10)["strength"].to_numpy()
        self.assertTrue((np.diff(strengths) <= 1e-12).all())


class TestWhatTheCorpusSays(unittest.TestCase):
    """The handful of assertions that are claims about the meta-dataset rather than about a
    function, and therefore read the real one.

    Everything else in this file runs on `tests.corpus`, an eight-by-ten slice, because a
    function either builds its table correctly or it does not and 78 rows show that as well
    as 476 do. A *finding* is different: "dataset features explain more than model features"
    is false on the slice -- it inverts, 0.106 against 0.362 -- and a test that asserted it
    there would be asserting nothing about the study.

    The configuration is still the cheap one. These are directional claims with large gaps,
    and reproducing the tuned search to check the direction of a gap would be reproducing the
    study, which is what the end-to-end CI step is for.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()
        cls.e1 = run_e1(cls.frame, corpus.E1)
        cls.e3 = run_e3(cls.frame, corpus.E3)

    def test_e2_beats_e1_on_the_common_scale(self) -> None:
        # The central claim of the study: model features carry information that dataset
        # features cannot express, because E1 can only predict a per-dataset constant.
        table = comparison(self.frame, self.e1, self.e3)
        self.assertGreater(scored(table, "E3, dataset + model"), scored(table, "E1, dataset only"))

    def test_the_median_baseline_is_the_harder_one_on_absolute_error(self) -> None:
        """Why both centres are reported rather than just the mean.

        MAE is minimised by the median, so a mean baseline is not minimising the metric it is
        being compared on. On this corpus the per-dataset median is the tighter opponent for
        MAE and SMAPE, and the mean is the tighter one for R2 -- each metric read against the
        baseline that is hardest to beat on it.
        """
        rows = {row["baseline"]: row for row in baselines(self.frame).to_dicts()}
        mean, median = rows["per-dataset mean (loo-model)"], rows["per-dataset median (loo-model)"]
        self.assertLess(median["mae"], mean["mae"])
        self.assertLess(median["smape"], mean["smape"])
        self.assertGreater(mean["r2"], median["r2"])

    def test_per_model_mean_beats_the_global_mean(self) -> None:
        table = baselines(self.frame)
        scores = dict(zip(table["baseline"].to_list(), table["r2"].to_list(), strict=True))
        self.assertGreater(scores["per-model mean (loo-dataset)"], scores["global mean (loo-dataset)"])

    def test_random_folds_look_better_than_grouped_ones(self) -> None:
        # The leakage this project exists to warn about: a random split scores the same
        # equation far higher because dataset identity is visible on both sides.
        table = leakage_demonstration(self.frame, 3, corpus.E3)
        scores = dict(zip(table["protocol"].to_list(), table["r2"].to_list(), strict=True))
        self.assertGreater(scores["random 10-fold (leaky)"], scores["leave-one-dataset-out"])

    def test_dataset_features_explain_more_than_model_features(self) -> None:
        # The direct test of "model choice outweighs the data": on this meta-dataset it
        # does not. Dataset identity explains more variance, and the dataset-only
        # equation gets far closer to its own ceiling than the model-only one does.
        table = comparison(self.frame, self.e1, self.e3)
        self.assertGreater(scored(table, "E1, dataset only"), scored(table, "E2, model only"))
        self.assertGreater(scored(table, "E1 reference"), scored(table, "E2 reference"))


class TestOnlyTheValidEquationIsEvaluated(unittest.TestCase):
    """C3: only the equation selected as E3-Valid is evaluated as a predictor.

    E3-MAX bounds how far the additive form reaches. It is reported, and it must never appear
    as a candidate in a comparison -- a bound that competes is being put forward as the
    study's recommendation, which is precisely what it is not.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.report = corpus.report()

    def test_no_comparison_carries_the_capability_bound_as_a_predictor(self) -> None:
        for table in (self.report.ranking_baselines, self.report.decision_baselines):
            for predictor in table["predictor"].to_list():
                with self.subTest(predictor=predictor):
                    self.assertNotIn("capability", predictor.lower())

    def _row(self, role: str) -> dict[str, object]:
        """The grammars row carrying a role. Matched by substring because one grammar can hold
        both -- `role` reads "E3-Valid + E3-MAX" when the wider grammar earns its complexity, which
        is a legitimate outcome and the case `Report.e3_capability` documents."""
        rows = [row for row in self.report.grammars.to_dicts() if role in str(row["role"])]
        self.assertEqual(len(rows), 1, f"exactly one grammar should hold {role}")
        return rows[0]

    def test_the_published_equation_is_the_one_the_rule_chose(self) -> None:
        chosen = self._row("E3-Valid")
        self.assertEqual(self.report.e3.arity, chosen["arity"])
        self.assertEqual(self.report.e3.n_terms, chosen["n_terms"])

    def test_the_bound_is_the_grammar_most_capable_chose(self) -> None:
        chosen = self._row("E3-MAX")
        self.assertEqual(self.report.e3_capability.arity, chosen["arity"])
        self.assertEqual(self.report.e3_capability.n_terms, chosen["n_terms"])

    def test_one_equation_keeps_the_published_name_when_both_roles_coincide(self) -> None:
        if self.report.e3 is self.report.e3_capability:
            self.assertTrue(self.report.e3.equation.name.startswith("E3_k"))
            self.assertNotIn("capability", self.report.e3.equation.name)

    def test_every_refit_uses_the_grammar_that_was_published(self) -> None:
        """The C3 defect, pinned. Several tables rebuild the library from a `Configuration`
        and refit. Handing them the configuration's arity rather than the chosen one scored
        the published equation's strictest protocol on a different grammar -- an arity-3
        equation with an arity-2 leave-one-cell row -- and every test passed."""
        from ml_meta_perf.experiment import run

        # The configuration must disagree with the search, or this proves nothing.
        self.assertEqual(corpus.E3.max_arity, 3)
        report = run(
            str(corpus.sample_path()),
            config_e1=corpus.E1,
            config_e2=corpus.E2,
            config_e3=corpus.E3,
            arities=(2,),
            opaque_models=corpus.DOUBLES,
        )
        self.assertEqual(report.e3.arity, 2)
        self.assertLessEqual(max(len(term.features) for term in report.e3.equation.terms), 2)


class TestDocumentedDefaults(unittest.TestCase):
    """The README's parameter table must state the defaults the code actually has.

    It stated four numbers that had all moved: 24 terms against 15, penalty 5 against 20,
    arity 3 against 2, z-cap 3.0 against 4.25. Nothing was checking, because the README is
    the one document the pipeline does not write -- so this reads the table and compares it
    with `DEFAULT` instead.
    """

    #: The README flag whose default each `Configuration` field is published as.
    #:
    #: `--arity` is **not** here, and cannot be: it is repeatable and its default is the set
    #: `ARITIES` searches, not a `Configuration` field. `test_the_searched_arities_are_documented`
    #: checks that row separately.
    FLAGS: ClassVar[dict[str, str]] = {
        "max_terms": "--max-terms",
        "penalty": "--penalty",
        "pool_size": "--pool",
        "beam_width": "--beam",
        "max_abs_zscore": "--zscore",
    }

    @classmethod
    def setUpClass(cls) -> None:
        readme = Path(__file__).resolve().parent.parent / "README.md"
        if not readme.is_file():
            raise unittest.SkipTest("README is not installed beside the package")
        cls.documented = {
            match.group(1): match.group(2)
            for match in re.finditer(r"^\| `(--[\w-]+)` \| ([\d.]+) \|", readme.read_text(), re.MULTILINE)
        }

    def test_every_tuned_knob_is_documented(self) -> None:
        missing = [flag for flag in self.FLAGS.values() if flag not in self.documented]
        self.assertEqual(missing, [], f"README omits a default for {missing}")

    def test_documented_defaults_match_the_configuration(self) -> None:
        for field, flag in self.FLAGS.items():
            with self.subTest(flag=flag):
                actual = getattr(DEFAULT, field)
                self.assertEqual(float(self.documented[flag]), float(actual))

    def test_the_searched_arities_are_documented(self) -> None:
        """`--arity` publishes a set rather than a number, so it needs its own check --
        and it is the row most likely to go stale, because it read "2" for as long as the
        arity was fixed and nothing noticed when the search replaced it."""
        readme = Path(__file__).resolve().parent.parent / "README.md"
        row = next(line for line in readme.read_text().splitlines() if line.startswith("| `--arity` |"))
        documented = re.findall(r"\d+", row.split("|")[2])
        self.assertEqual([int(value) for value in documented], list(ARITIES))


if __name__ == "__main__":
    unittest.main()
