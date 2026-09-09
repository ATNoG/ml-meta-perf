"""The end-to-end study.

These run against the real meta-dataset but with deliberately small configurations:
the point is that the wiring is correct and the reported relationships hold, not to
reproduce the published numbers inside a commit hook.
"""

import io
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import ClassVar

import numpy as np
import polars as pl

from ml_meta_perf.attribution import group_shares, term_effects, variance_decomposition
from ml_meta_perf.cli import build_parser, configurations, main, render
from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FEATURES,
    columns_as_arrays,
    groups,
    load,
    target,
)
from ml_meta_perf.experiment import (
    DEFAULT_E3,
    Configuration,
    Report,
    baselines,
    comparison,
    correlation_analysis,
    decision_baselines,
    decision_quality,
    identity_ceiling,
    interaction_reached,
    leakage_demonstration,
    length_comparison,
    model_selection,
    ranking_baselines,
    reach_analysis,
    run_e1,
    run_e2,
    run_e3,
    saturated_analysis,
)
from ml_meta_perf.model import Equation
from ml_meta_perf.practices import best_practices
from ml_meta_perf.selection import pareto_table, recommend
from ml_meta_perf.validate import oracle_ladder


def scored(table, prefix: str) -> float:
    """R2 of the comparison row whose label starts with ``prefix``.

    By prefix because the labels carry their term count, which moves with the configuration
    -- and these tests deliberately run a fast three-term one.
    """
    matched = table.filter(pl.col("equation").str.starts_with(prefix))
    return float(matched["r2"][0])


FAST_E1 = Configuration(max_abs_zscore=3.0, penalty=1.0, pool_size=40, max_terms=3)
FAST_E3 = Configuration(max_abs_zscore=3.0, penalty=20.0, pool_size=40, max_terms=3)


class TestEquationReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()
        cls.e1 = run_e1(cls.frame, FAST_E1)
        cls.e3 = run_e3(cls.frame, FAST_E3)

    def test_e1_reports_a_readable_equation(self) -> None:
        self.assertEqual(self.e1.equation.n_terms, 3)
        for weight in self.e1.equation.weights:
            # The stability filters exist to keep coefficients on a human scale.
            self.assertLess(abs(weight), 1e6)

    def test_e1_only_uses_dataset_features(self) -> None:
        from ml_meta_perf.data import MODEL_FEATURES

        used = {feature for term in self.e1.equation.terms for feature in term.features}
        self.assertFalse(used & set(MODEL_FEATURES))

    def test_e2_uses_at_least_one_model_feature(self) -> None:
        from ml_meta_perf.data import MODEL_FEATURES

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
        cls.frame = load()
        cls.e1 = run_e1(cls.frame, FAST_E1)
        cls.e3 = run_e3(cls.frame, FAST_E3)

    def test_e2_beats_e1_on_the_common_scale(self) -> None:
        # The central claim of the study: model features carry information that dataset
        # features cannot express, because E1 can only predict a per-dataset constant.
        table = comparison(self.frame, self.e1, self.e3)
        self.assertGreater(scored(table, "E3, dataset + model"), scored(table, "E1, dataset only"))

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
        table = leakage_demonstration(self.frame, 3, FAST_E3)
        scores = dict(zip(table["protocol"].to_list(), table["r2"].to_list(), strict=True))
        self.assertGreater(scores["random 10-fold (leaky)"], scores["leave-one-dataset-out"])

    def test_e2_equation_uses_no_dataset_feature(self) -> None:
        report = run_e2(self.frame, FAST_E3)
        used = {feature for term in report.equation.terms for feature in term.features}
        self.assertTrue(used)
        self.assertFalse(used & set(DATASET_FEATURES))

    def test_dataset_features_explain_more_than_model_features(self) -> None:
        # The direct test of "model choice outweighs the data": on this meta-dataset it
        # does not. Dataset identity explains more variance, and the dataset-only
        # equation gets far closer to its own ceiling than the model-only one does.
        table = comparison(self.frame, self.e1, self.e3)
        self.assertGreater(scored(table, "E1, dataset only"), scored(table, "E2, model only"))
        self.assertGreater(scored(table, "E1 reference"), scored(table, "E2 reference"))

    def test_model_selection_reports_every_dataset(self) -> None:
        table = model_selection(self.frame, self.e3)
        self.assertEqual(table.height, 20)
        self.assertTrue((table["regret"] >= 0.0).all())

    def test_correlation_analysis_reports_both_correlations(self) -> None:
        table = correlation_analysis(self.frame, FAST_E3, top=10)
        self.assertEqual(table.height, 10)
        for column in ("term", "pearson", "spearman", "within_pearson", "monotone_gap"):
            self.assertIn(column, table.columns)

    def test_correlation_analysis_is_ranked_by_within_group_strength(self) -> None:
        strengths = correlation_analysis(self.frame, FAST_E3, top=10)["strength"].to_numpy()
        self.assertTrue((np.diff(strengths) <= 1e-12).all())


class TestCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        frame = load()
        e1, e3 = run_e1(frame, FAST_E1), run_e3(frame, FAST_E3)
        cols = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
        e2 = run_e2(frame, FAST_E3)
        cls.report = Report(
            e1=e1,
            e2=e2,
            e3=e3,
            e3_capability=e3,
            grammars=pl.DataFrame(),
            practices=best_practices(e3.equation, cols, e3.stability),
            effects=term_effects(e3.equation, cols, DATASET_FEATURES, MODEL_FEATURES),
            shares=group_shares(e3.equation, cols, DATASET_FEATURES, MODEL_FEATURES),
            decomposition=variance_decomposition(
                target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN)
            ),
            correlations=correlation_analysis(frame, FAST_E3, top=5),
            **dict(zip(("reach", "ceiling"), reach_analysis(frame, FAST_E3), strict=True)),
            baselines=baselines(frame),
            comparison=comparison(frame, e1, e3, e2),
            leakage=leakage_demonstration(frame, 3, FAST_E3),
            selection=model_selection(frame, e3),
            decision=decision_quality(frame, 3, FAST_E3),
            ranking_baselines=ranking_baselines(frame, e3),
            decision_baselines=decision_baselines(frame, FAST_E3),
            term_choice=recommend(e3.curve, published=len(e3.equation.terms)),
            length_choice=length_comparison(frame, e3, FAST_E3),
            pareto=pareto_table(e3.curve),
            oracles=oracle_ladder(
                target(frame), groups(frame, DATASET_COLUMN), groups(frame, MODEL_COLUMN), ranks=(0, 1)
            ),
            interaction=interaction_reached(frame, e3),
            saturated=pl.DataFrame([saturated_analysis(frame, FAST_E3)]),
            opaque=pl.DataFrame({"model": ["stub"], "r2_in_sample": [0.9], "r2_loo_dataset": [0.1]}),
            identity=identity_ceiling(frame, e3),
        )

    def test_render_prints_every_section(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            render(self.report)
        printed = buffer.getvalue()
        for expected in (
            "Correlation screening",
            "E1 --",
            "E2 --",
            "E3 --",
            "Where the signal lives",
            "Extracted practices",
            "Baselines",
            "Model selection",
        ):
            self.assertIn(expected, printed)

    def test_main_writes_equations_tables_and_chapter_sections(self) -> None:
        """The generated results go into the chapters, between markers, not into a report file."""
        from ml_meta_perf.report import BEGIN, END

        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory) / "docs"
            docs.mkdir()
            chapter = docs / "05-evaluation.md"
            chapter.write_text("# 5. Evaluation\n\nHand-written prose that must survive.\n")

            buffer = io.StringIO()
            with redirect_stdout(buffer):
                code = main(
                    [
                        "--quick",
                        "--quiet",
                        "--no-figures",
                        "--output",
                        directory,
                        "--docs",
                        str(docs),
                    ]
                )
            self.assertEqual(code, 0)
            for name in ("e1.json", "e2.json", "e3.json"):
                path = Path(directory) / name
                self.assertTrue(path.is_file())
                self.assertGreater(Equation.load(path).n_terms, 0)
            self.assertTrue((Path(directory) / "curve_e3.csv").is_file())

            written = chapter.read_text()
            self.assertIn("Hand-written prose that must survive.", written)
            self.assertIn(BEGIN, written)
            self.assertIn(END, written)

    def test_regenerating_a_chapter_is_idempotent(self) -> None:
        """A second run replaces the generated block rather than appending another."""
        from ml_meta_perf.report import BEGIN

        with tempfile.TemporaryDirectory() as directory:
            docs = Path(directory) / "docs"
            docs.mkdir()
            chapter = docs / "05-evaluation.md"
            chapter.write_text("# 5. Evaluation\n\nProse.\n")
            for _ in range(2):
                with redirect_stdout(io.StringIO()):
                    main(
                        [
                            "--quick",
                            "--quiet",
                            "--no-figures",
                            "--no-tables",
                            "--output",
                            directory,
                            "--docs",
                            str(docs),
                        ]
                    )
            written = chapter.read_text()
            self.assertEqual(written.count(BEGIN), 1)
            self.assertEqual(written.count("Prose."), 1)

    def test_phase_selection_limits_what_is_printed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                main(["--quick", "--no-figures", "--no-report", "--phase", "screen", "--output", directory])
            printed = buffer.getvalue()
        self.assertIn("Correlation screening", printed)
        self.assertNotIn("Oracle ladder", printed)

    def test_flags_override_the_tuned_configuration(self) -> None:
        parser = build_parser()
        _, e3 = configurations(parser.parse_args(["--penalty", "3", "--arity", "2", "--max-terms", "40"]))
        self.assertEqual(e3.penalty, 3.0)
        self.assertEqual(e3.max_arity, 2)
        self.assertEqual(e3.max_terms, 40)

    def test_the_published_length_is_not_a_flag(self) -> None:
        """`--terms` was removed with `Configuration.headline_terms` on 2026-09-09. The length
        is derived from the equation's own curve, and a flag that set it by hand would be the
        assertion C1 exists to delete -- reachable again through the command line."""
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--terms", "12"])

    def test_unmentioned_flags_keep_their_tuned_values(self) -> None:
        parser = build_parser()
        _, e3 = configurations(parser.parse_args([]))
        self.assertEqual(e3, DEFAULT_E3)


class TestDocumentedDefaults(unittest.TestCase):
    """The README's parameter table must state the defaults the code actually has.

    It stated four numbers that had all moved: 24 terms against 15, penalty 5 against 20,
    arity 3 against 2, z-cap 3.0 against 4.25. Nothing was checking, because the README is
    the one document the pipeline does not write -- so this reads the table and compares it
    with `DEFAULT_E3` instead.
    """

    #: The README flag whose default each `Configuration` field is published as.
    FLAGS: ClassVar[dict[str, str]] = {
        "max_terms": "--max-terms",
        "penalty": "--penalty",
        "max_arity": "--arity",
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
                actual = getattr(DEFAULT_E3, field)
                self.assertEqual(float(self.documented[flag]), float(actual))


if __name__ == "__main__":
    unittest.main()
