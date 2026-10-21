"""Fitting one equation: the wiring is correct and the equations are readable.

These run on a slice of the shipped corpus with deliberately small configurations: the point is that
the pieces fit together, not to reproduce published numbers inside a commit hook.
"""

import dataclasses
import re
import unittest
from pathlib import Path

from ml_meta_perf.config import load_config
from ml_meta_perf.experiment import run_equation
from ml_meta_perf.terms import simplify
from tests import corpus
from tests.study import DATASET_FEATURES, MODEL_FEATURES


class TestEquationReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = corpus.sample()
        cls.e1 = run_equation(cls.frame, DATASET_FEATURES, (), corpus.E1, corpus.SELECTION, "E1")
        cls.e3 = run_equation(cls.frame, DATASET_FEATURES, MODEL_FEATURES, corpus.E3, corpus.SELECTION, "E3")

    def test_e1_reports_a_readable_equation(self) -> None:
        # A count, not a number: pruning decides the length and the search horizon bounds it.
        self.assertGreater(self.e1.equation.n_terms, 0)
        self.assertLessEqual(self.e1.equation.n_terms, corpus.E1.max_terms)
        for weight in self.e1.equation.weights:
            # The stability filters exist to keep coefficients on a human scale.
            self.assertLess(abs(weight), 1e6)

    def test_e1_only_uses_dataset_features(self) -> None:
        used = {feature for term in self.e1.equation.terms for feature in term.features}
        self.assertFalse(used & set(MODEL_FEATURES))

    def test_e3_uses_at_least_one_model_feature(self) -> None:
        used = {feature for term in self.e3.equation.terms for feature in term.features}
        self.assertTrue(used & set(MODEL_FEATURES))

    def test_curves_cover_the_requested_sizes(self) -> None:
        self.assertGreater(self.e1.curve.height, 0)
        self.assertIn("r2_in_sample", self.e1.curve.columns)
        self.assertIn("r2_loo_dataset", self.e3.curve.columns)
        self.assertIn("r2_loo_model", self.e3.curve.columns)
        self.assertIn("r2_loo_cell", self.e3.curve.columns)

    def test_cross_validated_scores_are_reported_for_every_protocol(self) -> None:
        """Both single-group protocols and the doubly held out (DHO) protocol are reported."""
        self.assertEqual(set(self.e3.cross_validated), {"loo_dataset", "loo_model", "loo_cell"})

    def test_cross_validated_scores_are_finite_and_bounded(self) -> None:
        # Deliberately *not* asserting cross-validated <= in-sample (IS): the fold equations can
        # beat a single heavily constrained one, and the ordering is not an invariant of the method.
        for scores in self.e3.cross_validated.values():
            self.assertTrue(-10.0 < scores["r2"] <= 1.0)

    def test_stability_table_is_populated(self) -> None:
        self.assertIsNotNone(self.e3.stability)
        assert self.e3.stability is not None
        self.assertIn("frequency", self.e3.stability.columns)

    def test_published_equations_are_already_simplified(self) -> None:
        # A term that `simplify` can still shorten is a defect: `([a] / [b]) * [b]` says `a` in six symbols.
        for equation in (self.e1.equation, self.e3.equation):
            for term in equation.terms:
                self.assertEqual(simplify(term), term, f"{term.name} is still reducible")

    def test_published_equations_carry_no_duplicate_terms(self) -> None:
        # Two terms that simplify to the same expression are collinear: the pair reads as two
        # pieces of evidence when it is one.
        for equation in (self.e1.equation, self.e3.equation):
            names = [term.name for term in equation.terms]
            self.assertEqual(len(names), len(set(names)))


class TestDocumentedDefaults(unittest.TestCase):
    """The README's configuration table must state the values ``config/study.json`` holds.

    A README table of defaults once stated four numbers that had all moved, and nothing was
    checking because the README is the one document the pipeline does not write. The values
    now live in one file; this reads the README's table and compares it with that file.
    """

    @classmethod
    def setUpClass(cls) -> None:
        readme = Path(__file__).resolve().parent.parent / "README.md"
        if not readme.is_file():
            raise unittest.SkipTest("README is not installed beside the package")
        cls.documented = {
            match.group(1): match.group(2)
            for match in re.finditer(
                r"^\| `((?:search|selection)\.\w+)` \| ([\w.]+) \|", readme.read_text(), re.MULTILINE
            )
        }
        cls.study = load_config()

    def test_every_hyperparameter_is_documented(self) -> None:
        expected = {
            f"{section}.{field.name}"
            for section in ("search", "selection")
            for field in dataclasses.fields(getattr(self.study, section))
        }
        self.assertEqual(sorted(expected - set(self.documented)), [], "README omits a hyperparameter")

    def test_documented_values_match_the_configuration_file(self) -> None:
        for key, value in self.documented.items():
            section, name = key.split(".")
            with self.subTest(key=key):
                actual = getattr(getattr(self.study, section), name)
                self.assertEqual(
                    value if isinstance(actual, str) else float(value),
                    actual if isinstance(actual, str) else float(actual),
                )


class TestReportedProtocol(unittest.TestCase):
    """Every reported cross-validated number comes from the fixed form: the terms are chosen once.

    Re-selecting the terms inside each fold answers a question about the search rather than about the
    equation, and the two differ by up to 0.3 R2, so a silent swap would be invisible.
    """

    def test_run_equation_reports_the_fixed_form(self) -> None:
        from ml_meta_perf.data import columns_as_arrays, groups, target
        from ml_meta_perf.terms import build_library
        from ml_meta_perf.validate import cross_validate_fixed_form

        frame = corpus.sample()
        small = dataclasses.replace(corpus.E3, max_terms=3, pool_size=30, beam_width=1, max_arity=2)
        report = run_equation(frame, DATASET_FEATURES, MODEL_FEATURES, small, corpus.SELECTION, "E3")
        columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
        library = build_library(
            DATASET_FEATURES,
            MODEL_FEATURES,
            columns,
            max_arity=small.max_arity,
            max_abs_zscore=small.max_abs_zscore,
        )
        size = len(report.equation.terms)
        direct = cross_validate_fixed_form(
            library,
            columns,
            target(frame),
            groups(frame, "Dataset"),
            {size: report.equation},
            penalty=corpus.E3.penalty,
        )
        self.assertAlmostEqual(
            report.cross_validated["loo_dataset"]["r2"], direct[size].scores(target(frame)).r2, places=6
        )
