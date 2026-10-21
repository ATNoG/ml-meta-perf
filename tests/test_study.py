"""Running a study: the equations its file lists are the equations the per-equation functions fit."""

import dataclasses
import unittest

import numpy as np

from ml_meta_perf.config import E3_MAX, load_config
from ml_meta_perf.experiment import EquationReport, run_capability, run_e1, run_e2, run_e3
from ml_meta_perf.study import VARIANT_NAMES, run_study
from tests import corpus

STUDY = dataclasses.replace(
    load_config(),
    equations=("E1", "E2", "E3", E3_MAX),
    search=dataclasses.replace(corpus.E3, max_terms=3, pool_size=30, beam_width=1, max_arity=2),
    selection=dataclasses.replace(corpus.SELECTION, window=1, capability_arity=2),
)


def same(a: EquationReport, b: EquationReport) -> bool:
    return (
        a.n_terms == b.n_terms
        and a.arity == b.arity
        and [t.name for t in a.equation.terms] == [t.name for t in b.equation.terms]
        and np.allclose(a.equation.weights, b.equation.weights, rtol=0, atol=0)
        and a.curve.equals(b.curve)
    )


class TestRunStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = corpus.sample()
        cls.report = run_study(STUDY, corpus.sample_path())

    def test_every_equation_the_file_lists_is_fitted_and_named(self) -> None:
        self.assertEqual(list(self.report.equations), ["E1", "E2", "E3", E3_MAX])
        self.assertEqual(self.report.valid, self.report.equations["E3"])
        self.assertEqual(self.report.maximum, self.report.equations[E3_MAX])

    def test_the_equations_equal_the_ones_the_per_equation_functions_fit(self) -> None:
        # E1, E2 and E3 share one configuration (the suite's E3 here), as the study does.
        config, selection = STUDY.search, STUDY.selection
        direct = {
            "E1": run_e1(self.frame, config, selection),
            "E2": run_e2(self.frame, config, selection),
            "E3": run_e3(self.frame, config, selection),
            E3_MAX: run_capability(self.frame, config, selection),
        }
        for name, expected in direct.items():
            with self.subTest(name):
                self.assertTrue(same(self.report.equations[name], expected))

    def test_the_scores_table_has_every_variant_protocol_and_scale(self) -> None:
        scores = self.report.scores
        self.assertEqual(set(scores["variant"]), {VARIANT_NAMES[n] for n in STUDY.equations})
        self.assertEqual(set(scores["protocol"]), {"IS", "LODO", "LOMO", "DHO"})
        self.assertEqual(set(scores["scale"]), {"fit", "original"})
        self.assertEqual(scores.height, 4 * 4 * 2)

    def test_the_effects_and_shares_are_those_of_e3_valid(self) -> None:
        self.assertEqual(self.report.effects.height, len(self.report.valid.equation.terms))
        self.assertGreater(self.report.shares.height, 0)

    def test_a_study_of_one_equation_fits_one(self) -> None:
        one = run_study(dataclasses.replace(STUDY, equations=("E3",)), corpus.sample_path())
        self.assertEqual(list(one.equations), ["E3"])
        self.assertIsNone(one.maximum)
        self.assertTrue(same(one.valid, self.report.valid))


if __name__ == "__main__":
    unittest.main()
