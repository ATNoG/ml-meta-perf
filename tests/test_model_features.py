"""The properties `MODEL_FEATURES` has to keep for the grammar and the identification claim.

These are not style checks. Each one is a property some *other* part of the study relies on,
so the generated meta-dataset must keep satisfying them:

* the term grammar applies ``log``, ``sqrt`` and ``1/f``, all undefined at zero, so a
  zero-based column can only ever enter as ``f`` and ``f^2`` -- the same narrowness that rules
  out binary indicators;
* a rung with no rows is a level the corpus cannot speak about, and a gap in a ladder makes
  "one rung higher" mean different things at different points;
* the ordinal model descriptors must be stable properties of the learner, not accidental
  by-products of the dataset used in a row.

The corpus carries exactly the twelve dataset features, the six model features and the target.
"""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FEATURES,
    MODEL_ORDINALS,
    load,
)
from tests import corpus


class TestModelFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()

    def values(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy().astype(float)

    def test_every_model_feature_is_strictly_positive(self) -> None:
        """So every transform in the grammar is defined on all of them."""
        for name in MODEL_FEATURES:
            with self.subTest(name):
                self.assertGreater(float(self.values(name).min()), 0.0)

    def test_every_ordinal_is_a_gapless_ladder_from_one(self) -> None:
        for name in MODEL_ORDINALS:
            with self.subTest(name):
                levels = np.unique(self.values(name))
                self.assertEqual(levels.min(), 1.0)
                np.testing.assert_array_equal(levels, np.arange(1.0, levels.size + 1.0))

    def test_every_rung_is_occupied(self) -> None:
        """A level no learner takes is one the corpus cannot say anything about."""
        for name, table in MODEL_ORDINALS.items():
            with self.subTest(name):
                self.assertEqual(set(np.unique(self.values(name)).astype(int)), set(table.values()))

    def test_the_ordinals_are_constant_within_a_model(self) -> None:
        """They describe the learner, so they must not move when the dataset does."""
        models = self.frame[MODEL_COLUMN].to_numpy()
        for name in MODEL_ORDINALS:
            with self.subTest(name):
                values = self.values(name)
                for model in np.unique(models):
                    self.assertEqual(len(np.unique(values[models == model])), 1, model)

    def test_the_csv_matches_the_asserted_tables(self) -> None:
        """The columns are provenance-checked, not just present."""
        models = self.frame[MODEL_COLUMN].to_numpy()
        for name, table in MODEL_ORDINALS.items():
            with self.subTest(name):
                values = self.values(name)
                for model in np.unique(models):
                    self.assertEqual(float(table[str(model)]), values[models == model][0], model)

    def test_the_corpus_carries_only_the_features_in_use(self) -> None:
        """One dataset file, and nothing in it that no equation may draw on."""
        import polars as pl

        from ml_meta_perf.data import ALL_FEATURES, DATASET_COLUMN, DEFAULT_PATH, TARGET_COLUMN

        self.assertEqual(
            list(pl.read_csv(DEFAULT_PATH).columns),
            [DATASET_COLUMN, MODEL_COLUMN, *ALL_FEATURES, TARGET_COLUMN],
        )

    def test_every_learner_has_a_rung_in_every_ordinal(self) -> None:
        """No applicability sentinel is possible if nothing is ever missing."""
        models = {str(model) for model in self.frame[MODEL_COLUMN].unique()}
        for name, table in MODEL_ORDINALS.items():
            with self.subTest(name):
                self.assertEqual(models - set(table), set())


class TestPublishedEquationsAreReadable(unittest.TestCase):
    """No published equation states one relationship twice.

    The constraint lives in `fit.Selector`, and `tests/test_fit.py` pins it there against a
    synthetic library. This is the end-to-end guard: it runs the three equations the study
    actually publishes, *after* `fit.prune` has simplified them, because `terms.simplify` can
    rewrite a term into a different feature combination and the selector never sees that.
    """

    def repeated(self, equation) -> list[list[str]]:
        seen: dict[frozenset[str], list[str]] = {}
        for term in equation.terms:
            features = frozenset(term.features)
            if len(features) > 1:
                seen.setdefault(features, []).append(term.name)
        return [names for names in seen.values() if len(names) > 1]

    def test_no_equation_repeats_a_feature_combination(self) -> None:
        from ml_meta_perf.experiment import run_e1, run_e2

        frame = load()
        published = (("E1", run_e1(frame)), ("E2", run_e2(frame)), ("E3", corpus.published()))
        for label, report in published:
            with self.subTest(label):
                self.assertEqual(self.repeated(report.equation), [])


class TestIdentification(unittest.TestCase):
    """The corpus must name every dataset and every learner it contains.

    This is a requirement on the *corpus*, settled at design time and before any equation
    exists, and it is the reason the feature set is as wide as it is. If two datasets share
    a feature vector then no equation over those features can ever tell them apart, and a
    difference between them is unexplainable rather than merely unexplained.

    It is deliberately *not* a requirement on the equation, which uses 13 of the 18 and is
    expected to use fewer as it improves -- see `TestCompression`.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = load()

    def test_the_dataset_features_name_every_dataset(self) -> None:
        vectors = self.frame.select(list(DATASET_FEATURES)).unique().height
        self.assertEqual(vectors, self.frame[DATASET_COLUMN].n_unique())

    def test_the_model_features_name_every_learner_within_a_dataset(self) -> None:
        # Jointly, not standalone: `Processing Units Number` varies with the dataset, and
        # it is what separates the five colliding groups the other five leave ambiguous.
        ambiguous = 0
        for _, rows in self.frame.group_by(DATASET_COLUMN):
            counts = rows.group_by(list(MODEL_FEATURES)).len()
            ambiguous += int(counts.filter(pl.col("len") > 1)["len"].sum())
        self.assertEqual(ambiguous, 0)

    def test_the_dataset_independent_model_features_do_not_suffice_alone(self) -> None:
        # The honest caveat, pinned so the paper cannot overstate the claim: five of the six
        # are constant per model and separate only 19 of the 25 learners on their own.
        constant = [name for name in MODEL_FEATURES if name != "Processing Units Number"]
        distinct = self.frame.select(constant).unique().height
        self.assertLess(distinct, self.frame[MODEL_COLUMN].n_unique())

    def test_dropping_the_identification_only_columns_costs_identification(self) -> None:
        # `Solution Stochasticity` and `Loss Margin Behaviour` add little to a fit and earn
        # their place in the corpus anyway: without them some learners stop being
        # distinguishable on the same dataset.
        reduced = [name for name in MODEL_FEATURES if name not in ("Solution Stochasticity", "Loss Margin Behaviour")]
        ambiguous = 0
        for _, rows in self.frame.group_by(DATASET_COLUMN):
            counts = rows.group_by(reduced).len()
            ambiguous += int(counts.filter(pl.col("len") > 1)["len"].sum())
        self.assertGreater(ambiguous, 0)


class TestCompression(unittest.TestCase):
    """The equation is allowed to use fewer features than the corpus carries.

    Identification and compression are different stages with different criteria, and this
    is the test that says the second is not held to the first. An equation that used all
    eighteen would not be a better equation; it would be one that had failed to generalise
    over families of learners and datasets.
    """

    def test_the_equation_uses_fewer_features_than_the_corpus_carries(self) -> None:

        equation = corpus.published().equation
        used = {feature for term in equation.terms for feature in term.features}
        available = set(DATASET_FEATURES) | set(MODEL_FEATURES)
        self.assertTrue(used <= available)
        self.assertLess(len(used), len(available))


if __name__ == "__main__":
    unittest.main()


class TestReportedProtocol(unittest.TestCase):
    """Every reported cross-validated number comes from the fixed form.

    The study fits one equation and recalibrates only its weights per fold. Re-selecting the
    terms inside each fold answers a question about the search rather than about the equation,
    and it is used for nothing but `CrossValidation.stability`. This is pinned as a test
    because the two paths differ by up to 0.3 R2 and a silent swap would be invisible.
    """

    def test_run_equation_reports_the_fixed_form(self) -> None:
        from ml_meta_perf.config import default_configuration
        from ml_meta_perf.data import DATASET_FEATURES, columns_as_arrays, groups, target
        from ml_meta_perf.terms import build_library
        from ml_meta_perf.validate import cross_validate_fixed_form

        frame = load()
        report = corpus.published()
        config = default_configuration()
        columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
        library = build_library(
            DATASET_FEATURES,
            MODEL_FEATURES,
            columns,
            max_arity=config.max_arity,
            max_abs_zscore=config.max_abs_zscore,
        )
        size = len(report.equation.terms)
        direct = cross_validate_fixed_form(
            library,
            columns,
            target(frame),
            groups(frame, "Dataset"),
            {size: report.equation},
            penalty=config.penalty,
        )
        self.assertAlmostEqual(
            report.cross_validated["loo_dataset"]["r2"],
            direct[size].scores(target(frame)).r2,
            places=6,
        )
