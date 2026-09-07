"""Weighing literature best practices against the study.

The claim under test is that the *verdicts* are measured rather than asserted: change
what the study found and the verdict has to change with it.
"""

import unittest

import numpy as np
import polars as pl

from ml_meta_perf.data import DATASET_COLUMN, MODEL_FAMILY, NEURAL_FAMILIES, TREE_FAMILIES, load
from ml_meta_perf.guidance import (
    CATALOGUE,
    CHALLENGED,
    CHECKS,
    NOT_TESTED,
    QUALIFIED,
    SUPPORTED,
    as_table,
    assess,
    gather,
    render,
)


def frame() -> pl.DataFrame:
    """A tiny meta-dataset: three datasets by four models, one row per cell."""
    generator = np.random.default_rng(0)
    datasets, models = ("a", "b", "c"), ("DT", "MLP", "XGBoost", "LR")
    return pl.DataFrame(
        {
            "Dataset": [dataset for dataset in datasets for _ in models],
            "Model": [model for _ in datasets for model in models],
            "MCC": generator.uniform(0.0, 1.0, len(datasets) * len(models)),
        }
    )


def ranking_table(
    ap: float, mrr: float, hit_at_1: float, regret: float, spearman: float = 0.5
) -> pl.DataFrame:
    """One ranking row per dataset of the real corpus, all identical.

    Per-group rather than a single mean, because `_beat_the_trivial_baseline` pairs the
    equation against the baseline fold by fold -- a difference of two means over twenty
    folds is what that check was rewritten to stop reading. The row count is taken from the
    corpus rather than written down, since the pairing refuses tables of unequal height and
    a hard-coded 20 would turn a corpus change into a confusing `not tested`.
    """
    names = sorted(str(name) for name in load()[DATASET_COLUMN].unique())
    return pl.DataFrame(
        {
            "group": names,
            "ap": [ap] * len(names),
            "mrr": [mrr] * len(names),
            "hit_at_1": [hit_at_1] * len(names),
            "regret": [regret] * len(names),
            "spearman": [spearman] * len(names),
        }
    )


class _Report:
    """The handful of tables `gather` reads, with nothing else attached."""

    def __init__(self, **overrides: pl.DataFrame) -> None:
        self.decomposition = overrides.get(
            "decomposition",
            pl.DataFrame(
                {
                    "knowing only": ["dataset identity", "model identity"],
                    "n_groups": [20, 25],
                    "variance_explained": [0.354, 0.282],
                }
            ),
        )
        self.comparison = overrides.get(
            "comparison",
            pl.DataFrame(
                {
                    "equation": ["E1 (dataset only)", "E2 (model only)", "E3 (dataset + model)"],
                    "r2": [0.337, 0.166, 0.600],
                }
            ),
        )
        self.leakage = overrides.get(
            "leakage",
            pl.DataFrame(
                {
                    "protocol": ["random 10-fold (leaky)", "leave-one-dataset-out"],
                    "r2": [0.540, 0.466],
                }
            ),
        )
        self.practices = overrides.get(
            "practices",
            pl.DataFrame(
                {
                    "feature": ["Robust to Outliers", "ns_ratio"],
                    "effect": [0.42, -0.17],
                }
            ),
        )
        self.selection = overrides.get("selection", ranking_table(0.62, 0.70, 0.55, 0.019, 0.648))


class TestCatalogue(unittest.TestCase):
    def test_every_practice_has_a_check(self) -> None:
        self.assertEqual({practice.id for practice in CATALOGUE}, set(CHECKS))

    def test_ids_are_unique(self) -> None:
        ids = [practice.id for practice in CATALOGUE]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_practice_cites_a_source(self) -> None:
        for practice in CATALOGUE:
            self.assertTrue(practice.source.strip(), practice.id)
            self.assertTrue(practice.rationale.strip(), practice.id)

    def test_statements_are_general_not_feature_level(self) -> None:
        # The point of this module: a practice must not name a meta-feature column. If it
        # does, it is a measurement wearing advice's clothes.
        columns = {"nr_norm", "ns_ratio", "class_ent", "eq_num_attr", "nr_cor_attr", "gravity"}
        for practice in CATALOGUE:
            for column in columns:
                self.assertNotIn(column, practice.statement, practice.id)


class TestVerdicts(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = load()
        self.report = _Report()

    def test_every_practice_is_assessed(self) -> None:
        verdicts = assess(self.frame, self.report)  # pyright: ignore[reportArgumentType]
        self.assertEqual(len(verdicts), len(CATALOGUE))
        for verdict in verdicts:
            self.assertIn(verdict.verdict, {SUPPORTED, QUALIFIED, CHALLENGED, NOT_TESTED})
            self.assertTrue(verdict.evidence.strip())

    def test_leakage_verdict_follows_the_measured_inflation(self) -> None:
        check = CHECKS["hold-out-whole-groups"]
        wide = gather(self.frame, self.report)  # pyright: ignore[reportArgumentType]
        self.assertEqual(check(wide).verdict, SUPPORTED)

        narrow = _Report(
            leakage=pl.DataFrame(
                {"protocol": ["random 10-fold (leaky)", "leave-one-dataset-out"], "r2": [0.470, 0.466]}
            )
        )
        evidence = gather(self.frame, narrow)  # pyright: ignore[reportArgumentType]
        self.assertEqual(check(evidence).verdict, QUALIFIED)

    def test_a_missing_practice_row_yields_not_tested_rather_than_a_guess(self) -> None:
        check = CHECKS["prefer-outlier-robust-learners"]
        empty = _Report(practices=pl.DataFrame({"feature": [], "effect": []}))
        evidence = gather(self.frame, empty)  # pyright: ignore[reportArgumentType]
        verdict = check(evidence)
        self.assertEqual(verdict.verdict, NOT_TESTED)
        self.assertTrue(np.isnan(verdict.magnitude))

    def test_the_metric_practice_is_honest_about_not_being_tested(self) -> None:
        # The study adopts MCC and never compares it to an alternative, so it cannot be
        # evidence for the practice however convenient that would be.
        check = CHECKS["use-a-balanced-metric"]
        evidence = gather(self.frame, self.report)  # pyright: ignore[reportArgumentType]
        self.assertEqual(check(evidence).verdict, NOT_TESTED)

    def test_the_baseline_verdict_follows_which_side_wins(self) -> None:
        check = CHECKS["beat-the-trivial-baseline"]
        losing = gather(self.frame, self.report)  # pyright: ignore[reportArgumentType]
        self.assertEqual(check(losing).verdict, SUPPORTED)

        strong = _Report(selection=ranking_table(0.99, 0.99, 0.99, 0.0))
        winning = gather(self.frame, strong)  # pyright: ignore[reportArgumentType]
        self.assertEqual(check(winning).verdict, CHALLENGED)

    def test_the_baseline_verdict_ignores_spearman(self) -> None:
        # Spearman is the same to within a thousandth for every predictor on this corpus,
        # so a verdict that moved with it would be reading noise. Only the head-weighted
        # metrics may decide, and this pins that by moving Spearman alone.
        check = CHECKS["beat-the-trivial-baseline"]
        low = _Report(selection=ranking_table(0.62, 0.70, 0.55, 0.019, spearman=0.10))
        high = _Report(selection=ranking_table(0.62, 0.70, 0.55, 0.019, spearman=0.99))
        self.assertEqual(
            check(gather(self.frame, low)).verdict,  # pyright: ignore[reportArgumentType]
            check(gather(self.frame, high)).verdict,  # pyright: ignore[reportArgumentType]
        )

    def test_an_unresolvable_difference_is_qualified_not_a_challenge(self) -> None:
        # The practice claims the trivial baseline is competitive. A margin this corpus
        # cannot resolve is that claim holding, not a failure to measure -- and it must not
        # be reported as the equation overturning the practice.
        check = CHECKS["beat-the-trivial-baseline"]
        evidence = gather(self.frame, self.report)  # pyright: ignore[reportArgumentType]
        baseline = evidence.baseline_ranking
        # Equal on two metrics and a hair ahead on two: no interval can exclude zero.
        narrow = _Report(
            selection=ranking_table(
                baseline["ap"], baseline["mrr"], baseline["hit_at_1"], baseline["regret"] - 1e-6
            )
        )
        self.assertEqual(
            check(gather(self.frame, narrow)).verdict,  # pyright: ignore[reportArgumentType]
            QUALIFIED,
        )

    def test_a_missing_ranking_table_is_not_tested(self) -> None:
        check = CHECKS["beat-the-trivial-baseline"]
        empty = _Report(selection=pl.DataFrame({"group": ["a"], "ap": [0.5]}))
        self.assertEqual(
            check(gather(self.frame, empty)).verdict,  # pyright: ignore[reportArgumentType]
            NOT_TESTED,
        )

    def test_tree_verdict_is_measured_from_the_families(self) -> None:
        check = CHECKS["tree-ensembles-first"]
        evidence = gather(self.frame, self.report)  # pyright: ignore[reportArgumentType]
        verdict = check(evidence)
        self.assertEqual(verdict.verdict, SUPPORTED)
        # The magnitude is the tree-minus-neural gap, so it has to be positive and large.
        self.assertGreater(verdict.magnitude, 0.05)


class TestEvidence(unittest.TestCase):
    def test_family_means_use_the_complete_grid(self) -> None:
        evidence = gather(load(), _Report())  # pyright: ignore[reportArgumentType]
        # Eight models are missing from three datasets, so the two tables differ.
        self.assertLess(evidence.n_complete_datasets, load()["Dataset"].n_unique())
        self.assertGreater(evidence.missing_cells, 0)

    def test_ranking_agreement_is_reported_rather_than_assumed(self) -> None:
        evidence = gather(load(), _Report())  # pyright: ignore[reportArgumentType]
        self.assertGreater(evidence.ranking_agreement, 0.9)
        self.assertLessEqual(evidence.ranking_agreement, 1.0)

    def test_a_complete_grid_leaves_nothing_missing(self) -> None:
        evidence = gather(frame(), _Report())  # pyright: ignore[reportArgumentType]
        self.assertEqual(evidence.missing_cells, 0)
        self.assertEqual(evidence.n_complete_datasets, 3)

    def test_family_lookup_returns_nan_for_an_absent_family(self) -> None:
        evidence = gather(frame(), _Report())  # pyright: ignore[reportArgumentType]
        self.assertTrue(np.isnan(evidence.family("tabular foundation")))


class TestTaxonomy(unittest.TestCase):
    def test_every_model_in_the_meta_dataset_has_a_family(self) -> None:
        models = set(load()["Model"].unique().to_list())
        self.assertEqual(models - set(MODEL_FAMILY), set())

    def test_tree_and_neural_families_are_disjoint_and_present(self) -> None:
        self.assertEqual(set(TREE_FAMILIES) & set(NEURAL_FAMILIES), set())
        assigned = set(MODEL_FAMILY.values())
        for family in TREE_FAMILIES + NEURAL_FAMILIES:
            self.assertIn(family, assigned)


class TestRendering(unittest.TestCase):
    def setUp(self) -> None:
        self.verdicts = assess(load(), _Report())  # pyright: ignore[reportArgumentType]

    def test_table_has_one_row_per_practice(self) -> None:
        table = as_table(self.verdicts)
        self.assertEqual(table.height, len(CATALOGUE))
        for column in ("id", "practice", "verdict", "magnitude", "source"):
            self.assertIn(column, table.columns)

    def test_render_counts_the_verdicts(self) -> None:
        text = render(self.verdicts)
        self.assertIn(f"{len(CATALOGUE)} practices assessed", text)
        for verdict in self.verdicts:
            self.assertIn(verdict.practice.statement, text)

    def test_empty_input_is_handled(self) -> None:
        self.assertEqual(as_table([]).height, 0)
        self.assertIn("No practices", render([]))


if __name__ == "__main__":
    unittest.main()
