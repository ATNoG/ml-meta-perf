"""Pins what `ml_meta_perf.data` offers and what the shipped study file describes.

A removal, a rename or an addition of a public name in the module fails the first test, so each
one is a decision and not an accident. The rest pin the facts the chapters state about the MCC
study's data description, which now lives in ``config/study.json``.
"""

import ast
import inspect
import unittest

import polars as pl

from ml_meta_perf import data
from ml_meta_perf.config import load_config
from ml_meta_perf.data import resolve_path

PUBLIC_NAMES = frozenset(
    {
        "SchemaError",
        "aggregate_by_dataset",
        "columns_as_arrays",
        "corpus_summary",
        "drop_constant_features",
        "groups",
        "feature_glossary",
        "load",
        "model_family",
        "model_ordinals",
        "neural_families",
        "tree_families",
        "resolve_path",
        "missing_cells",
        "target",
        "target_summary",
    }
)


def defined_names() -> set[str]:
    """The public names the module itself defines at top level (imports are not its surface)."""
    names: set[str] = set()
    for node in ast.parse(inspect.getsource(data)).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return {name for name in names if not name.startswith("_")}


class TestDataSurface(unittest.TestCase):
    def test_the_public_names_are_the_pinned_ones(self) -> None:
        defined = defined_names()
        self.assertEqual(PUBLIC_NAMES - defined, set(), "a pinned name was removed")
        self.assertEqual(defined - PUBLIC_NAMES, set(), "a public name was added: pin it, or make it private")


class TestTheShippedStudyFile(unittest.TestCase):
    """What ``config/study.json`` must describe for the MCC study to be the study it is.

    These replace the comparison against the old constants (removed with them, step 2.3): the
    description is read from the file, so what is pinned here is its shape and the facts the
    chapters state about it, not a second copy of its lists.
    """

    def setUp(self) -> None:
        self.schema = load_config().schema()

    def test_the_target_and_its_bounds(self) -> None:
        self.assertEqual((self.schema.target_column, self.schema.bounds), ("MCC", (-1.0, 1.0)))
        self.assertEqual(self.schema.transform, "none")
        self.assertIsNone(self.schema.row_filter)

    def test_the_groups(self) -> None:
        self.assertEqual((self.schema.dataset_column, self.schema.model_group), ("Dataset", "Model"))

    def test_twelve_dataset_features_and_six_model_features(self) -> None:
        self.assertEqual((len(self.schema.dataset_features), len(self.schema.model_features)), (12, 6))

    def test_the_two_feature_sets_do_not_overlap(self) -> None:
        self.assertEqual(set(self.schema.dataset_features) & set(self.schema.model_features), set())
        self.assertEqual(self.schema.features, self.schema.dataset_features + self.schema.model_features)

    def test_every_listed_column_is_in_the_corpus_and_nothing_else_is(self) -> None:
        columns = pl.read_csv(resolve_path(load_config().corpus.path)).columns
        expected = [
            self.schema.dataset_column,
            self.schema.model_group,
            *self.schema.features,
            self.schema.target_column,
        ]
        self.assertEqual(columns, expected, "the corpus carries exactly the columns the study file names")

    def test_the_corpus_has_the_pinned_shape(self) -> None:
        frame = data.load()
        self.assertEqual(frame.height, 484)
        self.assertEqual(frame[self.schema.dataset_column].n_unique(), 20)
        self.assertEqual(frame[self.schema.model_group].n_unique(), 25)


if __name__ == "__main__":
    unittest.main()
