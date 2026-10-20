"""Pins what `ml_meta_perf.data` offers and what the MCC schema contains.

The study description is moving out of the module and into ``config/study.json``. Until that has
happened, these tests fail on any removal or rename of a public name, or on any change to the MCC
columns, so each one is a decision and not an accident. Once the move is done they become the
equivalence test: the schema read from the study file must equal the one pinned here.
"""

import ast
import inspect
import unittest

from ml_meta_perf import data
from ml_meta_perf.config import load_config

PUBLIC_NAMES = frozenset(
    {
        "ALL_FEATURES",
        "DATASET_COLUMN",
        "DATASET_FEATURES",
        "FEATURE_GLOSSARY",
        "FITTING_REGIME",
        "FLEXFL_COST_TARGETS",
        "FLEXFL_DATASET_COLUMN",
        "FLEXFL_DATASET_FEATURES",
        "FLEXFL_DECOMPOSITION_TARGETS",
        "FLEXFL_EARLY_STOP_RULE_COLUMNS",
        "FLEXFL_EPOCH_CAP_COLUMN",
        "FLEXFL_MODEL_COLUMN",
        "FLEXFL_MODEL_FEATURES",
        "FLEXFL_PERFORMANCE_BOUNDS",
        "FLEXFL_TARGETS",
        "INPUT_DISTRIBUTION_MODELLING",
        "LOSS_MARGIN_BEHAVIOUR",
        "MCC_SCHEMA",
        "MODEL_CAPABILITY",
        "MODEL_COLUMN",
        "MODEL_FAMILY",
        "MODEL_FEATURES",
        "MODEL_ORDINALS",
        "NEURAL_FAMILIES",
        "SMAPE_LOWER",
        "SMAPE_UPPER",
        "SOLUTION_STOCHASTICITY",
        "TARGET_COLUMN",
        "TASK_TYPES",
        "TREE_FAMILIES",
        "SchemaError",
        "add_target_arguments",
        "aggregate_by_dataset",
        "check_target_arguments",
        "columns_as_arrays",
        "corpus_summary",
        "drop_constant_features",
        "flexfl_schema",
        "groups",
        "load",
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


class TestMccSchemaIsFrozen(unittest.TestCase):
    """TEMPORARY ORACLE: the pinned MCC columns, to be replaced by fixed expectations on the study file (2.3)."""

    def test_the_study_file_describes_the_same_schema(self) -> None:
        """TEMPORARY ORACLE (removed with the MCC constants, step 2.3): file == constants."""
        self.assertEqual(load_config().schema(), data.MCC_SCHEMA)

    def test_columns(self) -> None:
        schema = data.MCC_SCHEMA
        self.assertEqual(
            (schema.dataset_column, schema.model_column, schema.target_column), ("Dataset", "Model", "MCC")
        )
        self.assertEqual(schema.bounds, (-1.0, 1.0))
        self.assertIsNone(schema.task_type)
        self.assertFalse(schema.log_target)

    def test_dataset_features(self) -> None:
        self.assertEqual(
            data.MCC_SCHEMA.dataset_features,
            (
                "class_ent",
                "eq_num_attr",
                "gravity",
                "inst_to_attr",
                "nr_attr",
                "nr_bin",
                "nr_class",
                "nr_cor_attr",
                "nr_inst",
                "nr_norm",
                "nr_outliers",
                "ns_ratio",
            ),
        )

    def test_model_features(self) -> None:
        self.assertEqual(
            data.MCC_SCHEMA.model_features,
            (
                "Processing Units Number",
                "Model Capability",
                "Solution Stochasticity",
                "Loss Margin Behaviour",
                "Input Distribution Modelling",
                "Fitting Regime",
            ),
        )

    def test_the_two_feature_sets_do_not_overlap(self) -> None:
        schema = data.MCC_SCHEMA
        self.assertEqual(set(schema.dataset_features) & set(schema.model_features), set())
        self.assertEqual(schema.features, data.ALL_FEATURES)

    def test_the_corpus_has_the_pinned_shape(self) -> None:
        frame = data.load()
        self.assertEqual(frame.height, 484)
        self.assertEqual(frame[data.DATASET_COLUMN].n_unique(), 20)
        self.assertEqual(frame[data.MODEL_COLUMN].n_unique(), 25)


if __name__ == "__main__":
    unittest.main()
