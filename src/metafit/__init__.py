"""metafit -- interpretable linear meta-models for classifier performance.

Fits equations of the form ``MCC = w1*t1 + w2*t2 + ...`` where each term is a simple
expression over dataset and model meta-features, and reports what they are actually
worth under leave-one-dataset-out and leave-one-model-out validation.
"""

from metafit.analysis import redundancy_groups, screen
from metafit.attribution import group_shares, term_effects, variance_decomposition
from metafit.construct import agglomerate, constructed_library, structural_terms
from metafit.data import (
    ALL_FEATURES,
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FEATURES,
    TARGET_COLUMN,
    aggregate_by_dataset,
    columns_as_arrays,
    groups,
    load,
    target,
)
from metafit.fit import FitResult, fit
from metafit.model import Equation
from metafit.practices import best_practices, feature_practices
from metafit.selection import knee_terms, pareto_front, pareto_table, recommend
from metafit.terms import Atom, Library, Term, build_library
from metafit.validate import (
    CrossValidation,
    Scores,
    additive_oracle,
    baseline_group_mean,
    cross_validate,
    ranking_report,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_FEATURES",
    "DATASET_COLUMN",
    "DATASET_FEATURES",
    "MODEL_COLUMN",
    "MODEL_FEATURES",
    "TARGET_COLUMN",
    "Atom",
    "CrossValidation",
    "Equation",
    "FitResult",
    "Library",
    "Scores",
    "Term",
    "__version__",
    "additive_oracle",
    "agglomerate",
    "aggregate_by_dataset",
    "baseline_group_mean",
    "best_practices",
    "build_library",
    "columns_as_arrays",
    "constructed_library",
    "cross_validate",
    "feature_practices",
    "fit",
    "group_shares",
    "groups",
    "knee_terms",
    "load",
    "pareto_front",
    "pareto_table",
    "ranking_report",
    "recommend",
    "redundancy_groups",
    "screen",
    "structural_terms",
    "target",
    "term_effects",
    "variance_decomposition",
]
