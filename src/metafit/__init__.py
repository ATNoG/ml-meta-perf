"""metafit -- interpretable linear meta-models for classifier performance.

Fits equations of the form ``MCC = w1*t1 + w2*t2 + ...`` where each term is a simple
expression over dataset and model meta-features, and reports what they are actually
worth under leave-one-dataset-out and leave-one-model-out validation.
"""

from metafit.analysis import redundancy_groups, screen
from metafit.attribution import group_shares, term_effects, variance_decomposition
from metafit.construct import (
    agglomerate,
    cluster_terms,
    constructed_library,
    guided_merge,
    structural_terms,
)
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
from metafit.fit import FitResult, fit, prune
from metafit.guidance import CATALOGUE, Practice, Verdict, assess
from metafit.model import Equation
from metafit.practices import best_practices, concordance, concordance_summary, feature_practices
from metafit.report import (
    coverage,
    feature_usage,
    marginal_versus_conditional,
    operation_usage,
    term_groups,
    term_importance,
    term_sentences,
    unstable_majors,
)
from metafit.selection import knee_index, knee_terms, pareto_front, pareto_table, recommend, simplify_curve
from metafit.terms import Atom, Library, Term, build_library, ratio_of_sums_terms, simplify
from metafit.validate import (
    CrossValidation,
    Scores,
    additive_oracle,
    baseline_group_mean,
    cross_validate,
    decision_report,
    ranking_report,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_FEATURES",
    "CATALOGUE",
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
    "Practice",
    "Scores",
    "Term",
    "Verdict",
    "__version__",
    "additive_oracle",
    "agglomerate",
    "aggregate_by_dataset",
    "assess",
    "baseline_group_mean",
    "best_practices",
    "build_library",
    "cluster_terms",
    "columns_as_arrays",
    "concordance",
    "concordance_summary",
    "constructed_library",
    "coverage",
    "cross_validate",
    "decision_report",
    "feature_practices",
    "feature_usage",
    "fit",
    "group_shares",
    "groups",
    "guided_merge",
    "knee_index",
    "knee_terms",
    "load",
    "marginal_versus_conditional",
    "operation_usage",
    "pareto_front",
    "pareto_table",
    "prune",
    "ranking_report",
    "ratio_of_sums_terms",
    "recommend",
    "redundancy_groups",
    "screen",
    "simplify",
    "simplify_curve",
    "structural_terms",
    "target",
    "term_effects",
    "term_groups",
    "term_importance",
    "term_sentences",
    "unstable_majors",
    "variance_decomposition",
]
