"""ml-meta-perf -- interpretable linear meta-models for classifier performance.

Fits equations of the form ``MCC = w1*t1 + w2*t2 + ...`` where each term is a simple
expression over dataset and model meta-features, and reports what they are actually
worth under leave-one-dataset-out and leave-one-model-out validation.
"""

from ml_meta_perf.analysis import redundancy_groups, screen
from ml_meta_perf.attribution import group_shares, term_effects, variance_decomposition
from ml_meta_perf.data import (
    ALL_FEATURES,
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_CAPABILITY,
    MODEL_COLUMN,
    MODEL_FAMILY,
    MODEL_FEATURES,
    TARGET_COLUMN,
    aggregate_by_dataset,
    columns_as_arrays,
    groups,
    load,
    target,
)
from ml_meta_perf.fit import FitResult, fit, prune
from ml_meta_perf.guidance import CATALOGUE, Practice, Verdict, assess
from ml_meta_perf.identity import ModelEffects, correct_out_of_fold, fit_effects
from ml_meta_perf.model import Equation
from ml_meta_perf.practices import best_practices, concordance, concordance_summary, feature_practices
from ml_meta_perf.report import (
    coverage,
    feature_usage,
    marginal_versus_conditional,
    operation_usage,
    term_groups,
    term_importance,
    term_sentences,
    unstable_majors,
)
from ml_meta_perf.selection import knee_index, knee_terms, pareto_front, pareto_table, recommend, simplify_curve
from ml_meta_perf.terms import Atom, Library, Term, build_library, ratio_of_sums_terms, simplify
from ml_meta_perf.validate import (
    CrossValidation,
    Scores,
    additive_oracle,
    baseline_group_centre,
    baseline_group_mean,
    cross_validate_fixed_form,
    decision_report,
    fold_selections,
    ranking_report,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_FEATURES",
    "CATALOGUE",
    "DATASET_COLUMN",
    "DATASET_FEATURES",
    "MODEL_CAPABILITY",
    "MODEL_COLUMN",
    "MODEL_FAMILY",
    "MODEL_FEATURES",
    "TARGET_COLUMN",
    "Atom",
    "CrossValidation",
    "Equation",
    "FitResult",
    "Library",
    "ModelEffects",
    "Practice",
    "Scores",
    "Term",
    "Verdict",
    "__version__",
    "additive_oracle",
    "aggregate_by_dataset",
    "assess",
    "baseline_group_centre",
    "baseline_group_mean",
    "best_practices",
    "build_library",
    "columns_as_arrays",
    "concordance",
    "concordance_summary",
    "correct_out_of_fold",
    "coverage",
    "cross_validate_fixed_form",
    "decision_report",
    "feature_practices",
    "feature_usage",
    "fit",
    "fit_effects",
    "fold_selections",
    "group_shares",
    "groups",
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
    "target",
    "term_effects",
    "term_groups",
    "term_importance",
    "term_sentences",
    "unstable_majors",
    "variance_decomposition",
]
