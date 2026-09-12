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
from ml_meta_perf.search import SearchResult, prune, search
from ml_meta_perf.selection import (
    arity_candidates,
    consensus_curve,
    floor_argmax,
    floor_curve,
    most_capable,
    plateau_configuration,
    plateau_index,
    protocol_spread,
)
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
    "Library",
    "ModelEffects",
    "Practice",
    "Scores",
    "SearchResult",
    "Term",
    "Verdict",
    "__version__",
    "additive_oracle",
    "aggregate_by_dataset",
    "arity_candidates",
    "assess",
    "baseline_group_centre",
    "baseline_group_mean",
    "best_practices",
    "build_library",
    "columns_as_arrays",
    "concordance",
    "concordance_summary",
    "consensus_curve",
    "correct_out_of_fold",
    "coverage",
    "cross_validate_fixed_form",
    "decision_report",
    "feature_practices",
    "feature_usage",
    "fit_effects",
    "floor_argmax",
    "floor_curve",
    "fold_selections",
    "group_shares",
    "groups",
    "load",
    "marginal_versus_conditional",
    "most_capable",
    "operation_usage",
    "plateau_configuration",
    "plateau_index",
    "protocol_spread",
    "prune",
    "ranking_report",
    "ratio_of_sums_terms",
    "redundancy_groups",
    "screen",
    "search",
    "simplify",
    "target",
    "term_effects",
    "term_groups",
    "term_importance",
    "term_sentences",
    "unstable_majors",
    "variance_decomposition",
]
