"""ml-meta-perf -- interpretable additive equations for the performance of a model on a dataset.

Fits equations of the form ``y = w0 + w1*t1 + ... + wk*tk`` where each term is a simple
expression over dataset and model meta-features, and reports what they are actually worth
under in-sample (IS), leave-one-dataset-out (LODO), leave-one-model-out (LOMO) and doubly
held-out (DHO) evaluation. A study is described by a file (``config/study.json``); the
command line is ``ml-meta-perf fit | sweep | init``.
"""

from ml_meta_perf.attribution import group_shares, term_effects
from ml_meta_perf.config import (
    Configuration,
    Schema,
    SelectionConfig,
    StudyConfig,
    default_schema,
    load_config,
)
from ml_meta_perf.data import columns_as_arrays, groups, load, target
from ml_meta_perf.model import Equation
from ml_meta_perf.search import SearchResult, prune, pruning_threshold, search
from ml_meta_perf.selection import (
    floor_argmax,
    floor_curve,
    knee_lengths,
    plateau_knee,
    protocol_spread,
    smoothed,
)
from ml_meta_perf.study import StudyReport, run_study
from ml_meta_perf.terms import Atom, Library, Term, build_library, ratio_of_sums_terms, simplify
from ml_meta_perf.validate import (
    CrossValidation,
    Scores,
    cross_validate_fixed_form,
    fold_selections,
)

__version__ = "0.1.0"

__all__ = [
    "Atom",
    "Configuration",
    "CrossValidation",
    "Equation",
    "Library",
    "Schema",
    "Scores",
    "SearchResult",
    "SelectionConfig",
    "StudyConfig",
    "StudyReport",
    "Term",
    "__version__",
    "build_library",
    "columns_as_arrays",
    "cross_validate_fixed_form",
    "default_schema",
    "floor_argmax",
    "floor_curve",
    "fold_selections",
    "group_shares",
    "groups",
    "knee_lengths",
    "load",
    "load_config",
    "plateau_knee",
    "protocol_spread",
    "prune",
    "pruning_threshold",
    "ratio_of_sums_terms",
    "run_study",
    "search",
    "simplify",
    "smoothed",
    "target",
    "term_effects",
]
