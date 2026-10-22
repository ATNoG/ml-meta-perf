"""ml-meta-perf -- interpretable additive equations for the performance of a model on a dataset.

Fits sparse additive equations of the form

$$\\hat{y}(\\mathbf{f}) = \\operatorname{clip}_B\\!\\left(w_0 + \\sum_{i=1}^{k} w_i \\, t_i(\\mathbf{f})\\right)$$

where each term $t_i$ is a symbolic expression over one to three dataset and model
meta-features (`ml_meta_perf.terms`), the weights $w_i$ are solved in closed form by ridge
regression (`ml_meta_perf.fit`), and predictions are evaluated under in-sample (IS),
leave-one-dataset-out (LODO), leave-one-model-out (LOMO) and doubly held-out (DHO)
cross-validation (`ml_meta_perf.validate`). Everything about a problem lives in a study JSON
file (such as ``config/mcc.json`` or ``config/flexfl/*.json``); the command line is
``ml-meta-perf fit | sweep | init``.
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
