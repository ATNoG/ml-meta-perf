"""What the test suite shares: a slice of the corpus.

A unit test asks whether *this project's functions* do what they say. That question does not need
484 rows: it needs a frame with the same schema and the same shape of structure -- more than one
dataset, more than one model, a missing cell -- and every extra row is paid for by every test that
fits anything. So the suite fits on a **derived slice of the shipped corpus**: eight datasets by ten
models, two cells missing on purpose (`KPI-KQI` was never run on the neural models), so the doubly
held out protocol cannot pass by assuming a rectangle. Derived rather than a checked-in second CSV,
because a copy would go stale the first time a column moved and nothing would say so.

This slice cannot answer whether the study's findings hold; tests of a finding fit what their claim
is about on the real corpus, or on the planted corpora of `tests/synthetic.py`.
"""

from __future__ import annotations

import atexit
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf.config import Configuration, SelectionConfig
from ml_meta_perf.data import (
    columns_as_arrays,
    load,
)
from tests.study import DATASET_COLUMN, DATASET_FEATURES, MODEL_COLUMN, MODEL_FEATURES

#: Eight of the twenty datasets, spread across the alphabetical order rather than taken from
#: one end, since the corpus is ordered by name and adjacent names are not independent.
#: `KPI-KQI` is here for its missing cells.
DATASETS = (
    "5G_Slicing",
    "BoTNeTIoT-L01",
    "DDOS-ANL",
    "IoT-DNL",
    "KPI-KQI",
    "NSL-KDD",
    "QoS-QoE",
    "X-IIoTID",
)

#: Ten of the twenty-five models, one from each family the taxonomy names -- boosting, naive
#: Bayes, trees, instance-based, linear, margin, neural, and the three linear-model variants
#: whose descriptors differ. `MLP` and `XGBoost` are two of the eight absent from `KPI-KQI`.
MODELS = (
    "AdaBoost",
    "BernoulliNB",
    "DT",
    "KNN",
    "LR",
    "LinearSVC",
    "MLP",
    "Ridge",
    "SGD",
    "XGBoost",
)


#: The configurations the unit suite fits under. Short and cheap, and **not** the tuned ones:
#: a test that reproduced the published search would be reproducing the study rather than
#: checking it. Six terms, because three is short enough that several tables come out
#: degenerate -- a practice carried by one term, a curve with two points -- and a degenerate
#: table passes checks a real one would fail.
#:
#: All three are written out. They used to come from `--quick`, which set E1 and E3 from a
#: preset, E2 from somewhere else, and the opaque ensembles from a third place; what a call
#: site was actually asking for could not be read off it.
E1 = Configuration(max_abs_zscore=3.0, penalty=1.0, pool_size=60, max_terms=6, beam_width=6, max_arity=3)
E3 = Configuration(max_abs_zscore=3.0, penalty=20.0, pool_size=60, max_terms=6, beam_width=6, max_arity=3)
#: The length rule at the suite's scale: a six-term horizon leaves room for a two-length window.
SELECTION = SelectionConfig(delta=0.01, window=2, smoothing=3, method="plateau_knee", capability_arity=3)

_FRAME: list[pl.DataFrame] = []
_PATH: list[Path] = []


def sample() -> pl.DataFrame:
    """The slice, loaded and validated by `data.load` exactly as the real corpus is."""
    if not _FRAME:
        _FRAME.append(load().filter(pl.col(DATASET_COLUMN).is_in(DATASETS) & pl.col(MODEL_COLUMN).is_in(MODELS)))
    return _FRAME[0]


def sample_path() -> Path:
    """The slice as a CSV on disk, for the code paths that take a path rather than a frame.

    Written once per process into a temporary directory that is removed at exit. `--data`
    is a real flag with a real reader behind it, and a test that never gives it a file is
    not testing it.
    """
    if not _PATH:
        directory = tempfile.mkdtemp(prefix="ml-meta-perf-tests-")
        path = Path(directory) / "sample_meta_dataset.csv"
        sample().write_csv(path)
        atexit.register(lambda: (path.unlink(missing_ok=True), Path(directory).rmdir()))
        _PATH.append(path)
    return _PATH[0]


def columns() -> dict[str, np.ndarray]:
    """The slice's feature arrays, in the order every caller in the package asks for them."""
    return columns_as_arrays(sample(), DATASET_FEATURES + MODEL_FEATURES)
