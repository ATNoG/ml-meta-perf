"""What the test suite shares: a slice of the corpus, and the two fits worth paying for once.

A unit test asks whether *this project's functions* do what they say. That question does not
need 476 rows: it needs a frame with the same schema and the same shape of structure -- more
than one dataset, more than one model, a missing cell -- and every extra row after that is
paid for by every test that fits anything.

So the suite fits on a **derived slice of the shipped corpus**: eight datasets by ten models,
78 of the 476 rows. Derived rather than a checked-in second CSV, because a copy of the corpus
would go stale the first time a column moved and nothing would say so; taking a slice of the
real one at import time cannot. It is a slice rather than a synthetic frame for the same
reason -- the features carry the real correlations, so a term library built over it is the
real library, and a test that passes here is not passing on numbers invented to make it pass.

**Two of the eighty cells are missing on purpose.** `KPI-KQI` is one of the three datasets the
eight neural models were never run on, and `MLP` and `XGBoost` are two of those eight, so the
slice is ragged exactly the way the corpus is. A grid with no holes would let the
doubly-held-out protocol pass while assuming a rectangle.

**What does not belong here.** This corpus cannot answer whether the study's findings hold --
eight datasets is not the corpus the paper reports, and a finding measured on a slice is not
the finding. A test asserting something about the study loads the real corpus, and it fits
only what its claim is about: `run_e3(load(), DEFAULT_E3)` is 4.6 s and a whole `run` is
twelve, so a claim about the published equation should never reach for the second.
"""

from __future__ import annotations

import atexit
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.dummy import DummyRegressor

from ml_meta_perf.data import (
    DATASET_COLUMN,
    DATASET_FEATURES,
    MODEL_COLUMN,
    MODEL_FEATURES,
    columns_as_arrays,
    load,
)
from ml_meta_perf.experiment import Configuration, EquationReport, Report
from ml_meta_perf.opaque import Builder

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


class Stub:
    """An estimator that fits and predicts and does nothing else.

    scikit-learn's regressors are not this project's to test; the folds around them are. A
    double exercises every one of those -- that the doubly-held-out split removes both groups,
    that predictions are clipped to the training range, that the table is scored from the
    predictions that are kept -- and costs nothing. The real ensembles are what makes
    `opaque.evaluate` the most expensive function in the package.
    """

    offset: float = 0.0

    def fit(self, design: np.ndarray, truth: np.ndarray, /) -> Stub:
        self.offset = float(truth.mean())
        return self

    def predict(self, design: np.ndarray, /) -> np.ndarray:
        # A hair of dependence on the design, so a test can tell this apart from a constant.
        return design[:, 0] * 1e-5 + self.offset


#: The doubles, in the shape `opaque.evaluate` takes: one that varies with the data and one
#: constant, so a table built from them has more than one row to be wrong about.
DOUBLES: tuple[tuple[str, Builder], ...] = (
    ("stub (varying)", lambda _threads=-1: Stub()),
    ("stub (constant)", lambda _threads=-1: DummyRegressor(strategy="mean")),
)

#: The configurations the unit suite fits under. Short and cheap, and **not** the tuned ones:
#: a test that reproduced the published search would be reproducing the study, which is what
#: `study/` is for. Six terms rather than the three `--quick` uses, because three is short
#: enough that several tables come out degenerate -- a practice carried by one term, a curve
#: with two points -- and a degenerate table passes checks a real one would fail.
E1 = Configuration(max_abs_zscore=3.0, penalty=1.0, pool_size=60, max_terms=6)
E3 = Configuration(max_abs_zscore=3.0, penalty=20.0, pool_size=60, max_terms=6)

_FRAME: list[pl.DataFrame] = []
_PATH: list[Path] = []
_REPORT: list[Report] = []
_PUBLISHED: list[EquationReport] = []


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


def published() -> EquationReport:
    """E3 as the study publishes it: the real corpus, the tuned configuration, fitted once.

    The exception to everything above, and the reason it is here rather than in whichever
    module wanted it first. Five modules assert something about the published equation -- that
    it repeats no feature combination, that it uses fewer features than the corpus carries,
    that no opaque model transfers as well, that the identity correction leaves its reported
    row alone -- and each of them was calling `run_e3(load())` for itself, at 4.6 s a call,
    seven times across a suite that otherwise ran in seconds.

    It carries the cross-validated scores and the fold paths, so a caller needing a protocol
    does not have to refit to get one.
    """
    if not _PUBLISHED:
        from ml_meta_perf.experiment import run_e3

        _PUBLISHED.append(run_e3(load()))
    return _PUBLISHED[0]


def report() -> Report:
    """One `experiment.run` over the slice, shared by every test that needs a whole report.

    Built once per process because it is the only expensive thing left in the unit suite, and
    shared rather than rebuilt so that the tests which read it are all reading the same
    object -- two fixtures that fit separately can disagree, and then a failure is about the
    fixtures rather than about the code.

    Under `quick` and over the doubles: the numbers in it are not the study's and no test here
    should read them as such. What it is for is shape -- that every table has the columns the
    figures and the chapters ask for, and that they line up with each other.
    """
    if not _REPORT:
        from ml_meta_perf.experiment import run

        _REPORT.append(run(str(sample_path()), quick=True, config_e1=E1, config_e3=E3, opaque_models=DOUBLES))
    return _REPORT[0]
