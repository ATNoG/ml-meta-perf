"""
Auxiliary methods to create, tune, and train the models.

Model wrapper classes are kept in model_wrappers.py so this file focuses on
model registration, Optuna Bayesian search, cross-validation, and final fitting.
"""

import ast
import gc
import os
import random
import tempfile
import time

import numpy as np
import optuna
import torch
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier,
    Perceptron,
    PassiveAggressiveClassifier,
)
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier

from exp_setup import (
    model_list,
    gpu_dl_models,
    run_gpu_dl_models,
    use_gpu_for_dl_models,
    gpu_device_name,
    tuning_stage_max_workers,
    tabicl_offload_mode,
    tabicl_disk_offload_dir,
    tabicl_max_pinned_memory_mb,
    tabicl_min_batch_size,
    tabicl_safety_factor,
    tabicl_use_async,
    tabicl_async_depth,
)

_DL_TUNE_EPOCHS = 40
_DL_FINAL_EPOCHS = 200
_DL_MIN_TRAIN_SAMPLES = 1000

from utils.model_wrappers import (
    GPU_FIXED_BATCH_SIZE,
    GPU_TABNET_VIRTUAL_BATCH_SIZE,
    GPU_TABICL_BATCH_SIZE,
    KerasTorchSklearnClassifier,
    TabNetSklearnClassifier,
    LucidTabTransformerSklearnClassifier,
    RTDLFTTransformerSklearnClassifier,
    TabICLSklearnClassifier,
    TabPFNSklearnClassifier,
    release_cuda_memory_if_needed,
)

sklearn_mcc = make_scorer(matthews_corrcoef, greater_is_better=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FAILED_TRIAL_SCORE = -1.000001


class LabelEncodedXGBClassifier(XGBClassifier):
    """
    XGBoost classifier that accepts arbitrary class labels.

    XGBClassifier requires labels encoded as consecutive integers starting at
    zero. This wrapper performs that conversion only for XGBoost and converts
    predictions back to the original labels so the rest of the pipeline,
    including MCC calculation, continues to use the original target values.
    """

    def fit(self, x, y, **kwargs):
        self._target_label_encoder = LabelEncoder()
        y_encoded = self._target_label_encoder.fit_transform(np.asarray(y))
        super().fit(x, y_encoded, **kwargs)
        return self

    def predict(self, x, **kwargs):
        y_pred_encoded = super().predict(x, **kwargs)
        y_pred_encoded = np.asarray(y_pred_encoded, dtype=np.int64)
        return self._target_label_encoder.inverse_transform(y_pred_encoded)


def get_number_of_combinations(param_grid):
    """
    Get the number of combinations for a given param_grid.
    """
    num_combinations = 1

    for key, value in param_grid.items():
        num_combinations *= len(value)

    return num_combinations


def to_python_value(value):
    """
    Convert numpy scalar values returned by search tools into plain Python values.
    This keeps saved parameter strings compatible with ast.literal_eval().
    """
    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, tuple):
        return tuple(to_python_value(v) for v in value)

    if isinstance(value, list):
        return [to_python_value(v) for v in value]

    return value


def suggest_params_from_grid(trial, param_grid):
    """
    Suggest one hyperparameter configuration using Optuna.

    The project param grids are discrete lists. Some values are tuples or None,
    which are not ideal to store directly as Optuna categorical choices. Therefore,
    Optuna samples an integer index and this helper maps that index back to the
    real sklearn parameter value.
    """
    params = {}

    for key, values in param_grid.items():
        idx = trial.suggest_categorical(
            f"{key}__idx",
            list(range(len(values)))
        )
        params[key] = values[idx]

    return params


def stop_when_perfect_score(study, trial):
    """
    Stop Bayesian optimization early when MCC reaches its maximum value.
    """
    if study.best_value is not None and study.best_value >= 1.0:
        study.stop()


def _take_rows(data, indices):
    """Return rows by integer indices for numpy arrays, pandas objects, sparse matrices, or lists."""
    if hasattr(data, "iloc"):
        return data.iloc[indices]

    try:
        return data[indices]
    except TypeError:
        return [data[int(i)] for i in indices]


def cross_val_mcc_with_optuna_pruning(
        estimator,
        x_train,
        y_train,
        cv,
        trial,
        global_random_seed,
        model_name,
):
    """
    Programmatic CV loop that allows Optuna to prune bad trials early.

    cross_val_score is intentionally not used here because it is a black box:
    it only returns after all folds finish, so Optuna cannot inspect partial
    fold results and stop unpromising configurations.
    """
    cv_splitter = StratifiedKFold(
        n_splits=int(cv),
        shuffle=True,
        random_state=int(global_random_seed),
    )

    y_for_split = np.asarray(y_train)
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(x_train, y_for_split)):
        fold_estimator = clone(estimator)

        x_fold_train = _take_rows(x_train, train_idx)
        y_fold_train = _take_rows(y_train, train_idx)
        x_fold_val = _take_rows(x_train, val_idx)
        y_fold_val = _take_rows(y_train, val_idx)

        try:
            fold_estimator.fit(x_fold_train, y_fold_train)
            y_pred = fold_estimator.predict(x_fold_val)
            fold_mcc = float(matthews_corrcoef(y_fold_val, y_pred))
            fold_scores.append(fold_mcc)

            running_mean_score = float(np.mean(fold_scores))
            trial.report(running_mean_score, step=fold_idx)

            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned {model_name} after fold {fold_idx + 1}/{cv}; "
                    f"running MCC={running_mean_score:.6f}"
                )

        finally:
            del fold_estimator
            gc.collect()
            release_cuda_memory_if_needed(model_name)

    return float(np.mean(fold_scores))

def __init_models(global_random_seed, x_train, y_train):
    dl_device = gpu_device_name if run_gpu_dl_models and use_gpu_for_dl_models else "cpu"

    # XGBoost/LightGBM can run on the GPU, but only when a single worker owns it.
    # In the CPU stage many loky workers run in parallel; pointing them all at one
    # GPU causes contention/OOM. Only use the GPU when running serially.
    _tree_use_gpu = torch.cuda.is_available() and tuning_stage_max_workers == 1

    n_jobs = 1
    # n_jobs = -1

    models = {
        'LR': LogisticRegression(random_state=global_random_seed, n_jobs=n_jobs),
        'Ridge': RidgeClassifier(random_state=global_random_seed),
        'SGD': SGDClassifier(random_state=global_random_seed, n_jobs=n_jobs),
        'Perceptron': Perceptron(random_state=global_random_seed, n_jobs=n_jobs),
        'DT': DecisionTreeClassifier(random_state=global_random_seed),
        'ExtraTree': ExtraTreeClassifier(random_state=global_random_seed),
        'LinearSVC': LinearSVC(random_state=global_random_seed),
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        'XGBoost': LabelEncodedXGBClassifier(
            device="cuda" if _tree_use_gpu else "cpu",
            tree_method="hist",
            random_state=global_random_seed,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
        ),
        'LightGBM_RF': LGBMClassifier(
            device="gpu" if _tree_use_gpu else "cpu",
            boosting_type="rf",
            bagging_freq=1,
            bagging_fraction=0.8,
            random_state=global_random_seed,
            n_jobs=n_jobs,
            verbose=-1,
        ),
        'LightGBM_ExtraTrees': LGBMClassifier(
            device="gpu" if _tree_use_gpu else "cpu",
            boosting_type="rf",
            extra_trees=True,
            bagging_freq=1,
            bagging_fraction=0.8,
            random_state=global_random_seed,
            n_jobs=n_jobs,
            verbose=-1,
        ),
        'AdaBoost': AdaBoostClassifier(random_state=global_random_seed),
        'Bagging': BaggingClassifier(
            random_state=global_random_seed,
            estimator=DecisionTreeClassifier(max_depth=160),
            n_jobs=n_jobs
        ),
        'MLP': KerasTorchSklearnClassifier(
            seed=global_random_seed,
            device_name=dl_device,
            batch_size=GPU_FIXED_BATCH_SIZE,
            max_iter=_DL_TUNE_EPOCHS,
        ),
        'DNN': KerasTorchSklearnClassifier(
            seed=global_random_seed,
            device_name=dl_device,
            batch_size=GPU_FIXED_BATCH_SIZE,
            max_iter=_DL_TUNE_EPOCHS,
        ),

        'TabNet': TabNetSklearnClassifier(
            seed=global_random_seed,
            device_name=dl_device,
            batch_size=GPU_FIXED_BATCH_SIZE,
            virtual_batch_size=GPU_TABNET_VIRTUAL_BATCH_SIZE,
            max_epochs=_DL_TUNE_EPOCHS,
        ),

        'TabTransformer': LucidTabTransformerSklearnClassifier(
            seed=global_random_seed,
            device_name=dl_device,
            batch_size=GPU_FIXED_BATCH_SIZE,
            max_epochs=_DL_TUNE_EPOCHS,
        ),

        'FT-Transformer': RTDLFTTransformerSklearnClassifier(
            seed=global_random_seed,
            device_name=dl_device,
            batch_size=GPU_FIXED_BATCH_SIZE,
            max_epochs=_DL_TUNE_EPOCHS,
        ),

        'TabICL': TabICLSklearnClassifier(
            seed=global_random_seed,
            device_name=dl_device,
            batch_size=GPU_TABICL_BATCH_SIZE,
            n_jobs=1,
            offload_mode=tabicl_offload_mode,
            disk_offload_dir=tabicl_disk_offload_dir,
            max_pinned_memory_mb=tabicl_max_pinned_memory_mb,
            min_batch_size=tabicl_min_batch_size,
            safety_factor=tabicl_safety_factor,
            use_async=tabicl_use_async,
            async_depth=tabicl_async_depth,
        ),

        'TabPFN': TabPFNSklearnClassifier(
            seed=global_random_seed,
            device_name=dl_device,
            n_preprocessing_jobs=1
        ),

        'KNN': KNeighborsClassifier(n_jobs=n_jobs),
        'PassiveAggressive': PassiveAggressiveClassifier(
            random_state=global_random_seed,
            n_jobs=n_jobs
        ),
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis(),
    }


    return models


def create_models(global_random_seed, x_train, y_train) -> list:
    """
    Search for the best model parameters.
    Returns:
        list of tuples: (model_name, model_parameters_desc)
    """

    models = __init_models(global_random_seed, x_train, y_train)

    param_grids = {
        'LR': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2', None],
            'class_weight': [None, 'balanced']
        },
        'Ridge': {
            'alpha': [0.1, 1, 10],
            'class_weight': [None, 'balanced']
        },
        'SGD': {
            'loss': ['hinge', 'log_loss'],
            'alpha': [0.0001, 0.001, 0.01],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'class_weight': [None, 'balanced']
        },
        'Perceptron': {
            'penalty': [None, 'l1', 'l2'],
            'alpha': [0.0001, 0.001, 0.01],
            'class_weight': [None, 'balanced']
        },
        'DT': {
            'max_depth': [10, 20, 40, None],
            'max_features': [None, 'sqrt'],
            'class_weight': [None, 'balanced']
        },
        'ExtraTree': {
            'max_features': [None, 'sqrt'],
            'max_depth': [10, 20, 40, None],
            'class_weight': [None, 'balanced']
        },
        'LinearSVC': {
            'C': [0.1, 1, 10],
            'class_weight': [None, 'balanced']
        },
        'GaussianNB': {
            'var_smoothing': [1e-8, 1e-6, 1e-4, 1e-2]
        },
        'BernoulliNB': {
            'alpha': [0.1, 1, 10]
        },
        'XGBoost': {
            'learning_rate': [0.05, 0.1, 0.3],
            'max_depth': [6, 10, 20],
            'n_estimators': [100, 300],
            'reg_lambda': [0.0, 1.0]
        },
        'LightGBM_RF': {
            'num_leaves': [31, 63, 127],
            'max_depth': [10, 20, 40],
            'n_estimators': [100, 300],
            'class_weight': [None, 'balanced']
        },
        'LightGBM_ExtraTrees': {
            'num_leaves': [31, 63, 127],
            'max_depth': [10, 20, 40],
            'n_estimators': [100, 300],
            'class_weight': [None, 'balanced']
        },
        'AdaBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.5, 1]
        },
        'Bagging': {
            'n_estimators': [10, 50]
        },
        'MLP': {
            'hidden_layer_sizes': [(50,), (100,)],
            'activation': ['relu'],
            'alpha': [0.0001, 0.001]
        },
        'DNN': {
            'hidden_layer_sizes': [
                (16, 16, 16),
                (32, 16, 8)
            ],
            'activation': ['relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.001]
        },
        'TabNet': {
            'n_d': [8, 16],
            'n_a': [8, 16],
            'n_steps': [3, 4],
            'gamma': [1.3, 1.5],
            'lambda_sparse': [1e-4, 1e-3]
        },
        'TabTransformer': {
            'mlp_hidden_mults': [(2, 1), (4, 2), (8, 4)],
            'lr': [1e-3, 3e-4, 1e-4],
            'weight_decay': [1e-4, 1e-3]
        },
        'FT-Transformer': {
            'n_blocks': [2, 3],
            'd_block': [64, 128],
            'attention_n_heads': [4, 8],
            'attention_dropout': [0.0, 0.1],
            'ffn_dropout': [0.0, 0.1],
            'weight_decay': [1e-5, 1e-3],
            'lr': [1e-3, 3e-4]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'PassiveAggressive': {
            'C': [0.01, 0.1, 1],
            'loss': ['hinge']
        },
        'LDA': {
            'solver': ['lsqr'],
            'shrinkage': [None, 0.2, 0.5]
        },
        'QDA': {
            'solver': ['svd', 'eigen'],
            'shrinkage': [None, 0.1, 0.5, 0.9, 0.99, 1.0],
            'reg_param': [0.0, 0.1, 0.5, 0.99, 1.0],
        }
    }

    model_configurations = []

    for model_name, model in models.items():
        if model_name not in model_list:
            continue

        if model_name in gpu_dl_models and x_train.shape[0] < _DL_MIN_TRAIN_SAMPLES:
            print(f"Skipping {model_name}: dataset too small ({x_train.shape[0]} < {_DL_MIN_TRAIN_SAMPLES})")
            model_configurations.append((model_name, "skipped_dataset_too_small"))
            continue

        print(f"Searching for best parameters for {model_name}")

        param_grid = param_grids.get(model_name, {})
        if param_grid == {}:
            model_configurations.append((model_name, "unique"))
            continue

        cv = 5
        _trial_budgets = {
            'XGBoost': 32,
            'LightGBM_RF': 32,
            'LightGBM_ExtraTrees': 32,
            'SGD': 32,
            'DT': 16,
            'ExtraTree': 16,
            'KNN': 16,
            'TabTransformer': 8,
            'FT-Transformer': 16,
        }
        max_bayesian_trials = _trial_budgets.get(model_name, 32)

        number_of_combinations = get_number_of_combinations(param_grid)
        n_trials = min(number_of_combinations, max_bayesian_trials)
        n_startup_trials = min(10, max(3, n_trials // 4))

        tic = time.time()

        search_n_jobs = 1
        if tuning_stage_max_workers <= 1 and model_name not in gpu_dl_models:
            search_n_jobs = min(cv, os.cpu_count() or 1)
        if run_gpu_dl_models and use_gpu_for_dl_models and model_name in gpu_dl_models:
            search_n_jobs = 1

        sampler = optuna.samplers.TPESampler(
            seed=global_random_seed,
            n_startup_trials=n_startup_trials
        )

        if model_name in gpu_dl_models:
            pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=cv, reduction_factor=2)
        else:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=1)

        _shm_dir = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
        _study_db = os.path.join(_shm_dir, f"optuna_study_{model_name}_{os.getpid()}_{int(time.time()*1000)}.db")

        def _new_study(storage_url):
            return optuna.create_study(
                direction="maximize",
                sampler=sampler,
                pruner=pruner,
                study_name=f"{model_name}_bayesian_search",
                storage=storage_url,
                load_if_exists=True,
            )

        def objective(trial):
            params = suggest_params_from_grid(trial, param_grid)
            params = {
                key: to_python_value(value)
                for key, value in params.items()
            }

            estimator = clone(model)
            estimator.set_params(**params)

            # Read the selected values from the configured model instead of
            # saving the values directly from the parameter grid.
            model_params = estimator.get_params(deep=False)
            actual_params = {
                key: to_python_value(model_params[key])
                for key in param_grid
            }
            trial.set_user_attr("actual_params", actual_params)

            try:
                score = cross_val_mcc_with_optuna_pruning(
                    estimator=estimator,
                    x_train=x_train,
                    y_train=y_train,
                    cv=cv,
                    trial=trial,
                    global_random_seed=global_random_seed,
                    model_name=model_name,
                )

            except optuna.TrialPruned:
                raise

            except Exception as exc:
                # Invalid combinations should not stop the complete experiment.
                # MCC ranges from -1 to 1, so -1 is the worst valid score.
                trial.set_user_attr("failed_reason", repr(exc))
                print(f"Trial failed for {model_name} with params {params}: {repr(exc)}")
                score = FAILED_TRIAL_SCORE
                # raise exc

            finally:
                gc.collect()
                release_cuda_memory_if_needed(model_name)

            return score

        try:
            study = _new_study(f"sqlite:///{_study_db}")
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=search_n_jobs,
                show_progress_bar=False,
                callbacks=[stop_when_perfect_score]
            )
        except Exception as exc:
            # The on-disk SQLite study can fail mid-run if the node-local
            # storage backing /dev/shm becomes unwritable (e.g. a SLURM
            # tmpfs quota/cgroup limit). Optuna cannot recover from a failed
            # commit on its own storage, so retry once with an in-memory
            # study rather than losing every other model's tuning results.
            print(f"Optuna sqlite storage failed for {model_name} ({repr(exc)}); retrying with in-memory storage")
            try:
                study = _new_study(None)
                study.optimize(
                    objective,
                    n_trials=n_trials,
                    n_jobs=search_n_jobs,
                    show_progress_bar=False,
                    callbacks=[stop_when_perfect_score]
                )
            except Exception as exc2:
                print(f"Optuna tuning failed for {model_name} after retry: {repr(exc2)}")
                model_configurations.append((model_name, "skipped_tuning_error"))
                gc.collect()
                continue

        toc = time.time()

        best_params = study.best_trial.user_attrs.get("actual_params", {})
        best_params = {
            key: to_python_value(value)
            for key, value in best_params.items()
        }

        print(
            f"Optuna Bayesian elapsed time: {toc - tic} seconds. "
            f"Model: {model_name}. "
            f"Trials: {len(study.trials)}/{number_of_combinations}. "
            f"Search n_jobs: {search_n_jobs}. "
            f"Best Score: {study.best_value}"
        )

        model_configurations.append((model_name, str(best_params)[1:-1]))

        del study
        gc.collect()

        # Clean up the per-study SQLite file from /dev/shm to avoid RAM accumulation
        try:
            if os.path.exists(_study_db):
                os.remove(_study_db)
            # SQLite also creates a WAL and SHM sidecar file
            for _suffix in ("-wal", "-shm"):
                _sidecar = _study_db + _suffix
                if os.path.exists(_sidecar):
                    os.remove(_sidecar)
        except OSError:
            pass  # Non-fatal: file may already be gone

    return model_configurations


def train_best_model(model_name: str, global_random_seed: int, x_train, y_train, config: str):
    """
    Train the best model with the best parameters.
    Returns:
        trained model
    """

    model = __init_models(global_random_seed, x_train, y_train)[model_name]

    if config == 'skipped_dataset_too_small':
        return None

    if config != 'unique':
        config_dict = ast.literal_eval('{' + config + '}')
        model.set_params(**config_dict)

    # Restore full epoch budget for DL models trained from scratch.
    # - KerasTorchSklearnClassifier (MLP, DNN) uses max_iter, not max_epochs.
    # - TabNetSklearnClassifier, LucidTabTransformerSklearnClassifier,
    #   RTDLFTTransformerSklearnClassifier all use max_epochs.
    # - TabICL and TabPFN are inference-only and have no epoch parameter.
    if model_name in ('MLP', 'DNN'):
        model.set_params(max_iter=_DL_FINAL_EPOCHS)
    elif model_name in ('TabNet', 'TabTransformer', 'FT-Transformer'):
        model.set_params(max_epochs=_DL_FINAL_EPOCHS)

    try:
        model.fit(x_train, y_train)
    finally:
        release_cuda_memory_if_needed(model_name)

    return model

def reset_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
