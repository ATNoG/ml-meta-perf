"""
Script for training and evaluating tuned ML models using K-Fold CV per configuration.

This version mirrors exp_stage_ml_tune.py:
1. Reads experiment configuration from exp_setup.
2. Uses results/results_stage_ml_tune.csv as input.
3. Filters evaluations by seed_list, dataset_list and model_list from exp_setup.
4. Parallelizes evaluation by (seed, dataset) setup.
5. Fits preprocessing inside each fold only.
6. Computes final metrics from all out-of-fold predictions.
7. Writes only results_stage_ml_eval.csv.
"""

import gc
import os
from pathlib import Path

# Keep each process from oversubscribing CPU threads.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"

PIPELINE_DIR = Path(__file__).resolve().parent
os.chdir(PIPELINE_DIR)

import re
import time
import warnings
import traceback
from datetime import datetime
from joblib import Parallel, delayed as jl_delayed

import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedKFold

from exp_setup import (
    dataset_list,
    get_dataset_list_ordered_by_rows,
    seed_list,
    model_list,
    tuning_stage_max_workers,
    experiment_name,
    run_gpu_dl_models,
    use_gpu_for_dl_models,
)
from utils.dataset_utils import (
    limit_dataset_size,
    preprocess_dataset,
    read_csv_fast,
)
from utils.model_utils import reset_seed, train_best_model
from utils.model_wrappers import release_cuda_memory_if_needed


# Ignore warnings in the main process as well.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

N_SPLITS = 5

ASNM_CDX_2009_FT_MICRO_BATCH_SIZE = 64
BOTNETIOT_L01_MAX_EVAL_SIZE = 1_000_000

RESULTS_DIR = "results/" + experiment_name
LOGS_DIR = "logs/" + experiment_name
LOG_FILE = f"{LOGS_DIR}/exp_stage_ml_eval_progress_log.txt"
ERROR_FILE = f"{LOGS_DIR}/exp_stage_ml_eval_errors.txt"

TUNE_RESULTS_FILE = "results/results_stage_ml_tune.csv"
EVAL_RESULTS_FILE = "results/results_stage_ml_eval.csv"

PARTIAL_PREFIX = "results_stage_ml_eval_setup_"
PARTIAL_SUFFIX = ".csv"


def log_progress(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

    print(f"LOG: [{timestamp}] {message}")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)

    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m {secs}s"

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def sanitize_for_filename(value) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def get_cuda_info():
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "No CUDA device available"

        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()

        return True, f"{device_name} ({device_count} CUDA device(s))"

    except Exception as exc:
        return False, f"Could not check CUDA: {repr(exc)}"


def compute_metrics(y_true, y_pred, prefix: str = "") -> dict:
    """
    Compute global classification metrics.

    Macro metrics treat all classes equally.
    Micro metrics aggregate TP/FP/FN globally.
    Weighted metrics weight each class by its support.
    MCC is computed globally over the full multiclass confusion matrix.
    """
    return {
        f"{prefix}Accuracy": accuracy_score(y_true, y_pred),

        f"{prefix}Precision_Macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{prefix}Recall_Macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        f"{prefix}F1_Macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        f"{prefix}Precision_Micro": precision_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        f"{prefix}Recall_Micro": recall_score(
            y_true, y_pred, average="micro", zero_division=0
        ),
        f"{prefix}F1_Micro": f1_score(
            y_true, y_pred, average="micro", zero_division=0
        ),

        f"{prefix}Precision_Weighted": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        f"{prefix}Recall_Weighted": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        f"{prefix}F1_Weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),

        f"{prefix}MCC": matthews_corrcoef(y_true, y_pred),
    }


def find_dataset_setup(dataset_name: str):
    for dataset_setup in dataset_list:
        if dataset_setup[0] == dataset_name:
            return dataset_setup

    raise ValueError(f"Dataset {dataset_name} not found in dataset_list.")


def normalize_model_parameters(model_parameters) -> str:
    if pd.isna(model_parameters):
        return "unique"
    return str(model_parameters)


def partial_result_is_complete(setup_id: str, required_group_df: pd.DataFrame) -> bool:
    """
    Check whether an existing partial setup file contains all rows required by
    the current exp_setup configuration. This avoids accidentally reusing stale
    partial files after changing model_list or seed_list without changing
    experiment_name.
    """
    partial_path = f"{RESULTS_DIR}/{PARTIAL_PREFIX}{setup_id}{PARTIAL_SUFFIX}"

    if not os.path.exists(partial_path):
        return False

    try:
        partial_df = pd.read_csv(partial_path)
    except Exception:
        return False

    required_keys = {
        (int(row["Seed"]), str(row["Dataset"]), str(row["Model"]))
        for _, row in required_group_df.iterrows()
    }

    completed_keys = {
        (int(row["Seed"]), str(row["Dataset"]), str(row["Model"]))
        for _, row in partial_df.iterrows()
    }

    return required_keys.issubset(completed_keys)


def run_eval_setup(setup: list):
    """
    Evaluate all tuned model rows for one (seed, dataset) setup.

    setup = [setup_id, seed, dataset_setup, rows_to_evaluate]
    """
    try:
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        setup_id = setup[0]
        seed = int(setup[1])
        dataset_setup = setup[2]
        rows_to_evaluate = setup[3]

        dataset_name = dataset_setup[0]
        useful_columns = dataset_setup[1]
        class_name = dataset_setup[2]

        setup_results = []

        dataset_folder = f"./datasets/{dataset_name}"
        csv_files = [
            file for file in os.listdir(dataset_folder)
            if file.endswith(".csv")
        ]

        if not csv_files:
            return f"ERROR: No CSV file found for {dataset_name}"

        log_progress(
            f"Loading dataset {dataset_name} from "
            f"{dataset_folder}/{csv_files[0]}..."
        )

        # Load directly into X and remove the target column in place. Using
        # DataFrame.pop() avoids keeping both full_df and a duplicated feature
        # DataFrame in memory. Values, column order, and row order are unchanged.
        X = read_csv_fast(f"{dataset_folder}/{csv_files[0]}")

        if useful_columns:
            X.drop(columns=useful_columns, inplace=True)

        if dataset_name == "BoTNeTIoT-L01":
            original_size = len(X)
            X = limit_dataset_size(
                X,
                class_name,
                seed,
                max_size=BOTNETIOT_L01_MAX_EVAL_SIZE,
            )
            log_progress(
                f"Evaluation dataset {dataset_name} limited from "
                f"{original_size} to {len(X)} rows."
            )

        # Separate X/y using the evaluation dataset (limited above when required).
        # Preprocessing is intentionally NOT applied here.
        y = X.pop(class_name)

        reset_seed(seed)

        skf = StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=seed,
        )

        for row_number, row in enumerate(rows_to_evaluate, start=1):
            model_name = row["Model"]
            model_parameters = normalize_model_parameters(row["Model Parameters"])

            if model_parameters == "skipped_dataset_too_small":
                log_progress(
                    f"Skipping {dataset_name}, seed={seed}, model={model_name}: "
                    f"Model Parameters == 'skipped_dataset_too_small'."
                )
                continue

            if model_name == 'FT-Transformer' and dataset_name == 'ASNM-CDX-2009':
                if model_parameters == 'unique':
                    model_parameters = (
                        f"'micro_batch_size': {ASNM_CDX_2009_FT_MICRO_BATCH_SIZE}, "
                        f"'stream_batches_from_cpu': True"
                    )
                else:
                    model_parameters = (
                        f"{model_parameters}, "
                        f"'micro_batch_size': {ASNM_CDX_2009_FT_MICRO_BATCH_SIZE}, "
                        f"'stream_batches_from_cpu': True"
                    )
                log_progress(
                    f"Using micro_batch_size={ASNM_CDX_2009_FT_MICRO_BATCH_SIZE} "
                    f"with gradient accumulation and CPU batch streaming for "
                    f"{model_name} on {dataset_name}, seed={seed} (eval stage)"
                )

            log_progress(
                f"Evaluating {dataset_name}, seed={seed}, "
                f"model={model_name} ({row_number}/{len(rows_to_evaluate)} "
                f"models in setup {setup_id})..."
            )

            try:
                all_y_true = []
                all_y_pred = []
                fold_mcc_scores = []
                fold_f1_macro_scores = []
                fold_f1_weighted_scores = []

                for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
                    # Fancy iloc indexing already materializes the selected
                    # rows. Avoiding an additional explicit copy reduces the
                    # fold-level memory peak without changing any values.
                    x_train = X.iloc[train_idx]
                    x_test = X.iloc[test_idx]
                    y_train = y.iloc[train_idx]
                    y_test = y.iloc[test_idx]

                    # Correct: fit preprocessing only on this fold's training set.
                    x_train, x_test = preprocess_dataset(x_train, x_test)

                    # Detect genuine invalid values before fitting.
                    if not np.isfinite(x_train).all():
                        invalid_count = int((~np.isfinite(x_train)).sum())
                        raise ValueError(
                            f"{dataset_name}, seed={seed}, fold={fold}, model={model_name}: "
                            f"x_train contains {invalid_count} non-finite values after preprocessing."
                        )

                    if not np.isfinite(x_test).all():
                        invalid_count = int((~np.isfinite(x_test)).sum())
                        raise ValueError(
                            f"{dataset_name}, seed={seed}, fold={fold}, model={model_name}: "
                            f"x_test contains {invalid_count} non-finite values after preprocessing."
                        )

                    # Ridge can overflow internally when its float32 Gram matrix is computed.
                    # Casting float32 -> float64 does not alter the represented input values;
                    # it only performs the linear algebra with greater numerical range.
                    if model_name == "Ridge":
                        x_train_for_model = x_train.astype(np.float64, copy=False)
                        x_test_for_model = x_test.astype(np.float64, copy=False)
                    else:
                        x_train_for_model = x_train
                        x_test_for_model = x_test


                    reset_seed(seed)

                    print(
                        f"\033[92mStage 2: Training the model: "
                        f"{dataset_name}, seed={seed}, fold={fold}/{N_SPLITS}, len={len(x_train_for_model)}, "
                        f"model={model_name}\033[0m"
                    )

                    model = train_best_model(
                        model_name,
                        seed,
                        x_train_for_model,
                        y_train,
                        model_parameters,
                    )

                    if model_name == "TabICL":
                        # TabICL has already retained the fitted context it needs.
                        # Release the evaluation stage's external references before
                        # prediction to reduce its CPU-memory peak. Assigning None
                        # keeps the common cleanup block below valid.
                        x_train = None
                        x_train_for_model = None
                        y_train = None
                        gc.collect()
                        torch.cuda.empty_cache()

                    y_pred = model.predict(x_test_for_model)
                    fold_metrics = compute_metrics(y_test, y_pred)

                    fold_mcc_scores.append(fold_metrics["MCC"])
                    fold_f1_macro_scores.append(fold_metrics["F1_Macro"])
                    fold_f1_weighted_scores.append(fold_metrics["F1_Weighted"])

                    all_y_true.extend(np.asarray(y_test))
                    all_y_pred.extend(np.asarray(y_pred))

                    del (
                        model,
                        y_pred,
                        x_train,
                        x_test,
                        x_train_for_model,
                        x_test_for_model,
                        y_train,
                        y_test,
                    )
                    gc.collect()

                all_y_true = np.asarray(all_y_true)
                all_y_pred = np.asarray(all_y_pred)

                final_metrics = compute_metrics(all_y_true, all_y_pred)

                result_row = {
                    key: value
                    for key, value in row.items()
                    if not str(key).startswith("_")
                }

                # Backward-compatible columns.
                result_row["Accuracy"] = final_metrics["Accuracy"]
                result_row["Precision"] = final_metrics["Precision_Weighted"]
                result_row["Recall"] = final_metrics["Recall_Weighted"]
                result_row["F1"] = final_metrics["F1_Weighted"]
                result_row["MCC"] = final_metrics["MCC"]

                # Explicit macro/micro/weighted metrics.
                result_row.update(final_metrics)

                # Fold variation columns.
                result_row["MCC_Fold_Mean"] = float(np.mean(fold_mcc_scores))
                result_row["MCC_Fold_Std"] = float(np.std(fold_mcc_scores, ddof=1))
                result_row["F1_Macro_Fold_Mean"] = float(np.mean(fold_f1_macro_scores))
                result_row["F1_Macro_Fold_Std"] = float(np.std(fold_f1_macro_scores, ddof=1))
                result_row["F1_Weighted_Fold_Mean"] = float(np.mean(fold_f1_weighted_scores))
                result_row["F1_Weighted_Fold_Std"] = float(np.std(fold_f1_weighted_scores, ddof=1))

                setup_results.append(result_row)

                log_progress(
                    f"Finished model {model_name} for {dataset_name}, seed={seed}. "
                    f"MCC={final_metrics['MCC']:.4f}, "
                    f"F1_macro={final_metrics['F1_Macro']:.4f}, "
                    f"F1_weighted={final_metrics['F1_Weighted']:.4f}."
                )

                del all_y_true, all_y_pred

            except Exception as exc:
                # Isolate this model's failure so it doesn't discard results
                # already computed for the other models in this setup.
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                model_error_message = (
                    f"[{timestamp}] ERROR evaluating model={model_name}, dataset={dataset_name}, "
                    f"seed={seed} (setup {setup_id}): {exc}\n{traceback.format_exc()}"
                )

                with open(ERROR_FILE, "a", encoding="utf-8") as f:
                    f.write(model_error_message)
                    f.write("\n\n")

                log_progress(
                    f"FAILED model {model_name} for {dataset_name}, seed={seed}: {exc}"
                )

                continue

            finally:
                gc.collect()
                release_cuda_memory_if_needed(model_name)

        partial_result_path = f"{RESULTS_DIR}/{PARTIAL_PREFIX}{setup_id}{PARTIAL_SUFFIX}"

        if not setup_results:
            if os.path.exists(partial_result_path):
                os.remove(partial_result_path)

            log_progress(
                f"No successful model evaluations for setup {setup_id}; "
                f"no partial result file was written."
            )

            return f"ERROR: {setup_id}"

        setup_results_df = pd.DataFrame(setup_results)
        setup_results_df.to_csv(partial_result_path, index=False)

        del X, y, setup_results_df
        gc.collect()

        return setup_id

    except Exception as exc:
        # Setup-level failure (e.g. dataset loading), not tied to one model.
        # Per-model failures are caught and logged individually above and
        # never reach this handler.
        try:
            setup_id_for_log = setup[0]
            seed_for_log = setup[1]
            dataset_name_for_log = setup[2][0]
            n_rows_for_log = len(setup[3])
        except Exception:
            setup_id_for_log = "<unknown>"
            seed_for_log = "<unknown>"
            dataset_name_for_log = "<unknown>"
            n_rows_for_log = "<unknown>"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_message = (
            f"[{timestamp}] ERROR in evaluation setup {setup_id_for_log} "
            f"(seed={seed_for_log}, dataset={dataset_name_for_log}, "
            f"{n_rows_for_log} model rows): {exc}\n"
            f"{traceback.format_exc()}"
        )

        with open(ERROR_FILE, "a", encoding="utf-8") as f:
            f.write(error_message)
            f.write("\n\n")

        return f"ERROR: {setup_id_for_log}"


def build_eval_setups(results_df: pd.DataFrame):
    allowed_seeds = {int(seed) for seed in seed_list}
    allowed_datasets = {dataset_setup[0] for dataset_setup in dataset_list}
    allowed_models = set(model_list)

    filtered_df = results_df.copy()
    filtered_df["Seed"] = filtered_df["Seed"].astype(int)
    filtered_df["_original_row_index"] = filtered_df.index

    filtered_df = filtered_df[
        filtered_df["Seed"].isin(allowed_seeds)
        & filtered_df["Dataset"].isin(allowed_datasets)
        & filtered_df["Model"].isin(allowed_models)
    ].copy()

    skipped_mask = (
        filtered_df["Model Parameters"].astype(str)
        == "skipped_dataset_too_small"
    )
    n_skipped = skipped_mask.sum()
    if n_skipped > 0:
        log_progress(
            f"Skipping {n_skipped} evaluation(s) because "
            f"Model Parameters == 'skipped_dataset_too_small'."
        )
    filtered_df = filtered_df[~skipped_mask].copy()

    if filtered_df.empty:
        return [], 0, []

    dataset_no_map = {
        dataset_setup[0]: idx + 1
        for idx, dataset_setup in enumerate(dataset_list)
    }

    try:
        ordered_dataset_list = get_dataset_list_ordered_by_rows()
    except Exception as exc:
        log_progress(
            f"Could not order datasets by rows using exp_setup helper: {repr(exc)}. "
            f"Falling back to dataset_list order."
        )
        ordered_dataset_list = dataset_list

    setups_to_parallelize = []
    all_relevant_setup_ids = []
    completed_count = 0
    stale_count = 0
    total_evaluations = 0

    for seed in seed_list:
        seed = int(seed)

        for dataset_setup in ordered_dataset_list:
            dataset_name = dataset_setup[0]

            group_df = filtered_df[
                (filtered_df["Seed"] == seed)
                & (filtered_df["Dataset"] == dataset_name)
            ].copy()

            if group_df.empty:
                continue

            # Keep row/model order stable and aligned with model_list when possible.
            model_order_map = {model_name: idx for idx, model_name in enumerate(model_list)}
            group_df["_model_order"] = group_df["Model"].map(model_order_map).fillna(999999)
            group_df = group_df.sort_values(["_model_order", "_original_row_index"])

            dataset_no = dataset_no_map[dataset_name]
            setup_id = f"seed_{seed}_dataset_{dataset_no}"
            all_relevant_setup_ids.append(setup_id)
            total_evaluations += len(group_df)

            partial_path = f"{RESULTS_DIR}/{PARTIAL_PREFIX}{setup_id}{PARTIAL_SUFFIX}"
            if os.path.exists(partial_path):
                if partial_result_is_complete(setup_id, group_df):
                    completed_count += 1
                    continue

                # Existing partial file does not match current exp_setup selection.
                # Remove it and recompute the setup.
                stale_count += 1
                os.remove(partial_path)

            rows_to_evaluate = group_df.drop(columns=["_model_order"]).to_dict("records")

            setups_to_parallelize.append([
                setup_id,
                seed,
                dataset_setup,
                rows_to_evaluate,
            ])

    if completed_count > 0:
        log_progress(
            f"Resuming: detected {completed_count} completed evaluation setup files "
            f"matching current exp_setup."
        )

    if stale_count > 0:
        log_progress(
            f"Removed {stale_count} stale partial evaluation setup files because "
            f"they did not match the current exp_setup model/seed/dataset selection."
        )

    return setups_to_parallelize, total_evaluations, all_relevant_setup_ids


def merge_partial_results(relevant_setup_ids):
    partial_files = []

    for setup_id in relevant_setup_ids:
        file_name = f"{PARTIAL_PREFIX}{setup_id}{PARTIAL_SUFFIX}"
        file_path = f"{RESULTS_DIR}/{file_name}"

        if os.path.exists(file_path):
            partial_files.append(file_name)

    partial_files = sorted(partial_files)

    if not partial_files:
        if os.path.exists(EVAL_RESULTS_FILE):
            log_progress(
                f"No partial evaluation files found. Existing final file remains: "
                f"{EVAL_RESULTS_FILE}"
            )
            return

        raise RuntimeError("No evaluation result files were generated.")

    all_results = pd.concat(
        [pd.read_csv(f"{RESULTS_DIR}/{file}") for file in partial_files],
        ignore_index=True,
    )

    all_results.to_csv(EVAL_RESULTS_FILE, index=False)

    log_progress(
        f"Merged {len(partial_files)} files into {EVAL_RESULTS_FILE}"
    )

    # Remove individual setup files to keep the results directory clean.
    for file in partial_files:
        os.remove(f"{RESULTS_DIR}/{file}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("ML evaluation progress log\n")
        f.write("==========================\n\n")

    with open(ERROR_FILE, "w", encoding="utf-8") as f:
        f.write("ML evaluation error log\n")
        f.write("=======================\n\n")

    if not os.path.exists(TUNE_RESULTS_FILE):
        raise FileNotFoundError(
            f"Tuning results file not found: {TUNE_RESULTS_FILE}. "
            f"Run exp_stage_ml_tune.py first or check experiment_name in exp_setup."
        )

    tuning_results_df = pd.read_csv(TUNE_RESULTS_FILE)

    setups_to_parallelize, total_evaluations, relevant_setup_ids = build_eval_setups(tuning_results_df)

    # Relevant setup ids include both pending and already-completed partial files
    # that match the current exp_setup configuration.
    relevant_setup_ids = sorted(set(relevant_setup_ids))

    total_setups = len(setups_to_parallelize)

    cuda_available, cuda_info = get_cuda_info()

    log_progress(f"Experiment name: {experiment_name}")
    log_progress(f"Input tuning file: {TUNE_RESULTS_FILE}")
    log_progress(f"Output evaluation file: {EVAL_RESULTS_FILE}")
    log_progress(f"Run GPU DL models: {'Yes' if run_gpu_dl_models else 'No'}")
    log_progress(f"Is CUDA available: {'Yes' if cuda_available else 'No'}")
    log_progress(f"CUDA info: {cuda_info}")
    log_progress(f"Running on {'GPU' if run_gpu_dl_models else 'CPU'}")
    log_progress(f"Configured workers: {tuning_stage_max_workers}")
    log_progress(f"Seeds configured: {len(seed_list)}")
    log_progress(f"Datasets configured: {len(dataset_list)}")
    log_progress(f"Models configured: {len(model_list)}")
    log_progress(f"Evaluations selected from tuning file: {total_evaluations}")
    log_progress(f"Pending evaluation setups: {total_setups}")

    if run_gpu_dl_models and use_gpu_for_dl_models and not cuda_available:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_message = f"[{timestamp}] GPU mode is enabled, but CUDA is not available"

        with open(ERROR_FILE, "a", encoding="utf-8") as f:
            f.write(error_message)
            f.write("\n\n")

        raise RuntimeError(error_message)

    if total_setups == 0:
        log_progress("All selected evaluation setups already completed or no pending setups found.")
        merge_partial_results(relevant_setup_ids)
        return

    effective_workers = min(tuning_stage_max_workers, total_setups)
    log_progress(f"Effective workers after resume filtering: {effective_workers}")

    start_time = time.time()
    completed_setups = 0

    if effective_workers <= 1:
        for setup in setups_to_parallelize:
            log_progress(f"Running evaluation setup {setup[0]} ({completed_setups + 1}/{total_setups})...")

            setup_result = run_eval_setup(setup)
            completed_setups += 1

            elapsed_time = time.time() - start_time
            avg_time_per_setup = elapsed_time / completed_setups
            remaining_setups = total_setups - completed_setups
            estimated_remaining_time = avg_time_per_setup * remaining_setups
            progress_percent = (completed_setups / total_setups) * 100

            log_progress(
                f"Completed {completed_setups}/{total_setups} evaluation setups "
                f"({progress_percent:.2f}%). "
                f"Elapsed: {format_seconds(elapsed_time)}. "
                f"Estimated remaining time: {format_seconds(estimated_remaining_time)}. "
                f"Last finished: {setup_result}"
            )

    else:
        results_iter = Parallel(
            n_jobs=effective_workers,
            backend="loky",
            verbose=0,
            return_as="generator_unordered",
        )(jl_delayed(run_eval_setup)(setup) for setup in setups_to_parallelize)

        for setup_result in results_iter:
            completed_setups += 1

            elapsed_time = time.time() - start_time
            avg_time_per_setup = elapsed_time / completed_setups
            remaining_setups = total_setups - completed_setups
            estimated_remaining_time = avg_time_per_setup * remaining_setups
            progress_percent = (completed_setups / total_setups) * 100

            log_progress(
                f"Completed {completed_setups}/{total_setups} evaluation setups "
                f"({progress_percent:.2f}%). "
                f"Elapsed: {format_seconds(elapsed_time)}. "
                f"Estimated remaining time: {format_seconds(estimated_remaining_time)}. "
                f"Last finished: {setup_result}"
            )

    log_progress("All evaluation setups finished. Merging result files...")

    merge_partial_results(relevant_setup_ids)

    total_elapsed_time = time.time() - start_time

    print("\033[92mEvaluation finished successfully.\033[0m")
    log_progress(
        f"Evaluation finished successfully. "
        f"Total runtime: {format_seconds(total_elapsed_time)}."
    )


if __name__ == "__main__":
    main()
