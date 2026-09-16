"""
Script to find the best ML model parameters for each full dataset using one train/test split.
"""
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PIPELINE_DIR = Path(__file__).resolve().parent
os.chdir(PIPELINE_DIR)

import time
import warnings
import traceback
from datetime import datetime
from joblib import Parallel, delayed as jl_delayed

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.model_selection import train_test_split

from exp_setup import (
    dataset_list,
    get_dataset_list_ordered_by_rows,
    seed_list,
    model_list,
    tuning_stage_max_workers,
    experiment_name,
    run_gpu_dl_models, use_gpu_for_dl_models
)
from utils.dataset_utils import preprocess_dataset, limit_dataset_size, read_csv_fast
from utils.model_utils import create_models, reset_seed


TUNING_TEST_SIZE = 0.2

RESULTS_DIR = "results/" + experiment_name
LOGS_DIR = "logs/" + experiment_name
LOG_FILE = f"{LOGS_DIR}/exp_stage_ml_tune_progress_log.txt"
TUNE_RESULTS_FILE = "results/results_stage_ml_tune.csv"

results_header = [
    "Seed",
    "Dataset",
    "Model",
    "Model Parameters"
]


def log_progress(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

    print(f"LOG: [{timestamp}] {message}")


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

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


def run_exp(setup: list):
    try:
        # Ignore warnings
        os.environ["PYTHONWARNINGS"] = "ignore"
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        results = []

        setup_name = setup[0]
        seed = setup[1]
        dataset_setup = setup[2]

        dataset_name = dataset_setup[0]
        useful_columns = dataset_setup[1]
        class_name = dataset_setup[2]

        # Loading the dataset
        dataset_folder = f"./datasets/{dataset_name}"
        csv_files = [
            file for file in os.listdir(dataset_folder)
            if file.endswith(".csv")
        ]

        if not csv_files:
            return f"ERROR: No CSV file found for {dataset_name}"

        log_progress(f"Loading dataset {dataset_name} from {dataset_folder}/{csv_files[0]}...")

        full_df = read_csv_fast(f"{dataset_folder}/{csv_files[0]}")

        log_progress(f"Limiting size and splitting {dataset_name}")

        if useful_columns:
            full_df.drop(columns=useful_columns, inplace=True)

        # Limiting dataset size
        full_df = limit_dataset_size(full_df, class_name, seed)

        # Separate X/y using the full dataset
        X = full_df.drop(class_name, axis=1)
        y = full_df[class_name]

        # One train/test split only
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TUNING_TEST_SIZE,
            random_state=seed,
            stratify=y
        )

        log_progress(f"Preprocessing dataset {dataset_name}...")

        # Fit preprocessing only on train, then transform test
        x_train, x_test = preprocess_dataset(x_train, x_test)

        reset_seed(seed)

        log_progress(f"Training models for dataset {dataset_name}...")

        # Hyperparameter tuning is done using the training partition only
        tuned_models = create_models(seed, x_train, y_train)

        for model_name, model_parameters_desc in tuned_models:
            results.append([
                seed,
                dataset_name,
                model_name,
                model_parameters_desc
            ])

        results_df = pd.DataFrame(results, columns=results_header)

        results_df.to_csv(
            f"{RESULTS_DIR}/results_stage_ml_tune_setup_{setup_name}.csv",
            index=False
        )

        return setup_name

    except Exception as e:
        error_message = (
            f"ERROR in setup {setup}: {e}\n"
            f"{traceback.format_exc()}"
        )

        error_file = f"{LOGS_DIR}/exp_stage_ml_tune_errors.txt"

        with open(error_file, "a", encoding="utf-8") as f:
            f.write(error_message)
            f.write("\n\n")

        return f"ERROR: {setup[0]}"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Clear previous progress log
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("ML tuning progress log\n")
        f.write("======================\n\n")

    # Clear previous error log
    with open(f"{LOGS_DIR}/exp_stage_ml_tune_errors.txt", "w", encoding="utf-8") as f:
        f.write("ML tuning error log\n")
        f.write("===================\n\n")

    # Resume: detect already-completed setups and skip them.
    # NOTE: setup IDs use the *original* dataset_list index so they remain stable
    # across reruns regardless of how get_dataset_list_ordered_by_rows() re-orders them.
    completed_setups_set = {
        file.replace("results_stage_ml_tune_setup_", "").replace(".csv", "")
        for file in os.listdir(RESULTS_DIR)
        if file.startswith("results_stage_ml_tune_setup_") and file.endswith(".csv")
    }
    if completed_setups_set:
        log_progress(f"Resuming: detected {len(completed_setups_set)} already-completed setup files.")

    # Build a stable dataset_no mapping from the canonical (unordered) dataset_list
    dataset_no_map = {dataset_setup[0]: idx + 1 for idx, dataset_setup in enumerate(dataset_list)}

    setups_to_parallelize = []

    for seed in seed_list:
        for dataset_setup in get_dataset_list_ordered_by_rows():
            dataset_no = dataset_no_map[dataset_setup[0]]

            setup_id = f"seed_{seed}_dataset_{dataset_no}"
            if setup_id in completed_setups_set:
                continue

            setups_to_parallelize.append([
                setup_id,
                seed,
                dataset_setup
            ])

    total_setups = len(setups_to_parallelize)
    if total_setups == 0:
        if os.path.exists(TUNE_RESULTS_FILE):
            log_progress("All setups already completed! Nothing to run.")
            return
        log_progress("All setup files already exist. Merging them into the tuning results file.")

    cuda_available, cuda_info = get_cuda_info()

    log_progress(f"Run GPU DL models: {'Yes' if run_gpu_dl_models else 'No'}")
    log_progress(f"Is CUDA available: {'Yes' if cuda_available else 'No'}")
    log_progress(f"CUDA info: {cuda_info}")
    log_progress(f"Running on {'GPU' if run_gpu_dl_models else 'CPU'}")
    log_progress(f"Workers: {tuning_stage_max_workers}")
    log_progress(f"Seeds: {len(seed_list)}")
    log_progress(f"Datasets: {len(dataset_list)}")
    log_progress(f"Models: {len(model_list)}")
    if run_gpu_dl_models and use_gpu_for_dl_models and not cuda_available:
        error_message = "GPU mode is enabled, but CUDA is not available"

        error_file = f"{LOGS_DIR}/exp_stage_ml_tune_errors.txt"
        with open(error_file, "a", encoding="utf-8") as f:
            f.write(error_message)
            f.write("\n\n")
        raise RuntimeError(
            error_message
        )

    effective_workers = min(tuning_stage_max_workers, total_setups)
    log_progress(f"Effective workers after resume filtering: {effective_workers}")

    start_time = time.time()
    completed_setups = 0

    if effective_workers <= 1:
        for setup in setups_to_parallelize:
            log_progress(f"Running setup {setup[0]} ({completed_setups + 1}/{total_setups})...")

            setup_result = run_exp(setup)
            completed_setups += 1

            elapsed_time = time.time() - start_time
            avg_time_per_setup = elapsed_time / completed_setups
            remaining_setups = total_setups - completed_setups
            estimated_remaining_time = avg_time_per_setup * remaining_setups

            progress_percent = (completed_setups / total_setups) * 100

            log_progress(
                f"Completed {completed_setups}/{total_setups} setups "
                f"({progress_percent:.2f}%). "
                f"Elapsed: {format_seconds(elapsed_time)}. "
                f"Estimated remaining: {format_seconds(estimated_remaining_time)}. "
                f"Last finished: {setup_result}"
            )

    else:
        results_iter = Parallel(
            n_jobs=effective_workers,
            backend="loky",
            verbose=0,
            return_as="generator_unordered",
        )(jl_delayed(run_exp)(setup) for setup in setups_to_parallelize)

        for setup_result in results_iter:
            completed_setups += 1

            elapsed_time = time.time() - start_time
            avg_time_per_setup = elapsed_time / completed_setups
            remaining_setups = total_setups - completed_setups
            estimated_remaining_time = avg_time_per_setup * remaining_setups

            progress_percent = (completed_setups / total_setups) * 100

            log_progress(
                f"Completed {completed_setups}/{total_setups} setups "
                f"({progress_percent:.2f}%). "
                f"Elapsed: {format_seconds(elapsed_time)}. "
                f"Estimated remaining: {format_seconds(estimated_remaining_time)}. "
                f"Last finished: {setup_result}"
            )

    log_progress("All parallel setups finished. Merging result files...")

    all_files = [
        file for file in os.listdir(RESULTS_DIR)
        if file.startswith("results_stage_ml_tune_setup_")
    ]

    all_files = sorted(all_files)

    if not all_files:
        log_progress("No tuning result files were generated.")
        raise RuntimeError("No tuning result files were generated.")

    all_results = pd.concat(
        [pd.read_csv(f"{RESULTS_DIR}/{file}") for file in all_files],
        ignore_index=True
    )

    all_results.to_csv(TUNE_RESULTS_FILE, index=False)

    log_progress(
        f"Merged {len(all_files)} files into "
        f"{TUNE_RESULTS_FILE}"
    )

    # Remove individual files
    for file in all_files:
        os.remove(f"{RESULTS_DIR}/{file}")

    total_elapsed_time = time.time() - start_time

    log_progress(
        f"Finished successfully. Total runtime: {format_seconds(total_elapsed_time)}"
    )


if __name__ == "__main__":
    main()
