"""
This file defines the datasets, models, seeds and dataset sizes to be used in the experiments 5 7 11 13 42
"""
import json
import os
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = PIPELINE_DIR / "datasets"
LOCAL_CONFIG_PATH = PIPELINE_DIR / "local_config.json"


def load_local_config() -> dict[str, str]:
    if not LOCAL_CONFIG_PATH.exists():
        return {}

    with LOCAL_CONFIG_PATH.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Local config must be a JSON object: {LOCAL_CONFIG_PATH}")

    return {str(key): str(value) for key, value in config.items()}


local_config = load_local_config()

seed_list = [5, 7, 11, 13, 42]

run_gpu_dl_models = False
# run_gpu_dl_models = True

use_gpu_for_dl_models = True

# the experiment name will be the concatenation of run_gpu_dl_models and the seeds
experiment_name = f"testing_{run_gpu_dl_models}_seeds_{seed_list}"

# format: [name, [features to be removed], output]] # dimension
dataset_list = [
    ['BoTNeTIoT-L01', ['Device_Name', 'Attack', 'Attack_subType'], 'label'],  # 7,062,606 x 27
    ['DDOS-ANL', [], 'PKT_CLASS'],  # 2,160,668 x 28
    ['X-IIoTID', ['Date', 'Timestamp', 'class1', 'class2'], 'class3'],  # 820,834 x 68
    ['IoTID20', ['Flow_ID', 'Cat', 'Sub_Cat'], 'Label'],  # 625,783 x 86
    ['5G_Slicing', [], 'Slice Type (Output)'],  # 466,739 x 9
    ['IoT-DNL', [], 'normality'],  # 477,426 x 14
    ['MQTTEEB-D', ['timestamp'], 'label'],  # 222,813 x 13
    ['NSL-KDD', [], 'class'],  # 148,517 x 42
    ['UAVIDS-2025', ['FlowID'], 'label'],  # 122,171 x 23
    ['RT-IoT2022', ['no'], 'Attack_type'],  # 123,117 x 84
    ['QoS-QoE', ['RebufferingRatio', 'AvgVideoBitRate', 'AvgVideoQualityVariation'], 'StallLabel'],  # 69,129 x 38
    ['DeepSlice', ['no'], 'slice Type'],  # 63,167 x 10
    ['CIC-MalMem-2022', ['Category','Filename'], 'Class'],  # 58,598 x 58
    ['NSR', [], 'slice Type'],  # 31,583 x 17
    ['IoTNet24', ['id'], 'label'],  # 23,145 x 17
    ['IoT-APD', ['second'], 'label'],  # 10,845 x 17
    ['ASNM-CDX-2009', ['id', 'label_poly'], 'label_2'],  # 5,771 x 877
    ['UNAC', ['file'], 'output'],  # 389 x 23
    ['KPI-KQI', [], 'Service'],  # 165 x 14
    ['Social Network Ads', [], 'Purchased'],  # 400 x 3
]
cpu_models = [
    'SGD',
    'LR',
    'Ridge',
    'Perceptron',
    'DT',
    'ExtraTree',
    'LinearSVC',
    'GaussianNB',
    'BernoulliNB',
    'LightGBM_RF',
    'LightGBM_ExtraTrees',
    'AdaBoost',
    'Bagging',
    'KNN',
    'PassiveAggressive',
    'LDA',
    'QDA'
]

gpu_dl_models = [
    'XGBoost',
    'MLP',
    'DNN',
    'TabNet',
    'TabTransformer',
    'FT-Transformer',
    'TabICL',
    'TabPFN'
]

gpu_device_name = "cuda"

if run_gpu_dl_models:
    model_list = gpu_dl_models
else:
    model_list = cpu_models

# If GPU mode is enabled, avoid launching several processes competing for the same GPU.
tuning_stage_max_workers = 1 if run_gpu_dl_models and use_gpu_for_dl_models else (os.cpu_count() or 4)

if "TABPFN_TOKEN" not in os.environ and "TABPFN_TOKEN" in local_config:
    os.environ["TABPFN_TOKEN"] = local_config["TABPFN_TOKEN"]

os.environ["HF_TOKEN"] = "my_hf_token_" + experiment_name

# TabICL memory policy.
# Keep every predictive parameter unchanged and store the large column-wise
# intermediate tensor in regular CPU RAM. Pinned memory is disabled because a
# single multi-gigabyte pinned allocation can fail even when sufficient normal
# RAM is available. No disk offloading is used.
tabicl_offload_mode = "cpu"
tabicl_disk_offload_dir = None
tabicl_max_pinned_memory_mb = 0.0

# Conservative internal batching for CUDA inference. These values affect only
# how TabICL partitions and transfers intermediate tensors; they do not change
# the checkpoint, ensemble members, model inputs, AMP, or prediction logic.
tabicl_min_batch_size = 1
tabicl_safety_factor = 0.25
# Use the project's direct GPU-to-output transfer patch. The transfer remains
# synchronous, but enabling this route avoids TabICL's additional out.cpu()
# allocation for every internal batch.
tabicl_use_async = True
tabicl_async_depth = 1

n_splits = 5

# get dataset list ordered by dataset's number of rows
def get_dataset_list_ordered_by_rows(dataset_root: str | os.PathLike | None = None):
    """
    Return dataset_list ordered from the median-sized dataset outward.

    Example:
        If datasets sorted by rows are:
        [smallest, small, medium, large, largest]

        This returns:
        [medium, small, large, smallest, largest]

    This is useful for runtime estimation because the first completed jobs
    are more representative than starting only with small or large datasets.
    """

    import pandas as pd

    if dataset_root is None:
        dataset_root = DATASETS_DIR

    dataset_rows = []

    for dataset_setup in dataset_list:
        dataset_name = dataset_setup[0]
        dataset_folder = os.path.join(dataset_root, dataset_name)

        if not os.path.isdir(dataset_folder):
            raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")

        csv_files = [
            file for file in os.listdir(dataset_folder)
            if file.endswith(".csv")
        ]

        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in: {dataset_folder}")

        csv_path = os.path.join(dataset_folder, csv_files[0])

        # Count rows without loading the file into memory.
        # Much faster than pd.read_csv() for multi-million row CSVs.
        with open(csv_path, "rb") as _f:
            number_of_rows = sum(1 for _ in _f) - 1  # subtract header row

        dataset_rows.append((dataset_setup, number_of_rows))

    # Sort normally first
    dataset_rows = sorted(dataset_rows, key=lambda item: item[1])

    # Reorder from the median outward
    n = len(dataset_rows)

    if n == 0:
        return []

    middle = n // 2

    ordered = []
    ordered_indices = set()

    # Start with the median
    ordered.append(dataset_rows[middle])
    ordered_indices.add(middle)

    step = 1

    while len(ordered) < n:
        left = middle - step
        right = middle + step

        if left >= 0 and left not in ordered_indices:
            ordered.append(dataset_rows[left])
            ordered_indices.add(left)

        if right < n and right not in ordered_indices:
            ordered.append(dataset_rows[right])
            ordered_indices.add(right)

        step += 1

    return [dataset_setup for dataset_setup, _ in ordered]
