"""
Scikit-learn compatible wrappers for GPU/deep-learning tabular models.

This module intentionally contains the model classes and GPU/data-feeding
policy only. The Optuna search logic and classical sklearn model registry live
in model_utils.py.
"""

import inspect
import math
import os
import random
import sys
from collections import OrderedDict
from types import MethodType

# Keras backend must be selected before importing keras.
# Keep this at module import time so Keras models remain backend-agnostic
# while this project explicitly uses the PyTorch backend.
if "keras" not in sys.modules:
    os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_tabnet.tab_model import TabNetClassifier
from rtdl_revisiting_models import FTTransformer
from tab_transformer_pytorch import TabTransformer
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from exp_setup import (
    gpu_dl_models,
    run_gpu_dl_models,
    use_gpu_for_dl_models,
)


GPU_FIXED_BATCH_SIZE = 2048


GPU_TABNET_VIRTUAL_BATCH_SIZE = 256


GPU_TABICL_BATCH_SIZE = 1
FOUNDATION_MODEL_MAX_TRAIN_SAMPLES = 10_000


def _is_cuda_device_name(device_name):
    if device_name is None:
        return False

    return str(device_name).lower().startswith("cuda")


GPU_TORCH_NUM_WORKERS = 0


def get_torch_loader_kwargs(device, shuffle):
    """
    DataLoader settings for in-memory tabular tensors.

    For the custom PyTorch tabular models, CUDA tensors are preloaded into VRAM
    before DataLoader construction. Therefore workers, pinned-memory copies,
    persistent workers, and prefetching are intentionally disabled to avoid IPC
    overhead and unnecessary CPU -> GPU transfer machinery.
    """
    return {
        "shuffle": bool(shuffle),
        "num_workers": 0,
        "pin_memory": False,
        "drop_last": False,
    }


def resolve_torch_device(device_name):
    """
    Resolve the PyTorch device used by custom torch-based tabular models.
    The project flags still decide whether GPU DL models are enabled.
    """
    if str(device_name) == "auto":
        use_cuda = (
            run_gpu_dl_models
            and use_gpu_for_dl_models
            and torch.cuda.is_available()
        )
        return torch.device("cuda" if use_cuda else "cpu")

    device = torch.device(str(device_name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested CUDA device '{device_name}', but torch.cuda.is_available() is False."
        )

    return device


def tensor_on_device(data, dtype, device):
    """Create a tensor directly on the target device when possible."""
    return torch.as_tensor(data, dtype=dtype, device=device)


def release_cuda_memory_if_needed(model_name):
    """Release cached CUDA memory after each GPU-model trial/final fit."""
    if (
        run_gpu_dl_models
        and use_gpu_for_dl_models
        and model_name in gpu_dl_models
        and torch.cuda.is_available()
    ):
        torch.cuda.empty_cache()


def make_adaptive_train_validation_split(X_np, y_np, validation_split, seed):
    """
    Create an internal validation split for early stopping.

    The split is created only from the fold-training data passed to fit(), so it
    does not leak information from the outer CV validation fold.
    """
    validation_split = float(validation_split)

    if validation_split <= 0.0 or X_np.shape[0] < 10:
        return X_np, y_np, None, None

    unique_labels, label_counts = np.unique(y_np, return_counts=True)
    if len(unique_labels) < 2 or np.min(label_counts) < 2:
        return X_np, y_np, None, None

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_np,
            y_np,
            test_size=validation_split,
            random_state=int(seed),
            stratify=y_np,
        )
    except ValueError:
        X_train, X_val, y_train, y_val = train_test_split(
            X_np,
            y_np,
            test_size=validation_split,
            random_state=int(seed),
            stratify=None,
        )

    return X_train, y_train, X_val, y_val


def subsample_context(X_np, y_np, max_samples, seed):
    """
    Down-sample (X, y) to at most `max_samples` rows, preserving class ratios.

    Used by the in-context foundation models (TabICL and TabPFN), whose `.fit()`
    stores the training set as transformer context. The context is capped to the
    same reproducible stratified subsample for both models. Falls back to a plain
    random subsample when stratification is not possible (e.g. a class with a
    single member).
    """
    if not max_samples or X_np.shape[0] <= int(max_samples):
        return X_np, y_np

    try:
        X_sub, _, y_sub, _ = train_test_split(
            X_np,
            y_np,
            train_size=int(max_samples),
            random_state=int(seed),
            stratify=y_np,
        )
    except ValueError:
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(X_np.shape[0], size=int(max_samples), replace=False)
        X_sub, y_sub = X_np[idx], y_np[idx]

    return X_sub, y_sub


def resolve_adaptive_patience(max_epochs, n_batches_per_epoch, patience):
    """Compute patience from data scale unless the user provides a fixed value."""
    if patience == "auto" or patience is None:
        return max(5, min(25, int(200 / max(1, n_batches_per_epoch))))

    return int(patience)


def resolve_warmup_epochs(max_epochs):
    """Warm up before allowing early stopping to avoid stopping on initial noise."""
    max_epochs = int(max_epochs)
    return max(1, max(15, int(max_epochs * 0.1)))


def clone_state_dict_to_cpu(model):
    """Save a detached CPU copy of a model state dict for best-epoch restoration."""
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _ensure_keras_torch_backend():
    """
    Verify that Keras is using the expected PyTorch backend.

    Keras model code below remains framework-agnostic: it does not use PyTorch
    device scopes or direct tensor transfers. Device management is delegated to
    Keras and its selected backend.
    """
    backend_name = keras.backend.backend()
    if backend_name != "torch":
        raise RuntimeError(
            f"Keras was imported with backend '{backend_name}', but this project "
            "currently expects KERAS_BACKEND='torch'. Set KERAS_BACKEND=torch "
            "before starting Python, or avoid importing keras/tensorflow before "
            "model_utils."
        )

    return keras


class KerasTorchSklearnClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
            self,
            hidden_layer_sizes=(100,),
            activation="relu",
            alpha=0.0001,
            solver="adam",
            batch_size=32,
            max_iter=200,
            learning_rate_init=0.001,
            validation_split=0.0,
            seed=0,
            device_name="cpu",
            verbose=0
    ):
        # Parameter names intentionally mimic sklearn.neural_network.MLPClassifier
        # so the existing Optuna grids and saved configurations keep working.
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.solver = solver
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.validation_split = validation_split
        self.seed = seed
        self.device_name = device_name
        self.verbose = verbose

    @staticmethod
    def _to_dense_float32(X):
        if hasattr(X, "toarray"):
            X = X.toarray()

        return np.asarray(X, dtype=np.float32)

    @staticmethod
    def _map_activation(activation):
        # sklearn calls the sigmoid activation "logistic". Keras calls it "sigmoid".
        if activation == "logistic":
            return "sigmoid"

        return activation

    def _build_model(self, n_features, n_classes):
        keras = _ensure_keras_torch_backend()
        layers = keras.layers
        regularizers = keras.regularizers

        activation = self._map_activation(str(self.activation))
        kernel_regularizer = None
        if self.alpha is not None and float(self.alpha) > 0:
            kernel_regularizer = regularizers.l2(float(self.alpha))

        inputs = keras.Input(shape=(int(n_features),), dtype="float32")
        x = inputs

        for units in tuple(self.hidden_layer_sizes):
            x = layers.Dense(
                int(units),
                activation=activation,
                kernel_regularizer=kernel_regularizer
            )(x)

        outputs = layers.Dense(int(n_classes), activation="softmax")(x)
        model = keras.Model(inputs=inputs, outputs=outputs)

        solver = str(self.solver).lower()
        if solver == "adam":
            optimizer = keras.optimizers.Adam(
                learning_rate=float(self.learning_rate_init)
            )
        elif solver == "sgd":
            optimizer = keras.optimizers.SGD(
                learning_rate=float(self.learning_rate_init)
            )
        else:
            raise ValueError(
                "KerasTorchSklearnClassifier only supports solver='adam' "
                "or solver='sgd'."
            )

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def fit(self, X, y):
        keras = _ensure_keras_torch_backend()

        keras.utils.set_random_seed(int(self.seed))

        X_np = self._to_dense_float32(X)

        self.label_encoder_ = LabelEncoder()
        y_np = self.label_encoder_.fit_transform(y).astype(np.int64)
        self.classes_ = self.label_encoder_.classes_

        self.model_ = self._build_model(
            n_features=X_np.shape[1],
            n_classes=len(self.classes_)
        )

        fit_kwargs = {
            "x": X_np,
            "y": y_np,
            "epochs": int(self.max_iter),
            "batch_size": int(self.batch_size),
            "verbose": int(self.verbose)
        }

        if float(self.validation_split) > 0:
            fit_kwargs["validation_split"] = float(self.validation_split)

        self.history_ = self.model_.fit(**fit_kwargs)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")

        X_np = self._to_dense_float32(X)
        y_proba = self.model_.predict(
            X_np,
            batch_size=int(self.batch_size),
            verbose=0
        )

        return np.asarray(y_proba, dtype=np.float64)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1).astype(int)

        return self.label_encoder_.inverse_transform(y_pred)


keras_torch_models = {"MLP", "DNN"}


class TabNetSklearnClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
            self,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            lambda_sparse=1e-3,
            mask_type="sparsemax",
            max_epochs=200,
            patience=10,
            validation_split=0.1,
            batch_size=128,
            virtual_batch_size=32,
            seed=0,
            device_name="cpu",
            verbose=0
    ):
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.mask_type = mask_type
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.seed = seed
        self.device_name = device_name
        self.verbose = verbose

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)

        self.label_encoder_ = LabelEncoder()
        y_np = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Hold out an internal validation split so TabNet's early stopping is
        # actually active. Without an eval_set, pytorch-tabnet ignores `patience`
        # and always runs the full `max_epochs`. The split comes only from the
        # fold-training data, so it does not leak the outer CV validation fold.
        X_tr, y_tr, X_val, y_val = make_adaptive_train_validation_split(
            X_np,
            y_np,
            validation_split=self.validation_split,
            seed=self.seed,
        )

        self.model_ = TabNetClassifier(
            n_d=int(self.n_d),
            n_a=int(self.n_a),
            n_steps=int(self.n_steps),
            gamma=float(self.gamma),
            lambda_sparse=float(self.lambda_sparse),
            mask_type=str(self.mask_type),
            seed=int(self.seed),
            device_name=str(self.device_name),
            verbose=int(self.verbose)
        )

        fit_kwargs = dict(
            max_epochs=int(self.max_epochs),
            batch_size=int(self.batch_size),
            virtual_batch_size=int(min(self.virtual_batch_size, self.batch_size)),
            num_workers=0,
            drop_last=False,
            pin_memory=_is_cuda_device_name(self.device_name),
            compute_importance=False,
        )

        if X_val is not None:
            # eval_set enables early stopping and best-weight restoration.
            fit_kwargs.update(
                eval_set=[(X_val, y_val)],
                eval_name=["val"],
                eval_metric=["logloss"],
                patience=int(self.patience),
            )
        else:
            # Too few samples to validate: disable early stopping explicitly.
            fit_kwargs["patience"] = 0

        self.model_.fit(X_tr, y_tr, **fit_kwargs)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")

        X_np = np.asarray(X, dtype=np.float32)
        y_proba = self.model_.predict_proba(X_np)

        return np.asarray(y_proba, dtype=np.float64)

    def predict(self, X):
        check_is_fitted(self, "model_")

        X_np = np.asarray(X, dtype=np.float32)
        y_pred = self.model_.predict(X_np)

        return self.label_encoder_.inverse_transform(np.asarray(y_pred).astype(int))


class LucidTabTransformerSklearnClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
            self,
            mlp_hidden_mults=(4, 2),
            lr=1e-3,
            weight_decay=1e-5,
            max_epochs=200,
            patience="auto",
            min_delta="auto",
            validation_split=0.1,
            batch_size=128,
            seed=0,
            device_name="cpu",
            verbose=0,
            dim=32,
            depth=1,
            heads=1,
            dim_head=16
    ):
        # In this project, preprocessing converts all columns to numeric values.
        # Therefore, TabTransformer is used with categories=() and only continuous inputs.
        # In that configuration, the categorical transformer branch is skipped, so the
        # useful tunable part is the final MLP plus the training hyperparameters.
        self.mlp_hidden_mults = mlp_hidden_mults
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.seed = seed
        self.device_name = device_name
        self.verbose = verbose

        # Required by the TabTransformer constructor, but fixed because there are no
        # categorical variables in the current preprocessing pipeline.
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head

    def fit(self, X, y):
        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))
        random.seed(int(self.seed))

        X_np = np.asarray(X, dtype=np.float32)

        self.label_encoder_ = LabelEncoder()
        y_np = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        n_features = X_np.shape[1]
        n_classes = len(self.classes_)

        X_train, y_train, X_val, y_val = make_adaptive_train_validation_split(
            X_np,
            y_np,
            validation_split=self.validation_split,
            seed=self.seed,
        )

        self.device_ = resolve_torch_device(self.device_name)

        self.model_ = TabTransformer(
            categories=(),                          # No categorical variables after preprocessing
            num_continuous=int(n_features),         # All input columns are continuous/numeric
            dim=int(self.dim),                      # Fixed; categorical transformer branch is skipped
            depth=int(self.depth),                  # Fixed; categorical transformer branch is skipped
            heads=int(self.heads),                  # Fixed; categorical transformer branch is skipped
            dim_head=int(self.dim_head),            # Fixed; categorical transformer branch is skipped
            dim_out=int(n_classes),
            mlp_hidden_mults=tuple(self.mlp_hidden_mults),
            mlp_act=nn.ReLU(),
            attn_dropout=0.0,
            ff_dropout=0.0,
            use_shared_categ_embed=False,
            num_residual_streams=1
        ).to(self.device_)

        x_cont_train = tensor_on_device(X_train, torch.float32, self.device_)
        x_categ_train = torch.empty(
            (X_train.shape[0], 0),
            dtype=torch.long,
            device=self.device_,
        )
        y_tensor_train = tensor_on_device(y_train, torch.long, self.device_)

        train_dataset = TensorDataset(x_categ_train, x_cont_train, y_tensor_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.batch_size),
            **get_torch_loader_kwargs(self.device_, shuffle=True),
        )

        val_loader = None
        if X_val is not None:
            x_cont_val = tensor_on_device(X_val, torch.float32, self.device_)
            x_categ_val = torch.empty(
                (X_val.shape[0], 0),
                dtype=torch.long,
                device=self.device_,
            )
            y_tensor_val = tensor_on_device(y_val, torch.long, self.device_)
            val_dataset = TensorDataset(x_categ_val, x_cont_val, y_tensor_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=int(self.batch_size) * 2,
                **get_torch_loader_kwargs(self.device_, shuffle=False),
            )

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

        loss_fn = nn.CrossEntropyLoss()
        verbose = int(self.verbose) > 0

        n_batches_per_epoch = max(1, len(train_loader))
        warmup_epochs = resolve_warmup_epochs(self.max_epochs)
        adaptive_patience = resolve_adaptive_patience(
            max_epochs=self.max_epochs,
            n_batches_per_epoch=n_batches_per_epoch,
            patience=self.patience,
        )
        adaptive_min_delta = None if self.min_delta == "auto" else float(self.min_delta)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        self.n_epochs_ = int(self.max_epochs)

        for epoch in range(int(self.max_epochs)):
            epoch_loss = 0.0
            self.model_.train()

            for batch_categ, batch_cont, batch_y in train_loader:
                optimizer.zero_grad(set_to_none=True)

                logits = self.model_(batch_categ, batch_cont)
                loss = loss_fn(logits, batch_y)

                loss.backward()
                optimizer.step()

                if verbose:
                    epoch_loss += float(loss.detach().item())

            self.n_epochs_ = epoch + 1

            if val_loader is None:
                if verbose:
                    print(
                        f"TabTransformer epoch {epoch + 1}/{self.max_epochs}, "
                        f"loss={epoch_loss:.4f}"
                    )
                continue

            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_categ, batch_cont, batch_y in val_loader:
                    logits = self.model_(batch_categ, batch_cont)
                    val_loss += float(loss_fn(logits, batch_y).detach().item())

            val_loss /= max(1, len(val_loader))

            if epoch == 0 and adaptive_min_delta is None:
                # Scale-invariant threshold: 0.05% of the initial validation loss.
                adaptive_min_delta = max(1e-5, 0.0005 * float(val_loss))

            if verbose:
                print(
                    f"TabTransformer epoch {epoch + 1}/{self.max_epochs}, "
                    f"train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"patience={adaptive_patience}, min_delta={adaptive_min_delta:.6g}"
                )

            improvement_threshold = adaptive_min_delta if adaptive_min_delta is not None else 1e-4
            if val_loss < best_val_loss - improvement_threshold:
                best_val_loss = float(val_loss)
                best_model_state = clone_state_dict_to_cpu(self.model_)
                patience_counter = 0
            elif epoch >= warmup_epochs:
                patience_counter += 1
                if patience_counter >= adaptive_patience:
                    if verbose:
                        print(
                            f"TabTransformer early stopping at epoch {epoch + 1}; "
                            f"best_val_loss={best_val_loss:.4f}"
                        )
                    break

        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")

        X_np = np.asarray(X, dtype=np.float32)

        x_cont = tensor_on_device(X_np, torch.float32, self.device_)
        x_categ = torch.empty(
            (X_np.shape[0], 0),
            dtype=torch.long,
            device=self.device_,
        )
        dataset = TensorDataset(x_categ, x_cont)

        loader = DataLoader(
            dataset,
            batch_size=int(self.batch_size) * 2,
            **get_torch_loader_kwargs(self.device_, shuffle=False),
        )

        self.model_.eval()
        probabilities = []

        with torch.no_grad():
            for batch_categ, batch_cont in loader:
                logits = self.model_(batch_categ, batch_cont)
                probabilities.append(torch.softmax(logits, dim=1))

        return torch.cat(probabilities, dim=0).cpu().numpy()

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1).astype(int)

        return self.label_encoder_.inverse_transform(y_pred)


class RTDLFTTransformerSklearnClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
            self,
            n_blocks=3,
            d_block=128,
            attention_n_heads=4,
            attention_dropout=0.1,
            ffn_dropout=0.1,
            residual_dropout=0.0,
            lr=1e-3,
            weight_decay=1e-5,
            max_epochs=200,
            patience="auto",
            min_delta="auto",
            validation_split=0.1,
            batch_size=128,
            micro_batch_size=None,
            stream_batches_from_cpu=False,
            seed=0,
            device_name="cpu",
            verbose=0
    ):
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.validation_split = validation_split
        self.batch_size = batch_size
        self.micro_batch_size = micro_batch_size
        self.stream_batches_from_cpu = stream_batches_from_cpu
        self.seed = seed
        self.device_name = device_name
        self.verbose = verbose

    def fit(self, X, y):
        torch.manual_seed(int(self.seed))
        np.random.seed(int(self.seed))
        random.seed(int(self.seed))

        X_np = np.asarray(X, dtype=np.float32)

        self.label_encoder_ = LabelEncoder()
        y_np = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        n_features = X_np.shape[1]
        n_classes = len(self.classes_)

        X_train, y_train, X_val, y_val = make_adaptive_train_validation_split(
            X_np,
            y_np,
            validation_split=self.validation_split,
            seed=self.seed,
        )

        self.device_ = resolve_torch_device(self.device_name)

        self.model_ = FTTransformer(
            n_cont_features=int(n_features),
            cat_cardinalities=[],                 # No categorical features
            d_out=int(n_classes),
            n_blocks=int(self.n_blocks),
            d_block=int(self.d_block),
            attention_n_heads=int(self.attention_n_heads),
            attention_dropout=float(self.attention_dropout),
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=float(self.ffn_dropout),
            residual_dropout=float(self.residual_dropout),
        ).to(self.device_)

        data_device = torch.device("cpu") if self.stream_batches_from_cpu else self.device_
        x_cont_train = tensor_on_device(X_train, torch.float32, data_device)
        y_tensor_train = tensor_on_device(y_train, torch.long, data_device)

        effective_batch_size = int(self.batch_size)
        if self.micro_batch_size is not None:
            micro_batch_size = int(self.micro_batch_size)
        else:
            micro_batch_size = effective_batch_size
        accumulation_steps = max(1, math.ceil(effective_batch_size / micro_batch_size))
        actual_micro_batch_size = min(micro_batch_size, effective_batch_size)

        train_dataset = TensorDataset(x_cont_train, y_tensor_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=actual_micro_batch_size,
            **get_torch_loader_kwargs(self.device_, shuffle=True),
        )

        val_loader = None
        if X_val is not None:
            x_cont_val = tensor_on_device(X_val, torch.float32, data_device)
            y_tensor_val = tensor_on_device(y_val, torch.long, data_device)
            val_dataset = TensorDataset(x_cont_val, y_tensor_val)
            val_batch_size = min(actual_micro_batch_size * 2, effective_batch_size * 2)
            val_loader = DataLoader(
                val_dataset,
                batch_size=val_batch_size,
                **get_torch_loader_kwargs(self.device_, shuffle=False),
            )

        if hasattr(self.model_, "make_parameter_groups"):
            parameters = self.model_.make_parameter_groups()
        else:
            parameters = self.model_.parameters()

        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.lr),
            weight_decay=float(self.weight_decay),
        )

        loss_fn = nn.CrossEntropyLoss()
        verbose = int(self.verbose) > 0

        n_micro_batches = max(1, len(train_loader))
        n_batches_per_epoch = max(1, math.ceil(n_micro_batches / accumulation_steps))
        warmup_epochs = resolve_warmup_epochs(self.max_epochs)
        adaptive_patience = resolve_adaptive_patience(
            max_epochs=self.max_epochs,
            n_batches_per_epoch=n_batches_per_epoch,
            patience=self.patience,
        )
        adaptive_min_delta = None if self.min_delta == "auto" else float(self.min_delta)

        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0
        self.n_epochs_ = int(self.max_epochs)

        for epoch in range(int(self.max_epochs)):
            epoch_loss = 0.0
            self.model_.train()
            optimizer.zero_grad(set_to_none=True)

            for step, (batch_cont, batch_y) in enumerate(train_loader):
                batch_cont = batch_cont.to(self.device_)
                batch_y = batch_y.to(self.device_)

                logits = self.model_(batch_cont, None)
                loss = loss_fn(logits, batch_y) / accumulation_steps

                loss.backward()

                if verbose:
                    epoch_loss += float(loss.detach().item() * accumulation_steps)

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # Flush any remaining gradients from an incomplete accumulation.
            if (step + 1) % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            self.n_epochs_ = epoch + 1

            if val_loader is None:
                if verbose:
                    print(
                        f"FT-Transformer epoch {epoch + 1}/{self.max_epochs}, "
                        f"loss={epoch_loss:.4f}"
                    )
                continue

            self.model_.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_cont, batch_y in val_loader:
                    batch_cont = batch_cont.to(self.device_)
                    batch_y = batch_y.to(self.device_)

                    logits = self.model_(batch_cont, None)
                    val_loss += float(loss_fn(logits, batch_y).detach().item())

            val_loss /= max(1, len(val_loader))

            if epoch == 0 and adaptive_min_delta is None:
                # Scale-invariant threshold: 0.05% of the initial validation loss.
                adaptive_min_delta = max(1e-5, 0.0005 * float(val_loss))

            if verbose:
                print(
                    f"FT-Transformer epoch {epoch + 1}/{self.max_epochs}, "
                    f"train_loss={epoch_loss:.4f}, val_loss={val_loss:.4f}, "
                    f"patience={adaptive_patience}, min_delta={adaptive_min_delta:.6g}"
                )

            improvement_threshold = adaptive_min_delta if adaptive_min_delta is not None else 1e-4
            if val_loss < best_val_loss - improvement_threshold:
                best_val_loss = float(val_loss)
                best_model_state = clone_state_dict_to_cpu(self.model_)
                patience_counter = 0
            elif epoch >= warmup_epochs:
                patience_counter += 1
                if patience_counter >= adaptive_patience:
                    if verbose:
                        print(
                            f"FT-Transformer early stopping at epoch {epoch + 1}; "
                            f"best_val_loss={best_val_loss:.4f}"
                        )
                    break

        if best_model_state is not None:
            self.model_.load_state_dict(best_model_state)

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")

        X_np = np.asarray(X, dtype=np.float32)

        data_device = torch.device("cpu") if self.stream_batches_from_cpu else self.device_
        x_cont = tensor_on_device(X_np, torch.float32, data_device)
        dataset = TensorDataset(x_cont)

        loader_batch_size = (
            int(self.micro_batch_size) * 2
            if self.micro_batch_size is not None
            else int(self.batch_size) * 2
        )
        loader = DataLoader(
            dataset,
            batch_size=loader_batch_size,
            **get_torch_loader_kwargs(self.device_, shuffle=False),
        )

        self.model_.eval()
        probabilities = []

        with torch.no_grad():
            for (batch_cont,) in loader:
                batch_cont = batch_cont.to(self.device_)
                logits = self.model_(batch_cont, None)
                probabilities.append(torch.softmax(logits, dim=1))

        return torch.cat(probabilities, dim=0).cpu().numpy()

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1).astype(int)

        return self.label_encoder_.inverse_transform(y_pred)


def _tabicl_memory_efficient_row_inference(
        row_interactor,
        embeddings,
        mgr_config=None,
):
    """
    Run TabICL's row-interaction inference without materializing class tokens
    for every row.

    TabICL expands the small class-token parameter to (B, T, C, E) before
    moving it to the embeddings device. When the column representations are
    offloaded to CPU, that device transfer materializes a multi-gigabyte
    temporary tensor. Moving the original (C, E) parameter first and relying
    on assignment broadcasting writes exactly the same values.
    """
    if mgr_config is None:
        from tabicl import InferenceConfig as TabICLInferenceConfig

        mgr_config = TabICLInferenceConfig().ROW_CONFIG

    row_interactor.inference_mgr.configure(**mgr_config)

    cls_tokens = row_interactor.cls_tokens.to(embeddings.device)
    embeddings[:, :, : row_interactor.num_cls] = cls_tokens

    return row_interactor.inference_mgr(
        row_interactor._aggregate_embeddings,
        inputs=OrderedDict([("embeddings", embeddings)]),
    )


def _tabicl_direct_output_copy(
        copy_manager,
        gpu_tensor,
        target,
        indices,
):
    """
    Copy an inference batch directly into TabICL's preallocated CPU/disk output.

    TabICL's synchronous path first materializes ``gpu_tensor.cpu()`` and then
    copies that temporary tensor into the output buffer. The async path normally
    uses an equally large pinned staging buffer. A synchronous cross-device
    ``copy_`` writes into the final slice directly and avoids both temporaries.
    """
    target[indices].copy_(gpu_tensor, non_blocking=False)
    copy_manager._bytes_written += (
        gpu_tensor.numel() * gpu_tensor.element_size() / (1024 * 1024)
    )


def _install_tabicl_direct_output_copy():
    """Install the transfer fix in memory without modifying the TabICL package."""
    from tabicl._model.inference import AsyncCopyManager

    if getattr(AsyncCopyManager, "_meta2perf_direct_copy_installed", False):
        return

    AsyncCopyManager.submit_copy = _tabicl_direct_output_copy
    AsyncCopyManager._meta2perf_direct_copy_installed = True


def _tabicl_device_safe_hierarchical_predict(
        icl_predictor,
        R_test,
        softmax_temperature=0.9,
        inference_recurrence=None,
):
    """
    Run TabICL hierarchical classification with consistent tensor devices.

    TabICL calls _predict_standard(auto_batch=False) in this path. That branch
    returns CUDA output even when the surrounding ICL configuration offloads
    representations to CPU, so group probabilities and child probabilities can
    end up on different devices. Moving only the returned probabilities to the
    R_test device preserves their values and the original hierarchy calculation.
    """
    del inference_recurrence  # Kept in the signature for TabICL compatibility.

    test_size = R_test.shape[0]
    device = R_test.device
    num_classes = len(icl_predictor.root.classes_)

    def process_node(node, node_R_test):
        combined_R = torch.cat([node.R.to(device), node_R_test], dim=0)

        if node.is_leaf:
            node_y = icl_predictor._label_encoding(node.y.to(device))
            leaf_preds = icl_predictor._predict_standard(
                R=combined_R.unsqueeze(0),
                y_train=node_y.unsqueeze(0),
                softmax_temperature=softmax_temperature,
                auto_batch=False,
            ).squeeze(0).to(device)

            global_preds = torch.zeros(
                (test_size, num_classes),
                device=device,
            )
            for local_idx, global_idx in enumerate(node.classes_):
                global_preds[:, global_idx] = leaf_preds[:, local_idx]

            return global_preds

        final_probs = torch.zeros(
            (test_size, num_classes),
            device=device,
        )

        node_y = node.group_indices.to(device)
        group_probs = icl_predictor._predict_standard(
            R=combined_R.unsqueeze(0),
            y_train=node_y.unsqueeze(0),
            softmax_temperature=softmax_temperature,
            auto_batch=False,
        ).squeeze(0).to(device)

        for group_idx, child_node in enumerate(node.child_nodes):
            child_probs = process_node(child_node, node_R_test)
            final_probs += child_probs * group_probs[:, group_idx: group_idx + 1]

        return final_probs

    return process_node(icl_predictor.root, R_test)


class TabICLSklearnClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
            self,
            n_estimators=8,
            softmax_temperature=0.9,
            average_logits=True,
            batch_size=1,
            kv_cache=False,
            checkpoint_version="tabicl-classifier-v2-20260212.ckpt",
            device_name=None,
            seed=0,
            n_jobs=None,
            verbose=False,
            allow_auto_download=True,
            offload_mode="auto",
            disk_offload_dir=None,
            max_pinned_memory_mb=0.0,
            min_batch_size=1,
            safety_factor=0.5,
            use_async=False,
            async_depth=1,
            max_train_samples=FOUNDATION_MODEL_MAX_TRAIN_SAMPLES,
    ):
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.average_logits = average_logits
        self.batch_size = batch_size
        self.kv_cache = kv_cache
        self.checkpoint_version = checkpoint_version
        self.device_name = device_name
        self.seed = seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.allow_auto_download = allow_auto_download
        self.offload_mode = offload_mode
        self.disk_offload_dir = disk_offload_dir
        self.max_pinned_memory_mb = max_pinned_memory_mb
        self.min_batch_size = min_batch_size
        self.safety_factor = safety_factor
        self.use_async = use_async
        self.async_depth = async_depth
        self.max_train_samples = max_train_samples

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)

        self.label_encoder_ = LabelEncoder()
        y_np = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Use exactly the same reproducible, stratified context sampling policy
        # as TabPFN. The label encoder is fitted on the complete label set first,
        # so its class mapping remains stable after sampling.
        X_np, y_np = subsample_context(
            X_np, y_np, self.max_train_samples, self.seed
        )

        # TabICL is already scikit-learn compliant. This wrapper is used only to
        # keep a consistent interface with the rest of the custom models and to
        # encode/decode labels safely in this project.
        device = None if self.device_name in [None, "auto"] else str(self.device_name)
        n_jobs = None if self.n_jobs is None else int(self.n_jobs)

        from tabicl import InferenceConfig as TabICLInferenceConfig
        _inference_config = TabICLInferenceConfig()

        # Apply the same conservative memory policy to all three inference
        # stages. This changes only batching/offloading of intermediate tensors;
        # predictive parameters and numerical precision remain untouched.
        common_memory_config = {
            "min_batch_size": int(self.min_batch_size),
            "safety_factor": float(self.safety_factor),
            "offload": self.offload_mode,
            "disk_offload_dir": self.disk_offload_dir,
            "max_pinned_memory_mb": float(self.max_pinned_memory_mb),
            "use_async": bool(self.use_async),
            "async_depth": int(self.async_depth),
        }

        for stage_config in (
            _inference_config.COL_CONFIG,
            _inference_config.ROW_CONFIG,
            _inference_config.ICL_CONFIG,
        ):
            stage_config.update(common_memory_config)

        _install_tabicl_direct_output_copy()

        self.model_ = TabICLClassifier(
            n_estimators=int(self.n_estimators),
            softmax_temperature=float(self.softmax_temperature),
            average_logits=bool(self.average_logits),
            batch_size=int(self.batch_size),
            kv_cache=bool(self.kv_cache),
            checkpoint_version=None if self.checkpoint_version is None else str(self.checkpoint_version),
            device=device,
            random_state=int(self.seed),
            n_jobs=n_jobs,
            verbose=bool(self.verbose),
            allow_auto_download=bool(self.allow_auto_download),
            inference_config=_inference_config,
        )

        self.model_.fit(X_np, y_np)

        # Apply the memory fix only to this fitted TabICL instance. This avoids
        # modifying the installed package while eliminating the large temporary
        # class-token allocation in RowInteraction._inference_forward().
        row_interactor = self.model_.model_.row_interactor
        row_interactor._inference_forward = MethodType(
            _tabicl_memory_efficient_row_inference,
            row_interactor,
        )

        # TabICL's hierarchical many-class path can mix CPU-offloaded child
        # probabilities with CUDA group probabilities. Patch only this fitted
        # instance so both tensors are combined on the representation device.
        icl_predictor = self.model_.model_.icl_predictor
        icl_predictor._predict_hierarchical = MethodType(
            _tabicl_device_safe_hierarchical_predict,
            icl_predictor,
        )

        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")

        if not hasattr(self.model_, "predict_proba"):
            raise AttributeError("The installed TabICLClassifier does not expose predict_proba().")

        X_np = np.asarray(X, dtype=np.float32)
        y_proba = self.model_.predict_proba(X_np)

        return np.asarray(y_proba, dtype=np.float64)

    def predict(self, X):
        print("Predicting...")

        check_is_fitted(self, "model_")

        X_np = np.asarray(X, dtype=np.float32)
        y_pred = self.model_.predict(X_np)

        return self.label_encoder_.inverse_transform(np.asarray(y_pred).astype(int))


class TabPFNSklearnClassifier(ClassifierMixin, BaseEstimator):
    _estimator_type = "classifier"

    def __init__(
            self,
            n_estimators=8,
            softmax_temperature=0.9,
            balance_probabilities=False,
            average_before_softmax=False,
            fit_mode="fit_preprocessors",
            memory_saving_mode="auto",
            ignore_pretraining_limits=True,
            inference_precision="auto",
            device_name="auto",
            seed=0,
            n_preprocessing_jobs=1,
            max_train_samples=FOUNDATION_MODEL_MAX_TRAIN_SAMPLES,
            show_progress_bar=False
    ):
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.fit_mode = fit_mode
        self.memory_saving_mode = memory_saving_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.device_name = device_name
        self.seed = seed
        self.n_preprocessing_jobs = n_preprocessing_jobs
        # Match TabICL's in-context sampling policy. Set to None to disable and
        # use the full training set.
        self.max_train_samples = max_train_samples
        self.show_progress_bar = show_progress_bar

    def fit(self, X, y):
        X_np = np.asarray(X, dtype=np.float32)

        self.label_encoder_ = LabelEncoder()
        y_np = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        # Bound the context size fed to the transformer (see max_train_samples).
        # classes_ is taken from the full label set above, so the encoder mapping
        # stays valid even if a rare class is dropped from the subsample.
        X_np, y_np = subsample_context(
            X_np, y_np, self.max_train_samples, self.seed
        )

        device = "auto" if self.device_name in [None, "auto"] else str(self.device_name)

        # Current TabPFN versions are sklearn-compatible, but their constructor
        # arguments have changed across releases. Therefore, we filter kwargs
        # against the installed version's __init__ signature.
        candidate_kwargs = {
            "n_estimators": int(self.n_estimators),
            "softmax_temperature": float(self.softmax_temperature),
            "balance_probabilities": bool(self.balance_probabilities),
            "average_before_softmax": bool(self.average_before_softmax),
            "fit_mode": str(self.fit_mode),
            "memory_saving_mode": self.memory_saving_mode,
            "ignore_pretraining_limits": bool(self.ignore_pretraining_limits),
            "inference_precision": self.inference_precision,
            "device": device,
            "random_state": int(self.seed),
            "n_preprocessing_jobs": int(self.n_preprocessing_jobs),
            "show_progress_bar": bool(self.show_progress_bar),
        }

        signature = inspect.signature(TabPFNClassifier.__init__)
        supported_kwargs = {
            key: value
            for key, value in candidate_kwargs.items()
            if key in signature.parameters
        }

        self.model_ = TabPFNClassifier(**supported_kwargs)
        self.model_.fit(X_np, y_np)

        return self

    def predict(self, X):
        check_is_fitted(self, "model_")

        X_np = np.asarray(X, dtype=np.float32)
        y_pred = self.model_.predict(X_np)

        return self.label_encoder_.inverse_transform(np.asarray(y_pred).astype(int))

    def predict_proba(self, X):
        check_is_fitted(self, "model_")

        X_np = np.asarray(X, dtype=np.float32)
        return self.model_.predict_proba(X_np)
