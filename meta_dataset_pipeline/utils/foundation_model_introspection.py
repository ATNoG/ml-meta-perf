"""
Shared logic for detecting the real parameter count of the pretrained
foundation-model checkpoints (TabICL, TabPFN) used by model_wrappers.py.

Unlike every other model in exp_stage_create_meta_dataset.py, TabICL and
TabPFN have no tunable hyperparameter that controls their capacity - they are
frozen pretrained transformers, so their "Processing Units Number" has to be
the real parameter count of the loaded checkpoint, not a formula derived from
hyperparameters. This module fits a tiny dummy batch (no real dataset, no
training loop) to force the checkpoint to load, then counts its real
torch parameters, so the resulting number tracks whatever checkpoint the
project's model_wrappers.py is actually using instead of a value hardcoded
by hand.

Used by exp_stage_create_meta_dataset.py for lazy, in-process detection. That
stage contains recorded fallback counts for environments without tabicl or
tabpfn installed.
"""


def _tiny_fit_data():
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.random((20, 4)).astype("float32")
    y = np.array([0, 1] * 10)
    return x, y


def _find_torch_module(obj, _depth=0, _seen=None):
    """
    Best-effort lookup of the underlying torch.nn.Module inside a fitted
    sklearn-style wrapper. Traverses instance attributes and common
    container types (list/tuple/set/dict) recursively, since library
    versions differ in what they expose and where (e.g. TabPFN exposes its
    ensemble backbone(s) through a list attribute).
    """
    import torch

    if _seen is None:
        _seen = set()
    if id(obj) in _seen or _depth > 5:
        return None
    _seen.add(id(obj))

    if isinstance(obj, torch.nn.Module):
        return obj

    if isinstance(obj, dict):
        children = list(obj.values())
    elif isinstance(obj, (list, tuple, set)):
        children = list(obj)
    else:
        obj_dict = getattr(obj, "__dict__", None)
        children = list(obj_dict.values()) if obj_dict else []

    for value in children:
        if isinstance(value, torch.nn.Module):
            return value

    for value in children:
        found = _find_torch_module(value, _depth + 1, _seen)
        if found is not None:
            return found

    return None


def detect_foundation_model_parameter_count(model_name):
    """
    Fits a tiny dummy batch through the given foundation model and returns
    its real parameter count. model_name must be "TabICL" or "TabPFN".

    Raises ImportError if the corresponding package is not installed, and
    RuntimeError if a fitted instance was obtained but no torch.nn.Module
    could be located inside it (the library's internal layout changed).
    """
    x, y = _tiny_fit_data()

    if model_name == "TabICL":
        from tabicl import TabICLClassifier
        estimator = TabICLClassifier()
    elif model_name == "TabPFN":
        from tabpfn import TabPFNClassifier
        estimator = TabPFNClassifier()
    else:
        raise ValueError(f"No foundation model detector for {model_name!r}.")

    estimator.fit(x, y)  # the backbone is only loaded/built on first fit()

    torch_module = _find_torch_module(estimator)
    if torch_module is None:
        raise RuntimeError(
            f"Could not find a torch.nn.Module anywhere inside the fitted "
            f"{model_name} estimator ({type(estimator)!r}).\n"
            f"Its top-level attributes are: {sorted(vars(estimator).keys())}\n"
            f"Inspect one of those (e.g. fitted_estimator.<attr>.__dict__) to "
            f"find the real torch.nn.Module and adjust _find_torch_module(), "
            f"or check the {model_name} package's docs/source for how it "
            f"exposes the backbone."
        )

    return sum(p.numel() for p in torch_module.parameters())
