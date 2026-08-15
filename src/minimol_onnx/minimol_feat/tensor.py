"""The three helpers graphium/features needs from graphium/utils/tensor.py."""

from typing import Any, Iterable, List, Union

import numpy as np
import torch
from torch import Tensor


def is_dtype_torch_tensor(dtype: Union[np.dtype, torch.dtype]) -> bool:
    """Verify if the dtype is a torch dtype."""
    return isinstance(dtype, torch.dtype) or (dtype == Tensor)


def is_dtype_numpy_array(dtype: Union[np.dtype, torch.dtype]) -> bool:
    """Verify if the dtype is a numpy dtype."""
    is_torch = is_dtype_torch_tensor(dtype)
    is_num = dtype in (int, float, complex)
    if hasattr(dtype, "__module__"):
        is_numpy = dtype.__module__ == "numpy"
    else:
        is_numpy = False

    return (not is_torch) and (not is_num) and is_numpy


def one_of_k_encoding(val: Any, classes: Iterable[Any]) -> List[int]:
    """Convert a single value to a one-hot vector of len(classes) + 1."""
    encoding = [False] * (len(classes) + 1)
    found = False
    for i, v in enumerate(classes):
        if v == val:
            encoding[i] = True
            found = True
            break
    if not found:
        encoding[-1] = True
    return encoding
