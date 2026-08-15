"""Build ``src/minimol_onnx/minimol_feat/`` -- graphium's featurizer, standalone.

Copies ``graphium/features/*`` plus three helpers and rewrites the imports.
Re-runnable; overwrites the output directory.

Usage::

    python vendor.py [--source <site-packages/graphium>] [--output ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
from pathlib import Path

MODULES = [
    "__init__.py",
    "commute.py",
    "electrostatic.py",
    "featurizer.py",
    "graphormer.py",
    "nmp.py",
    "positional_encoding.py",
    "properties.py",
    "rw.py",
    "spectral.py",
    "transfer_pos_level.py",
]
DATA_FILES = ["periodic_table.csv"]

def _installed_graphium() -> Path | None:
    """Locate graphium in the current interpreter, whatever env it lives in."""
    spec = importlib.util.find_spec("graphium")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


DEFAULT_SOURCE = _installed_graphium()
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "src" / "minimol_onnx" / "minimol_feat"
)

IMPORT_REWRITES = [
    (r"\bfrom graphium\.features\.(\w+) import", r"from .\1 import"),
    (r"\bfrom graphium\.features import", r"from . import"),
    (r"\bfrom graphium\.utils\.tensor import", r"from .tensor import"),
    (r"\bfrom torch_scatter import", r"from .pure_scatter import"),
]

# nmp.py loads its data file out of the installed graphium package.
NMP_OLD = 'with importlib.resources.open_text("graphium.features", "periodic_table.csv") as f:\n    PERIODIC_TABLE = pd.read_csv(f)'
NMP_NEW = '''_PERIODIC_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "periodic_table.csv")
with open(_PERIODIC_TABLE_PATH) as f:
    PERIODIC_TABLE = pd.read_csv(f)'''

# The scipy float16 rejection, fixed at the source. Numerically inert: the
# adjacency is upcast before use and every entry is exactly 1.
FEATURIZER_OLD = """    # Convert to torch coo sparse tensor
    if len(adj_val) > 0:  # ensure tensor is not empty"""
FEATURIZER_NEW = """    # VENDORED CHANGE: modern scipy rejects float16. Numerically inert --
    # the adjacency is upcast before use.
    if np.dtype(dtype) == np.float16:
        dtype = np.float32

    # Convert to torch coo sparse tensor
    if len(adj_val) > 0:  # ensure tensor is not empty"""

TENSOR_PY = '''"""The three helpers graphium/features needs from graphium/utils/tensor.py."""

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
'''

PURE_SCATTER_PY = '''"""Native-torch replacement for the featurizer's single torch_scatter call."""

from __future__ import annotations

import torch


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def _out_shape(src, dim, dim_size, index):
    size = list(src.size())
    if dim_size is not None:
        size[dim] = dim_size
    elif index.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = int(index.max()) + 1
    return size


def scatter_add(src, index, dim=-1, out=None, dim_size=None):
    index = broadcast(index, src, dim)
    if out is None:
        out = src.new_zeros(_out_shape(src, dim, dim_size, index))
    return out.scatter_add_(dim, index, src)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    summed = scatter_add(src, index, dim, None, dim_size)
    count = scatter_add(torch.ones_like(src), index, dim, None, summed.size(dim))
    return summed / count.clamp(min=1)


def _reduce(src, index, dim, dim_size, reduce):
    index = broadcast(index, src, dim)
    out = src.new_zeros(_out_shape(src, dim, dim_size, index))
    return out.scatter_reduce_(dim, index, src, reduce=reduce, include_self=False)


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce in ("sum", "add"):
        return scatter_add(src, index, dim, out, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    aliases = {"max": "amax", "amax": "amax", "min": "amin",
               "amin": "amin", "mul": "prod", "prod": "prod"}
    if reduce not in aliases:
        raise ValueError(f"unsupported reduce: {reduce}")
    return _reduce(src, index, dim, dim_size, aliases[reduce])
'''


def rewrite(text: str, name: str) -> str:
    for pattern, replacement in IMPORT_REWRITES:
        text = re.sub(pattern, replacement, text)

    if name == "nmp.py":
        if NMP_OLD not in text:
            raise RuntimeError("nmp.py: periodic table loader not found; check version")
        text = text.replace(NMP_OLD, NMP_NEW)
        if not re.search(r"^import os$", text, re.M):
            text = text.replace("import importlib.resources", "import importlib.resources\nimport os", 1)

    if name == "featurizer.py":
        if FEATURIZER_OLD not in text:
            raise RuntimeError("featurizer.py: coo_matrix site not found; check version")
        text = text.replace(FEATURIZER_OLD, FEATURIZER_NEW, 1)

    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.source is None:
        print(
            "ERROR: graphium is not importable in this interpreter.\n"
            "       Run this with an env that has graphium 2.4.7 installed, "
            "or pass --source <site-packages/graphium>."
        )
        return 1

    features = args.source / "features"
    if not features.is_dir():
        print(f"ERROR: {features} does not exist")
        return 1

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    for name in MODULES:
        source_text = (features / name).read_text()
        (args.output / name).write_text(rewrite(source_text, name))
        print(f"  vendored {name}")

    for name in DATA_FILES:
        shutil.copy2(features / name, args.output / name)
        print(f"  copied   {name}")

    (args.output / "tensor.py").write_text(TENSOR_PY)
    (args.output / "pure_scatter.py").write_text(PURE_SCATTER_PY)
    print("  wrote    tensor.py, pure_scatter.py")

    # Docstrings mention graphium everywhere; only real imports matter.
    banned = re.compile(
        r"^\s*(?:from\s+(graphium|torch_scatter|torch_sparse|torch_cluster)\b"
        r"|import\s+(graphium|torch_scatter|torch_sparse|torch_cluster)\b)"
    )
    leftovers = []
    for path in sorted(args.output.glob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if banned.match(line):
                leftovers.append(f"{path.name}:{line_no}: {line.strip()[:90]}")

    if leftovers:
        print("\nWARNING: imports of graphium/compiled extensions remain:")
        for item in leftovers:
            print(f"  {item}")
        return 1

    print(f"\nvendored featurizer -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
