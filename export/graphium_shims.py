"""Make graphium importable and ONNX-exportable without compiled extensions.

Import this module before anything imports ``graphium``. It installs pure-torch
stand-ins for ``torch_scatter`` / ``torch_sparse`` / ``torch_cluster``
"""

from __future__ import annotations

import sys
import types

import numpy as np
import torch

# --------------------------------------------------------------------------
# 1. pure-torch scatter
# --------------------------------------------------------------------------

# Modules that copy scatter refs at import time; rebound as a backstop.
_SCATTER_IMPORTERS = {
    "graphium.nn.pyg_layers.pooling_pyg": ("scatter",),
    "graphium.nn.pyg_layers.gated_gcn_pyg": ("scatter",),
    "graphium.nn.pyg_layers.pna_pyg": ("scatter",),
    "graphium.nn.encoders.signnet_pos_encoder": ("scatter",),
    "graphium.ipu.to_dense_batch": ("scatter_add",),
    "graphium.features.rw": ("scatter_add",),
}

_REDUCE_ALIASES = {
    "max": "amax",
    "amax": "amax",
    "min": "amin",
    "amin": "amin",
    "mul": "prod",
    "prod": "prod",
}


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Expand ``src`` so it lines up with ``other`` along ``dim``."""
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


def _reduce(src, index, dim, dim_size, reduce):
    index = broadcast(index, src, dim)
    out = src.new_zeros(_out_shape(src, dim, dim_size, index))
    return out.scatter_reduce_(dim, index, src, reduce=reduce, include_self=False)


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    summed = scatter_add(src, index, dim, None, dim_size)
    count = scatter_add(torch.ones_like(src), index, dim, None, summed.size(dim))
    return summed / count.clamp(min=1)


def scatter(src, index, dim=-1, out=None, dim_size=None, reduce="sum"):
    if reduce in ("sum", "add"):
        return scatter_add(src, index, dim, out, dim_size)
    if reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    try:
        aten_reduce = _REDUCE_ALIASES[reduce]
    except KeyError:
        raise ValueError(f"unsupported reduce: {reduce}") from None
    return _reduce(src, index, dim, dim_size, aten_reduce)


# torch_scatter returns (values, argindex); graphium ignores the argindex.
def scatter_max(src, index, dim=-1, out=None, dim_size=None):
    return _reduce(src, index, dim, dim_size, "amax"), None


def scatter_min(src, index, dim=-1, out=None, dim_size=None):
    return _reduce(src, index, dim, dim_size, "amin"), None


class SparseTensor:  # pragma: no cover - only ever reached by isinstance()
    """Stand-in for ``torch_sparse.SparseTensor``; graphium only isinstance()s it."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "torch_sparse.SparseTensor is stubbed out; pass a dense edge_index."
        )


def _install_fake_modules() -> None:
    scatter_mod = types.ModuleType("torch_scatter")
    scatter_mod.scatter = scatter
    scatter_mod.scatter_add = scatter_add
    scatter_mod.scatter_mean = scatter_mean
    scatter_mod.scatter_max = scatter_max
    scatter_mod.scatter_min = scatter_min
    scatter_mod.scatter_mul = lambda s, i, dim=-1, out=None, dim_size=None: scatter(
        s, i, dim, out, dim_size, "mul"
    )
    scatter_mod.composite = types.ModuleType("torch_scatter.composite")
    scatter_mod.__version__ = "0.0.0+graphium_shim"

    sparse_mod = types.ModuleType("torch_sparse")
    sparse_mod.SparseTensor = SparseTensor
    sparse_mod.__version__ = "0.0.0+graphium_shim"

    cluster_mod = types.ModuleType("torch_cluster")
    cluster_mod.__version__ = "0.0.0+graphium_shim"

    sys.modules["torch_scatter"] = scatter_mod
    sys.modules["torch_scatter.composite"] = scatter_mod.composite
    sys.modules["torch_sparse"] = sparse_mod
    sys.modules["torch_cluster"] = cluster_mod

    # Backstop in case graphium was somehow imported before us.
    for mod_name, names in _SCATTER_IMPORTERS.items():
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for name in names:
            setattr(mod, name, getattr(scatter_mod, name))


# --------------------------------------------------------------------------
# 2. scipy float16 fix
# --------------------------------------------------------------------------

_scipy_patched = False


def patch_featurizer_dtype() -> None:
    """Force the adjacency ``coo_matrix`` to float32. Idempotent."""
    global _scipy_patched
    if _scipy_patched:
        return

    import graphium.features.featurizer as featurizer

    original = featurizer.mol_to_adjacency_matrix

    def mol_to_adjacency_matrix(
        mol, use_bonds_weights=False, add_self_loop=False, dtype=np.float32
    ):
        if np.dtype(dtype) == np.float16:
            dtype = np.float32
        return original(mol, use_bonds_weights, add_self_loop, dtype)

    featurizer.mol_to_adjacency_matrix = mol_to_adjacency_matrix
    _scipy_patched = True


# --------------------------------------------------------------------------
# 3. export blockers
# --------------------------------------------------------------------------

_export_patched = False


def patch_export_blockers() -> None:
    """Drop ``FCLayer.forward``'s empty-input guard, which ``torch.export``
    cannot trace. Dead for our inputs; verified exact (0.0) against MiniMol."""
    global _export_patched
    if _export_patched:
        return

    import graphium.nn.base_layers as base_layers

    def fc_forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.linear(h)
        if self.normalization is not None:
            if h.shape[1] != self.out_dim:
                h = self.normalization(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.normalization(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        return h

    base_layers.FCLayer.forward = fc_forward
    _export_patched = True


# --------------------------------------------------------------------------

_install_fake_modules()


def using_shims() -> dict:
    """Report which stand-ins are active."""
    return {
        name: getattr(sys.modules[name], "__version__", "?")
        for name in ("torch_scatter", "torch_sparse", "torch_cluster")
        if name in sys.modules
    }
