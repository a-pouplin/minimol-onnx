"""Native-torch replacement for the featurizer's single torch_scatter call."""

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
