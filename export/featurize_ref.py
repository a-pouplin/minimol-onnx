"""Load MiniMol under the shims and featurize SMILES serially.

Serial because joblib worker processes do not inherit the shims.
"""

from __future__ import annotations

import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import graphium_shims  # noqa: F401  -- must precede any graphium import

graphium_shims.patch_featurizer_dtype()

import datamol as dm  # noqa: E402
import torch  # noqa: E402
from graphium.features.featurizer import mol_to_pyggraph  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from torch_geometric.nn import global_max_pool  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Featurizer keys the backbone consumes, in forward-argument order.
INPUT_KEYS = (
    "feat",
    "edge_feat",
    "edge_index",
    "laplacian_eigvec",
    "laplacian_eigval",
    "rw_return_probs",
)


def load_minimol(batch_size: int = 100, disable_readout_cache: bool = False):
    """Build a ``Minimol`` with the shims active, in eval mode.

    Keep the readout cache on for reference values; disable only for tracing.
    """
    from minimol import Minimol

    with open(os.devnull, "w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
        model = Minimol(batch_size=batch_size)

    # Serial featurization so Minimol.__call__ works under the shims too.
    model.datamodule.featurization_n_jobs = 0

    net = model.predictor.network
    if disable_readout_cache and hasattr(net, "_disable_readout_cache"):
        net._disable_readout_cache()
    # Proper .eval(): the Laplacian PE encoder sign-flips in training mode.
    net.eval()
    return model


def dump_featurization_kwargs(model, path: Path | None = None) -> dict:
    """Write the live featurization config to JSON for the standalone runtime."""
    path = path or ARTIFACTS / "featurization_kwargs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = model.datamodule.featurization
    path.write_text(json.dumps(kwargs, indent=2, sort_keys=True) + "\n")
    return kwargs


def to_fp32(data):
    """Mirror ``Minimol.to_fp32``: half -> float, int32 -> long."""
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            if value.dtype == torch.half:
                data[key] = value.float()
            elif value.dtype == torch.int32:
                data[key] = value.long()
    return data


def featurize(smiles, featurization_kwargs):
    """SMILES -> list of PyG ``Data``, with MiniMol's dtype casts.

    Pass strings, not mols: atom order changes the Laplacian basis (~0.1 shift).
    """
    out = []
    for smi in smiles:
        data = mol_to_pyggraph(
            smi, on_error="raise", mask_nan="raise", **featurization_kwargs
        )
        if isinstance(data, str):  # on_error="raise" should prevent this
            raise ValueError(f"featurization failed for {smi!r}: {data}")
        out.append(to_fp32(data))
    return out


def collate(datas):
    """Batch featurized molecules into the backbone's positional arguments."""
    batch = Batch.from_data_list(datas)
    tensors = [batch[key] for key in INPUT_KEYS]
    tensors[INPUT_KEYS.index("edge_index")] = batch.edge_index.long()
    graph_slots = torch.zeros(batch.num_graphs)
    return batch, (*tensors, batch.batch, graph_slots)


@torch.no_grad()
def reference_fingerprints(model, datas):
    """Exactly what ``Minimol.__call__`` computes for one batch."""
    batch = Batch.from_data_list(datas)
    node_features = model.predictor.get_fingerprints_for_batch(
        {"features": batch, "batch_indices": batch.batch}
    )
    return global_max_pool(node_features, batch.batch)


if __name__ == "__main__":
    print("shims active:", graphium_shims.using_shims())

    model = load_minimol(batch_size=8)
    kwargs = dump_featurization_kwargs(model)
    print("featurization keys:", sorted(kwargs))

    smiles = [
        "COc1ccc2cc(C(=O)NC3(C(=O)N[C@H](Cc4ccccc4)C(=O)NCC4CCN(CC5CCOCC5)CC4)CCCC3)sc2c1",
        "Nc1nc(=O)c2c([nH]1)NCC(CNc1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)N2C=O",
        "O=C1CCCN1CCCCN1CCN(c2cc(C(F)(F)F)ccn2)CC1",
        "c1ccc(-c2cccnc2)cc1",
    ]

    datas = featurize(smiles, kwargs)
    shapes = {k: tuple(datas[3][k].shape) for k in INPUT_KEYS}
    print("shapes (last molecule):", shapes)

    fps = reference_fingerprints(model, datas)
    print("fingerprints:", tuple(fps.shape))

    # End-to-end check through the unmodified public API.
    direct = model(smiles)
    stacked = torch.stack(direct)
    print("Minimol.__call__:", len(direct), tuple(direct[0].shape))
    print("max abs diff vs serial path:", (stacked - fps).abs().max().item())
