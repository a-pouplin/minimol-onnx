"""MiniMol fingerprints from SMILES, with no graphium and no compiled extensions.

    from minimol_onnx import MinimolONNX

    model = MinimolONNX()
    fingerprints = model(["c1ccc(-c2cccnc2)cc1", "CCO"])   # (2, 512) float32

Matches stock MiniMol to ~6e-6 max absolute difference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnxruntime as ort
import torch

from .minimol_feat.featurizer import mol_to_pyggraph

HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = HERE / "minimol_v1.onnx"
DEFAULT_KWARGS = HERE / "featurization_kwargs.json"

# Order = the ONNX graph's input signature.
INPUT_NAMES = (
    "feat",
    "edge_feat",
    "edge_index",
    "laplacian_eigvec",
    "laplacian_eigval",
    "rw_return_probs",
    "batch",
    "graph_slots",
)
FEATURE_KEYS = INPUT_NAMES[:6]

FINGERPRINT_DIM = 512


class MinimolONNX:
    """SMILES -> 512-dimensional fingerprints."""

    def __init__(
        self,
        model_path: Path | str = DEFAULT_MODEL,
        featurization_kwargs: dict | Path | str = DEFAULT_KWARGS,
        batch_size: int = 100,
        providers: Sequence[str] | None = None,
    ):
        if not isinstance(featurization_kwargs, dict):
            featurization_kwargs = json.loads(Path(featurization_kwargs).read_text())
        self.featurization_kwargs = featurization_kwargs
        self.batch_size = batch_size
        self.session = ort.InferenceSession(
            str(model_path), providers=list(providers or ["CPUExecutionProvider"])
        )

    # -- featurization ----------------------------------------------------
    def featurize_one(self, smiles: str):
        """One SMILES -> a PyG ``Data``. Pass the string, not a pre-built mol:
        atom order changes the Laplacian basis and shifts the fingerprint by ~0.1."""
        data = mol_to_pyggraph(
            smiles, on_error="raise", mask_nan="raise", **self.featurization_kwargs
        )
        if isinstance(data, str):
            raise ValueError(f"featurization failed for {smiles!r}: {data}")

        # Mirror Minimol.to_fp32: half -> float, int32 -> long.
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if value.dtype == torch.half:
                    data[key] = value.float()
                elif value.dtype == torch.int32:
                    data[key] = value.long()
        return data

    # -- collation --------------------------------------------------------
    @staticmethod
    def collate(datas: Sequence) -> dict[str, np.ndarray]:
        """Concatenate featurized molecules into one disconnected graph.
        No padding: all three ONNX axes are dynamic."""
        feeds: dict[str, list] = {key: [] for key in FEATURE_KEYS}
        batch_index, node_offset = [], 0

        for graph_index, data in enumerate(datas):
            num_nodes = int(data.num_nodes)
            for key in FEATURE_KEYS:
                if key == "edge_index":
                    continue
                feeds[key].append(data[key].numpy())
            edge_index = data["edge_index"].numpy()
            if edge_index.size:
                feeds["edge_index"].append(edge_index.astype(np.int64) + node_offset)
            batch_index.append(np.full(num_nodes, graph_index, dtype=np.int64))
            node_offset += num_nodes

        out = {}
        for key in FEATURE_KEYS:
            if key == "edge_index":
                continue
            out[key] = np.concatenate(feeds[key]).astype(np.float32)

        if feeds["edge_index"]:
            out["edge_index"] = np.concatenate(feeds["edge_index"], axis=1)
        else:  # every molecule in the batch was a single atom
            out["edge_index"] = np.zeros((2, 0), dtype=np.int64)

        out["batch"] = np.concatenate(batch_index)
        # Only the length is read (num_graphs); values are unused.
        out["graph_slots"] = np.zeros(len(datas), dtype=np.float32)
        return out

    # -- inference --------------------------------------------------------
    def __call__(self, smiles: str | Iterable[str]) -> np.ndarray:
        """Return ``(len(smiles), 512)`` float32 fingerprints."""
        if isinstance(smiles, str):
            smiles = [smiles]
        smiles = list(smiles)
        if not smiles:
            return np.zeros((0, FINGERPRINT_DIM), dtype=np.float32)

        results = []
        for start in range(0, len(smiles), self.batch_size):
            chunk = smiles[start : start + self.batch_size]
            feeds = self.collate([self.featurize_one(s) for s in chunk])
            results.append(self.session.run(None, feeds)[0])
        return np.concatenate(results)


if __name__ == "__main__":
    model = MinimolONNX()
    demo = [
        "COc1ccc2cc(C(=O)NC3(C(=O)N[C@H](Cc4ccccc4)C(=O)NCC4CCN(CC5CCOCC5)CC4)CCCC3)sc2c1",
        "Nc1nc(=O)c2c([nH]1)NCC(CNc1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)N2C=O",
        "O=C1CCCN1CCCCN1CCN(c2cc(C(F)(F)F)ccn2)CC1",
        "c1ccc(-c2cccnc2)cc1",
    ]
    fingerprints = model(demo)
    print(f"{len(demo)} SMILES -> {fingerprints.shape} {fingerprints.dtype}")
    print(fingerprints[0][:6])
