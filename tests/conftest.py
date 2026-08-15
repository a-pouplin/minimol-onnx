"""Block graphium and the compiled extensions before anything imports them.

``None`` in ``sys.modules`` makes ``import x`` raise ImportError, so a passing
suite proves the extensions are genuinely unused, not merely absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

BLOCKED = ("torch_scatter", "torch_sparse", "torch_cluster", "graphium")

for _name in BLOCKED:
    sys.modules[_name] = None  # type: ignore[assignment]

DATA = Path(__file__).resolve().parent / "data"

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def model():
    """One session-wide model; loading the 32 MB graph is not free."""
    from minimol_onnx import MinimolONNX

    return MinimolONNX(batch_size=32)


@pytest.fixture(scope="session")
def reference():
    """(smiles, fingerprints) captured from stock MiniMol + graphium."""
    import numpy as np

    smiles = (DATA / "reference_smiles.txt").read_text().split()
    fingerprints = np.load(DATA / "reference_fingerprints.npy")
    assert len(smiles) == fingerprints.shape[0]
    return smiles, fingerprints


@pytest.fixture(scope="session")
def reference_inputs():
    """The same molecules already featurized, as a list of ONNX feed dicts.

    Lets the exported graph be tested without running the featurizer, whose
    eigenvector signs are LAPACK-dependent (see test_matches_reference).
    """
    import numpy as np

    stored = np.load(DATA / "reference_inputs.npz")
    chunks = sorted({int(key.split("_", 1)[0]) for key in stored.files})
    return [
        {
            key.split("_", 1)[1]: stored[key]
            for key in stored.files
            if int(key.split("_", 1)[0]) == chunk
        }
        for chunk in chunks
    ]
