"""Numerical equivalence with stock MiniMol, with graphium blocked (see conftest).

The reference in ``tests/data/`` was produced by the real minimol + graphium
stack over 300 diverse molecules (1-190 atoms).
"""

from __future__ import annotations

import numpy as np
import pytest

# Observed max is 6.4e-6; set 10x above to absorb BLAS/platform variation.
TOLERANCE = 5e-5

DEMO = ["c1ccc(-c2cccnc2)cc1", "CCO"]


def test_shape_and_dtype(model):
    got = model(DEMO)
    assert got.shape == (2, 512)
    assert got.dtype == np.float32


def test_single_string_is_accepted(model):
    got = model("CCO")
    assert got.shape == (1, 512)
    assert np.allclose(got, model(["CCO"]))


def test_empty_input(model):
    got = model([])
    assert got.shape == (0, 512)


def test_single_atom_molecule(model):
    """No edges: exercises the empty edge_index branch in collate."""
    got = model(["C"])
    assert got.shape == (1, 512)
    assert np.isfinite(got).all()


def test_invalid_smiles_raises(model):
    with pytest.raises(Exception):
        model(["not_a_molecule)))"])


def test_exported_graph_matches_reference(model, reference, reference_inputs):
    """The exported graph reproduces stock MiniMol, from stored featurized inputs.

    This is the strict test of what the package actually exports. Feeding stored
    tensors keeps the featurizer -- and its platform-dependent eigenvectors --
    out of the comparison, so the bound holds on every platform.
    """
    _, expected = reference
    got = np.concatenate([model.session.run(None, feeds)[0] for feeds in reference_inputs])

    assert got.shape == expected.shape
    worst = np.abs(expected - got).max()
    assert worst < TOLERANCE, f"max abs diff {worst:.3e} exceeds {TOLERANCE:.0e}"


def test_matches_reference(model, reference):
    """End to end from SMILES, with a tolerance for LAPACK's eigenvector signs.

    Laplacian positional encodings are eigenvectors, and ``v`` and ``-v`` are
    equally valid: the sign a solver returns depends on the LAPACK build, so
    macOS (Accelerate) and Linux (OpenBLAS) disagree on most molecules by up to
    ~0.1 in a few dimensions. Stock MiniMol does the same -- it is trained with
    random sign flipping precisely because the sign is arbitrary -- so this
    cannot be asserted bit-exactly across platforms. Direction is preserved,
    which is what the fingerprint is used for; the strict numerical check lives
    in test_exported_graph_matches_reference above.
    """
    smiles, expected = reference
    got = model(smiles)

    assert got.shape == expected.shape

    cosine = (expected * got).sum(1) / (
        np.linalg.norm(expected, axis=1) * np.linalg.norm(got, axis=1)
    )
    per_molecule = np.abs(expected - got).max(axis=1)
    worst = int(np.argmin(cosine))

    assert cosine.min() > 0.99, (
        f"cosine similarity {cosine.min():.4f} on {smiles[worst]}; "
        f"max abs diff {per_molecule.max():.3e}, median {np.median(per_molecule):.3e}"
    )
    assert per_molecule.max() < 1.0, (
        f"max abs diff {per_molecule.max():.3e} is larger than an eigenvector sign "
        f"flip explains: {smiles[int(np.argmax(per_molecule))]}"
    )


def test_batching_does_not_change_results(model, reference):
    """Batch size is a performance knob, not a numerical one."""
    smiles = reference[0][:8]
    batched = model(smiles)
    one_at_a_time = np.concatenate([model([s]) for s in smiles])
    assert np.abs(batched - one_at_a_time).max() < TOLERANCE


def test_batch_size_boundary(model, reference):
    """More molecules than batch_size exercises the chunking loop."""
    from minimol_onnx import MinimolONNX

    smiles = reference[0][:10]
    small = MinimolONNX(batch_size=3)
    assert np.abs(small(smiles) - model(smiles)).max() < TOLERANCE
