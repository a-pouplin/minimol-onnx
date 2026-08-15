"""Compare the exported ONNX model against stock MiniMol at scale.

Reports the error distribution, checks it does not grow with molecule size,
and writes the reference fingerprints to ``artifacts/``.

Usage::

    python validate.py [--n 300] [--batch-size 32]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import featurize_ref as FR
from export_onnx import DEFAULT_OUTPUT, INPUT_NAMES

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Cases that exercise specific code paths.
EDGE_CASES = [
    "O",  # single atom, zero edges
    "[Na+]",  # single charged atom
    "C",  # methane
    "CC.CC",  # disconnected -> the disconnected_comp branch
    "CCO.CCO.CCO",  # three components
    "[NH3+]CC(=O)[O-]",  # zwitterion
    "C/C=C/C",  # E stereo
    "C/C=C\\C",  # Z stereo
    "N[C@@H](C)C(=O)O",  # tetrahedral stereo
    "N[C@H](C)C(=O)O",  # the other enantiomer
    "c1ccccc1",  # simplest aromatic
    "C1CCCCCCCCCCC1",  # macrocycle
    "C1CCCCCCCCCCCCCCC1",  # bigger macrocycle
    "CCCCCCCCCCCCCCCCCC(=O)O",  # long flexible chain
    "FC(F)(F)c1ccccc1",  # halogens
    "ClC(Br)I",  # heavy halogens
    "O=S(=O)(O)c1ccccc1",  # sulfur oxidation state
    "[2H]C([2H])([2H])O",  # isotopes
    "c1ccc2c(c1)ccc1ccccc12",  # fused aromatics
    "C1=CC2=CC=CC3=C2C(=C1)C=C3",  # more fused
]


def build_smiles(n: int) -> list[str]:
    """Edge cases first, then a deterministic spread of drug molecules."""
    import datamol as dm

    smiles = list(EDGE_CASES)
    seen = set(smiles)

    for loader in (dm.data.chembl_drugs, dm.data.solubility):
        if len(smiles) >= n:
            break
        pool = sorted(set(loader()["smiles"].dropna().tolist()))
        # Even stride: deterministic, no RNG, spread across the pool.
        need = n - len(smiles)
        stride = max(1, len(pool) // max(need, 1))
        for smi in pool[::stride]:
            if len(smiles) >= n:
                break
            if smi not in seen:
                seen.add(smi)
                smiles.append(smi)

    return smiles[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    model = FR.load_minimol(batch_size=args.batch_size)
    kwargs = model.datamodule.featurization
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    smiles = build_smiles(args.n)
    print(f"validating {len(smiles)} molecules, batch size {args.batch_size}")

    kept, refs, gots, sizes = [], [], [], []
    skipped = []
    stored_feeds = {}

    for start in range(0, len(smiles), args.batch_size):
        chunk = smiles[start : start + args.batch_size]

        try:
            datas = FR.featurize(chunk, kwargs)
        except Exception:
            # Isolate the offender, keep the rest of the batch.
            datas, chunk_ok = [], []
            for smi in chunk:
                try:
                    datas.append(FR.featurize([smi], kwargs)[0])
                    chunk_ok.append(smi)
                except Exception as exc:
                    skipped.append((smi, f"{type(exc).__name__}: {exc}"[:120]))
            chunk = chunk_ok
            if not chunk:
                continue

        reference = FR.reference_fingerprints(model, datas).numpy()
        _, tensors = FR.collate(datas)
        feeds = {name: t.numpy() for name, t in zip(INPUT_NAMES, tensors)}
        got = session.run(None, feeds)[0]
        # Kept as a test fixture: lets the graph be checked without the
        # featurizer, whose eigenvector signs are LAPACK-dependent.
        for name, array in feeds.items():
            stored_feeds[f"{len(refs)}_{name}"] = array

        kept.extend(chunk)
        refs.append(reference)
        gots.append(got)
        sizes.extend(int(d.num_nodes) for d in datas)

    reference = np.concatenate(refs)
    got = np.concatenate(gots)
    per_mol = np.abs(reference - got).max(axis=1)
    sizes = np.asarray(sizes)

    print()
    print(f"compared      : {len(kept)} molecules ({len(skipped)} skipped)")
    print(f"atoms         : min {sizes.min()}, median {int(np.median(sizes))}, max {sizes.max()}")
    print(f"max abs diff  : {per_mol.max():.3e}")
    print(f"median        : {np.median(per_mol):.3e}")
    print(f"p99           : {np.percentile(per_mol, 99):.3e}")
    print(f"mean          : {per_mol.mean():.3e}")

    # Float noise is uncorrelated with molecule size; structural error is not.
    if len(kept) > 2 and sizes.std() > 0 and per_mol.std() > 0:
        corr = float(np.corrcoef(sizes, per_mol)[0, 1])
        print(f"corr(atoms, error): {corr:+.3f}")
    else:
        corr = 0.0

    worst = np.argsort(per_mol)[-5:][::-1]
    print("\nworst 5:")
    for i in worst:
        print(f"  {per_mol[i]:.3e}  {sizes[i]:3d} atoms  {kept[i][:70]}")

    if skipped:
        print(f"\nskipped {len(skipped)}:")
        for smi, reason in skipped[:10]:
            print(f"  {smi[:60]} -- {reason}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS / "reference_fingerprints.npy", reference)
    (ARTIFACTS / "reference_smiles.txt").write_text("\n".join(kept) + "\n")
    np.savez_compressed(ARTIFACTS / "reference_inputs.npz", **stored_feeds)
    print(f"\nwrote {ARTIFACTS/'reference_fingerprints.npy'} {reference.shape}")
    print(f"wrote {ARTIFACTS/'reference_smiles.txt'}")
    print(f"wrote {ARTIFACTS/'reference_inputs.npz'}")

    failures = []
    if per_mol.max() > args.tolerance:
        failures.append(f"max abs diff {per_mol.max():.3e} > {args.tolerance:.0e}")
    if abs(corr) > 0.5 and per_mol.max() > 1e-5:
        failures.append(f"error correlates with molecule size (r={corr:+.3f})")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: ONNX matches MiniMol within {args.tolerance:.0e} across {len(kept)} molecules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
