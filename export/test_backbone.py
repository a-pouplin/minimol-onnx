"""The wrapper must reproduce Minimol() exactly in eager mode, before tracing."""

from __future__ import annotations

import torch

import featurize_ref as FR
from backbone import MiniMolBackbone

# Deliberately awkward: single atom, fragments, charged, stereo, fused rings.
SMILES = [
    "COc1ccc2cc(C(=O)NC3(C(=O)N[C@H](Cc4ccccc4)C(=O)NCC4CCN(CC5CCOCC5)CC4)CCCC3)sc2c1",
    "Nc1nc(=O)c2c([nH]1)NCC(CNc1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)N2C=O",
    "O=C1CCCN1CCCCN1CCN(c2cc(C(F)(F)F)ccn2)CC1",
    "c1ccc(-c2cccnc2)cc1",
    "O",
    "CC.CC",
    "[NH3+]CC(=O)[O-]",
    "C/C=C/C",
]


def main() -> int:
    model = FR.load_minimol(batch_size=64)
    kwargs = model.datamodule.featurization
    backbone = MiniMolBackbone(model.predictor.network).eval()

    failures = []

    # --- 1. same batch, wrapper vs Minimol ---------------------------------
    datas = FR.featurize(SMILES, kwargs)
    reference = FR.reference_fingerprints(model, datas)
    _, args = FR.collate(datas)
    with torch.no_grad():
        got = backbone(*args)

    diff = (reference - got).abs().max().item()
    print(f"wrapper vs Minimol ({len(SMILES)} molecules): max abs diff {diff:.3e}")
    if diff != 0.0:
        failures.append(f"eager equivalence: expected 0.0, got {diff:.3e}")

    # --- 2. batch-size independence ----------------------------------------
    # Same molecule alone vs inside various batches. Target is not zero: stock
    # Minimol itself drifts ~1e-6 across batch compositions (float addition is
    # not associative), so assert the drift is noise and matches stock exactly.
    NOISE_TOL = 1e-5

    target = SMILES[3]
    compositions = [
        ("alone", [target], 0),
        ("head of 3", [target] + SMILES[:2], 0),
        ("tail of 8", SMILES[:7] + [target], 7),
        ("middle of 5", SMILES[:2] + [target] + SMILES[4:6], 2),
    ]

    rows, stock_rows = {}, {}
    for label, batch_smiles, position in compositions:
        _, batch_args = FR.collate(FR.featurize(batch_smiles, kwargs))
        with torch.no_grad():
            rows[label] = backbone(*batch_args)[position]
        stock_rows[label] = torch.stack(model(batch_smiles))[position]

    baseline, stock_baseline = rows["alone"], stock_rows["alone"]
    for label, _, _ in compositions:
        drift = (rows[label] - baseline).abs().max().item()
        stock_drift = (stock_rows[label] - stock_baseline).abs().max().item()
        agree = (rows[label] - stock_rows[label]).abs().max().item()
        print(
            f"batch-size independence [{label:12s}]: drift {drift:.3e} "
            f"(stock {stock_drift:.3e}), vs stock {agree:.3e}"
        )
        if drift > NOISE_TOL:
            failures.append(f"batch-size drift [{label}]: {drift:.3e} > {NOISE_TOL:.0e}")
        if agree != 0.0:
            failures.append(f"differs from stock Minimol [{label}]: {agree:.3e}")

    # --- 3. output shape tracks the graph count ----------------------------
    for n in (1, 3, 8):
        _, batch_args = FR.collate(FR.featurize(SMILES[:n], kwargs))
        with torch.no_grad():
            shape = tuple(backbone(*batch_args).shape)
        print(f"shape for {n} molecule(s): {shape}")
        if shape != (n, 512):
            failures.append(f"shape for n={n}: expected ({n}, 512), got {shape}")

    print()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: wrapper is exact and batch-size independent in eager mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
