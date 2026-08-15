"""Trace the backbone and write ``minimol_v1.onnx``, all axes dynamic.

Usage::

    python export_onnx.py [--output ../minimol_onnx/minimol_v1.onnx]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

import featurize_ref as FR
import graphium_shims
from backbone import MiniMolBackbone

INPUT_NAMES = [
    "feat",
    "edge_feat",
    "edge_index",
    "laplacian_eigvec",
    "laplacian_eigval",
    "rw_return_probs",
    "batch",
    "graph_slots",
]
OUTPUT_NAMES = ["fingerprint"]

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "minimol_onnx" / "minimol_v1.onnx"

# Mixed batch: fused rings, single atom, disconnected fragments, zwitterion, stereo.
TRACE_SMILES = [
    "COc1ccc2cc(C(=O)NC3(C(=O)N[C@H](Cc4ccccc4)C(=O)NCC4CCN(CC5CCOCC5)CC4)CCCC3)sc2c1",
    "Nc1nc(=O)c2c([nH]1)NCC(CNc1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1)N2C=O",
    "O=C1CCCN1CCCCN1CCN(c2cc(C(F)(F)F)ccn2)CC1",
    "c1ccc(-c2cccnc2)cc1",
    "O",
    "CC.CC",
    "[NH3+]CC(=O)[O-]",
    "C/C=C/C",
]

# Different shape in all three dimensions, to check nothing was baked in.
CHECK_SMILES = ["CCO", "c1ccccc1", "N[C@@H](C)C(=O)O", "OCCOCCO", "C1CCCCC1"]


def dynamic_shapes():
    """One shared Dim per axis."""
    Dim = torch.export.Dim
    num_nodes = Dim("num_nodes", min=2)
    num_edges = Dim("num_edges", min=2)
    num_graphs = Dim("num_graphs", min=2)
    return (
        {0: num_nodes},  # feat
        {0: num_edges},  # edge_feat
        {1: num_edges},  # edge_index
        {0: num_nodes},  # laplacian_eigvec
        {0: num_nodes},  # laplacian_eigval
        {0: num_nodes},  # rw_return_probs
        {0: num_nodes},  # batch
        {0: num_graphs},  # graph_slots
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=18)
    args_ns = parser.parse_args()

    graphium_shims.patch_export_blockers()

    print("loading MiniMol...")
    model = FR.load_minimol(batch_size=64)
    kwargs = model.datamodule.featurization
    network = model.predictor.network
    backbone = MiniMolBackbone(network).eval()

    # Compute reference values before the readout cache is disabled.
    trace_datas = FR.featurize(TRACE_SMILES, kwargs)
    reference = FR.reference_fingerprints(model, trace_datas)
    _, trace_args = FR.collate(trace_datas)

    check_datas = FR.featurize(CHECK_SMILES, kwargs)
    check_reference = FR.reference_fingerprints(model, check_datas)
    _, check_args = FR.collate(check_datas)

    with torch.no_grad():
        eager = backbone(*trace_args)
    eager_diff = (reference - eager).abs().max().item()
    print(f"eager wrapper vs Minimol: {eager_diff:.3e}")
    if eager_diff != 0.0:
        print("ABORT: wrapper is not exact in eager mode; fix that before tracing")
        return 1

    print(
        f"tracing with nodes={trace_args[0].shape[0]} "
        f"edges={trace_args[2].shape[1]} graphs={trace_args[7].shape[0]}"
    )

    # The exported graph never reads the readout cache.
    if hasattr(network, "_disable_readout_cache"):
        network._disable_readout_cache()

    exported = torch.export.export(backbone, trace_args, dynamic_shapes=dynamic_shapes())
    print("torch.export: OK (num_nodes / num_edges / num_graphs all symbolic)")

    # Check the exported program generalizes before writing ONNX.
    with torch.no_grad():
        replayed = exported.module()(*check_args)
    print(
        f"exported program on a different batch "
        f"({len(CHECK_SMILES)} graphs, {check_args[0].shape[0]} nodes, "
        f"{check_args[2].shape[1]} edges): shape {tuple(replayed.shape)}, "
        f"max abs diff {(check_reference - replayed).abs().max().item():.3e}"
    )

    args_ns.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        backbone,
        trace_args,
        str(args_ns.output),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_shapes=dynamic_shapes(),
        opset_version=args_ns.opset,
        dynamo=True,
        external_data=False,
    )
    size_mb = args_ns.output.stat().st_size / 1e6
    print(f"wrote {args_ns.output} ({size_mb:.1f} MB, opset {args_ns.opset})")

    import onnx

    onnx_model = onnx.load(str(args_ns.output))
    onnx.checker.check_model(onnx_model, full_check=True)
    print("onnx.checker: OK")

    # A surviving torch_scatter custom op would show up as an unknown domain.
    domains = {node.domain for node in onnx_model.graph.node}
    print("op domains:", domains or {"(default)"})
    if any(d not in ("", "ai.onnx", "ai.onnx.ml") for d in domains):
        print("ABORT: non-standard op domain in graph")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
