# Export pipeline

How `minimol_v1.onnx` and `src/minimol_onnx/minimol_feat/` were produced. Nothing here
ships in the wheel — it is included in the sdist so the result is reproducible from
source.

## Prerequisites

An environment with `minimol` 1.3.5 and `graphium` 2.4.7 installed. That stack is
version-locked and does not coexist with current torch — which is the reason this package
exists. Create it separately from the one you install `minimol-onnx` into.

`graphium_shims.py` replaces `torch_scatter`, `torch_sparse` and `torch_cluster` with
pure-torch equivalents. This is what makes graphium importable at all when its compiled
extensions are built against a different torch (`OSError: Could not load this library:
.../_version_cpu.so`), and it is required for the export regardless: torch_scatter's
custom C++ ops do not survive tracing.

## Order

```bash
python export_onnx.py        # trace the backbone -> minimol_v1.onnx
python test_backbone.py      # exported graph vs. the live model
python validate.py --n 300   # end-to-end, writes tests/data/reference_*
python vendor.py             # copy + rewrite graphium.features -> minimol_feat/
```

`vendor.py` finds graphium via `importlib.util.find_spec` in whatever interpreter runs it;
pass `--source <site-packages/graphium>` to override. It overwrites its output directory,
so re-running is safe.

Then, from a clean environment that has none of the above:

```bash
pip install -e ".[dev]"
pytest
```

## Files

| File | What |
|---|---|
| `export_onnx.py` | Traces the backbone and writes the ONNX graph |
| `backbone.py` | The `encoder → pre_nn → pre_nn_edges → gnn → max-pool` chain, as an `nn.Module` |
| `graphium_shims.py` | Pure-torch stand-ins for the three compiled extensions |
| `test_backbone.py` | Exported graph vs. live model, same inputs |
| `featurize_ref.py` | Featurizes the reference SMILES with stock graphium |
| `validate.py` | Full-stack comparison; writes the reference fingerprints |
| `vendor.py` | Extracts and rewrites `graphium.features` into `minimol_feat/` |
