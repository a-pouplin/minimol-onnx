# minimol-onnx

[MiniMol](https://github.com/graphcore-research/minimol) molecular fingerprints, frozen into a portable ONNX graph. The goal is a straightforward install and a single call to get fingerprints, without the dependency conflicts of the original package.

Given the same featurized input, the fingerprints match the original MiniMol to within **6.4e-6** max absolute difference across 300 diverse molecules (1–190 atoms).


## Install

```bash
pip install minimol-onnx
```

The 32 MB ONNX graph ships inside the wheel — no download step, no cache directory, works
offline.

## Use

```python
from minimol_onnx import MinimolONNX

model = MinimolONNX()
fingerprints = model(["c1ccc(-c2cccnc2)cc1", "CCO"])   # (2, 512) float32
```

`model` accepts a single SMILES string or any iterable of them, and returns an
`(n, 512) float32` array. Batching is a performance knob only — results are identical
either way.

```python
MinimolONNX(
    model_path=...,             # override the bundled graph
    featurization_kwargs=...,   # dict, or path to JSON
    batch_size=100,
    providers=["CPUExecutionProvider"],   # any onnxruntime provider
)
```

For GPU inference, `pip install minimol-onnx[gpu]` and pass
`providers=["CUDAExecutionProvider"]`.

## Dependencies

`onnxruntime numpy scipy pandas networkx loguru rdkit datamol torch torch-geometric`,
verified against torch 2.13, numpy 2.4, scipy 1.17, pandas 3.0, onnxruntime 1.28.
`torch` and `torch-geometric` are pulled in by the featurizer, not by inference.
Torch runs entirely in onnxruntime but featurization still builds a PyG `Data` object, so the
install is ~2 GB. 


## Development

The ONNX export pipeline lives in `export/` and ships in the sdist. It needs an
environment with `minimol` 1.3.5 and `graphium` 2.4.7 installed; see `export/README.md`.

```bash
git clone https://github.com/a-pouplin/minimol-onnx
cd minimol-onnx
pip install -e ".[dev]"
pytest
```

## License and attribution

`Apache-2.0 AND MIT`. This package combines two upstream sources:

- The vendored featurizer in `minimol_onnx/minimol_feat/` derives from
  [graphium](https://github.com/datamol-io/graphium) 2.4.7 (**Apache-2.0**, Valence Labs /
  Recursion Pharmaceuticals / Graphcore), modified to drop the compiled extensions.
- The weights in `minimol_v1.onnx` are exported from
  [minimol](https://github.com/graphcore-research/minimol) 1.3.5 (**MIT**, Graphcore Ltd.)
  and are numerically unchanged.

See [NOTICE](NOTICE) for the full breakdown. If you use this, cite the original work:

```bibtex
@article{klaser2024minimol,
  title   = {{MiniMol}: A Parameter-Efficient Foundation Model for Molecular Learning},
  author  = {Kl{\"a}ser, Kerstin and Banaszewski, B{\l}a{\.z}ej and Maddrell-Mander, Samuel and
             McLean, Callum and M{\"u}ller, Luis and Parviz, Ali and Huang, Shenyang and
             Fitzgibbon, Andrew},
  journal = {arXiv preprint arXiv:2404.14986},
  year    = {2024},
  url     = {https://arxiv.org/abs/2404.14986}
}
```
