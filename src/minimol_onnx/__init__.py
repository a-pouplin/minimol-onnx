"""MiniMol molecular fingerprints from SMILES, as a self-contained ONNX model.

    from minimol_onnx import MinimolONNX

    model = MinimolONNX()
    fingerprints = model(["c1ccc(-c2cccnc2)cc1", "CCO"])   # (2, 512) float32
"""

from .runtime import FINGERPRINT_DIM, MinimolONNX

__all__ = ["MinimolONNX", "FINGERPRINT_DIM"]
__version__ = "0.1.1"
