"""Checks that fail loudly when the wheel is built wrong."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BLOCKED = ("torch_scatter", "torch_sparse", "torch_cluster", "graphium")

# Pinned: a git-lfs pointer checkout would be ~133 bytes and fail opaquely.
ONNX_BYTES = 33_128_988


@pytest.fixture(scope="session")
def package_dir() -> Path:
    import minimol_onnx

    return Path(minimol_onnx.__file__).resolve().parent


def test_import_is_not_the_source_tree(package_dir):
    """Tests must exercise the installed package, not the source tree."""
    assert "src" not in package_dir.parts, (
        f"imported minimol_onnx from the source tree ({package_dir}); "
        "install the package before running the suite"
    )


@pytest.mark.parametrize("name", BLOCKED)
def test_compiled_extensions_stay_blocked(name):
    with pytest.raises(ImportError):
        __import__(name)


def test_version_is_exposed():
    import minimol_onnx

    assert minimol_onnx.__version__.count(".") >= 2


def test_public_api():
    import minimol_onnx

    assert minimol_onnx.MinimolONNX
    assert minimol_onnx.FINGERPRINT_DIM == 512


def test_onnx_weights_shipped_intact(package_dir):
    onnx = package_dir / "minimol_v1.onnx"
    assert onnx.is_file(), "minimol_v1.onnx missing from the installed package"
    assert onnx.stat().st_size == ONNX_BYTES, (
        f"minimol_v1.onnx is {onnx.stat().st_size} bytes, expected {ONNX_BYTES} "
        "-- a git-lfs pointer or a truncated copy?"
    )


def test_data_files_shipped(package_dir):
    assert (package_dir / "featurization_kwargs.json").is_file()
    assert (package_dir / "minimol_feat" / "periodic_table.csv").is_file()


def test_featurizer_is_not_top_level():
    """``minimol_feat`` must not be a top-level module in site-packages."""
    assert "minimol_feat" not in sys.modules or sys.modules["minimol_feat"] is None
    with pytest.raises(ImportError):
        __import__("minimol_feat")


def test_constructs_with_no_arguments():
    """Catches a wheel missing its data files."""
    from minimol_onnx import MinimolONNX

    assert MinimolONNX() is not None
