"""Vendored subset of ``graphium.features`` (graphium 2.4.7).

MODIFIED from the original. Licensed under the Apache License, Version 2.0.

    Copyright 2023 Valence Labs
    Copyright 2023 Recursion Pharmaceuticals
    Copyright 2023 Graphcore Limited
    https://github.com/datamol-io/graphium

Changes made by this project (Apache-2.0 section 4(b)):

  * Removed the dependency on the compiled extensions ``torch-scatter``,
    ``torch-sparse`` and ``torch-cluster``. The one scatter call on the
    featurization path (``scatter_add`` in ``rw.py``) now goes to
    ``pure_scatter.py``, a pure-PyTorch module added here.
  * Dropped feature paths not reached by MiniMol v1 featurization.
  * Rewrote imports to be package-relative.

Numerical output is unchanged. See the top-level NOTICE file.
"""

from .featurizer import get_mol_atomic_features_onehot
from .featurizer import get_mol_atomic_features_float
from .featurizer import get_mol_edge_features
from .featurizer import mol_to_adj_and_features
from .featurizer import mol_to_graph_dict
from .featurizer import mol_to_graph_signature
from .featurizer import GraphDict
from .featurizer import mol_to_pyggraph
from .featurizer import to_dense_array
