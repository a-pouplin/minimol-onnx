"""The exportable slice of MiniMol: featurized graph in, 512-d fingerprint out.

Skips MiniMol's Fingerprinter: layer 15 is the last gnn layer, so its cached
tensor equals ``gnn(g)["feat"]`` (verified at 0.0).
"""

from __future__ import annotations

import torch
from torch_geometric.data import Batch


class MiniMolBackbone(torch.nn.Module):
    """``FullGraphMultiTaskNetwork`` minus the task heads, plus max pooling.

    Takes flat tensors so it can be traced. The dummy ``graph_slots`` tensor
    makes ``num_graphs`` a shape -- a Python int would be baked in at trace time.
    """

    def __init__(self, network):
        super().__init__()
        self.encoder_manager = network.encoder_manager
        self.pre_nn = network.pre_nn
        self.pre_nn_edges = network.pre_nn_edges
        self.gnn = network.gnn

    def forward(
        self,
        feat: torch.Tensor,  # (num_nodes, 85)
        edge_feat: torch.Tensor,  # (num_edges, 13)
        edge_index: torch.Tensor,  # (2, num_edges) int64
        laplacian_eigvec: torch.Tensor,  # (num_nodes, 8)
        laplacian_eigval: torch.Tensor,  # (num_nodes, 8)
        rw_return_probs: torch.Tensor,  # (num_nodes, 16)
        batch: torch.Tensor,  # (num_nodes,) int64, node -> graph
        graph_slots: torch.Tensor,  # (num_graphs,) values unused
    ) -> torch.Tensor:  # (num_graphs, 512)
        num_graphs = graph_slots.shape[0]

        g = Batch(
            feat=feat,
            edge_feat=edge_feat,
            edge_index=edge_index,
            laplacian_eigvec=laplacian_eigvec,
            laplacian_eigval=laplacian_eigval,
            rw_return_probs=rw_return_probs,
            batch=batch,
        )
        g.num_nodes = feat.shape[0]
        # `num_graphs` is a read-only @property; `_num_graphs` is the field it reads.
        g._num_graphs = num_graphs

        g = self.encoder_manager(g)
        g["feat"] = self.pre_nn(g["feat"])
        g["edge_feat"] = self.pre_nn_edges(g["edge_feat"])
        node_feat = self.gnn(g)["feat"]

        # global_max_pool, written so it exports (scatter with amax, opset 16+).
        out = node_feat.new_full((num_graphs, node_feat.shape[-1]), float("-inf"))
        index = batch.unsqueeze(-1).expand_as(node_feat)
        return out.scatter_reduce_(0, index, node_feat, reduce="amax", include_self=True)
