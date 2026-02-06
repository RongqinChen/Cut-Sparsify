from itertools import product

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class K2FWLData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, time=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "pair_index":         # [(v_i, v_j)]
            return self.num_nodes
        if key == "pair_x":
            return 0
        if key == "diag_pos":
            return self.num_nodes**2
        if key == "triple_index":       # [(pair_i, pair_j, pair_k)]
            return self.num_nodes**2
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "pair_index":         # [(v_i, v_j)]
            return 1
        if key == "pair_x":
            return 0
        if key == "diag_pos":
            return 0
        if key == "triple_index":       # [(pair_i, pair_j, pair_k)]
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class K2FWLTransform(BaseTransform):
    def __init__(self, **kwargs):
        # all triples (with repeats)
        super().__init__()

    def forward(self, data: Data):
        num_nodes = data.num_nodes
        triples = product(range(num_nodes), repeat=3)
        triples = [
            (a * num_nodes + b, a * num_nodes + c, c * num_nodes + b)
            for a, b, c in triples
        ]
        triples = torch.tensor(triples, dtype=torch.long).t().contiguous()
        node_idx = torch.arange(num_nodes, dtype=torch.long)
        diag_pos = node_idx * num_nodes + node_idx
        full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
        pair_index = full_mat.nonzero(as_tuple=False).t().contiguous()

        store_dict = dict(data.__dict__["_store"])
        store_dict.update({
            "diag_pos": diag_pos,
            "pair_index": pair_index,
            "triple_index": triples
        })
        data = K2FWLData(**store_dict)
        return data
