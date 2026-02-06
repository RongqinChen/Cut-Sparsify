from itertools import chain, permutations, product
from typing import List

import torch
from sage.all import Graph as SAGE_Graph
from sage.graphs.connectivity import blocks_and_cuts_tree
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def to_sagegraph(num_nodes, edges):
    nodes = list(range(num_nodes))
    sagegraph = SAGE_Graph([nodes, edges], format="vertices_and_edges")
    return sagegraph


class K2FWLConnSplitData(Data):
    """Connectivity-Guided Spliting For 2FWL-3Tuples Data"""
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
        if key == "c01_triple_index":   # [(pair_i, pair_j, pair_k)]
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
        if key == "c01_triple_index":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class K2FWLConnSplitDataTransform(BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data: Data):
        num_nodes = data.num_nodes
        triples, e1c_triples = compute_triples(data)

        triples = torch.tensor(triples, dtype=torch.long).t().contiguous()
        e1c_triples = torch.tensor(e1c_triples, dtype=torch.long).t().contiguous()
        node_idx = torch.arange(num_nodes, dtype=torch.long)
        diag_pos = node_idx * num_nodes + node_idx

        store_dict = dict(data.__dict__["_store"])
        store_dict.update({
            "diag_pos": diag_pos,
            "triple_index": triples,
            "c01_triple_index": e1c_triples,
        })
        if "pair_index" not in store_dict:
            full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
            pair_index = full_mat.nonzero(as_tuple=False).t().contiguous()
            store_dict['pair_index'] = pair_index

        data = K2FWLConnSplitData(**store_dict)
        return data


def compute_triples(data: Data) -> Tensor:
    num_nodes = data.num_nodes
    edges = data.edge_index.T.tolist()
    sage_graph = to_sagegraph(num_nodes, edges)
    comp_list, bcnodes_list, e1c_triples = components_decomposition(sage_graph)

    # 2c3verts
    # pair0, pair1, pair2 = hyper_triples
    # pair0 = Index(a, b); pair1 = Index(a, c); pair2 = Index(c, b); a != b != c
    full_c2_3verts = chain.from_iterable(permutations(bcnodes, 3) for bcnodes in bcnodes_list)
    hyper_triples = [
        (a * num_nodes + b, a * num_nodes + c, c * num_nodes + b)
        for a, b, c in full_c2_3verts
    ]
    hyper_triples += [
        (a * num_nodes + a, a * num_nodes + a, a * num_nodes + a)
        for a in range(num_nodes)
    ]
    hyper_triples = list(sorted(hyper_triples))

    hyper_e1c_triples = [
        (a * num_nodes + b, b * num_nodes + c)
        for a, b, c in e1c_triples
    ]
    for comp in comp_list:
        full_2verts = list(permutations(comp, r=2))
        hyper_e1c_triples += [
            (a * num_nodes + b, b * num_nodes + b)
            for a, b in full_2verts
        ]
        hyper_e1c_triples += [
            (a * num_nodes + a, a * num_nodes + b)
            for a, b in full_2verts
        ]

    hyper_e1c_triples += [
        (a * num_nodes + a, a * num_nodes + a)
        for a in range(num_nodes)
    ]
    hyper_e1c_triples = list(sorted(hyper_e1c_triples))

    return hyper_triples, hyper_e1c_triples


def components_decomposition(sage_graph) -> List[List[int]]:
    nodes_list: List[List[int]] = sage_graph.connected_components(sort=False)

    e1c_triples = []
    bcnodes_list: List[List[int]] = []

    for nodes in nodes_list:
        component = sage_graph.subgraph(nodes)
        nodes = set(nodes)

        bctree = blocks_and_cuts_tree(component)
        for bcnode in bctree.vertices():
            if bcnode[0] == "B" and len(bcnode[1]) > 2:
                bc_nodes = bcnode[1]
                bcnodes_list.append(bc_nodes)
                other_nodes = nodes - set(bc_nodes)
                e1c_triples += [(a, b, c) for a, b, c in product(bc_nodes, bc_nodes, other_nodes) if a != b]

    return nodes_list, bcnodes_list, e1c_triples
