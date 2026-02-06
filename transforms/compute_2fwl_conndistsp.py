from itertools import chain, permutations
from typing import List
import numpy as np
import torch
from sage.all import Graph as SAGE_Graph
from sage.graphs.connectivity import blocks_and_cuts_tree
from sage.graphs.distances_all_pairs import distances_all_pairs
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def to_sagegraph(num_nodes, edges):
    nodes = list(range(num_nodes))
    sagegraph = SAGE_Graph([nodes, edges], format="vertices_and_edges")
    return sagegraph


class K2FWLConnDistSpData(Data):
    """Connectivity and Distance Co-Guided Sparsifying For 2FWL-3Tuples Data"""
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, time=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "pair_index":         # [(v_i, v_j)]
            return self.num_nodes
        if key == "pair_x":
            return 0
        if key == "diag_pos":
            return self['pair_index'].size(1)
        if key == "triple_index":       # [(pair_i, pair_j, pair_k)]
            return self['pair_index'].size(1)
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


class K2FWLConnDistSpTransform(BaseTransform):
    """Connectivity and Distance Co-Guided Sparsifying"""
    def __init__(self, threshold_1, threshold_2, **kwargs):
        super().__init__()
        self.t1 = threshold_1
        self.t2 = threshold_2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold_1={self.t1}, threshold_2={self.t2})"

    def forward(self, data: Data):
        nums = data.num_nodes
        store_dict = dict(data.__dict__["_store"])
        assert "pair_x" in store_dict, "computing polynomials first"

        reserved_pairs, triples = compute_triples(data, self.t1, self.t2)
        diag_pos = torch.tensor([idx for idx, (a, b) in enumerate(reserved_pairs) if a == b], dtype=torch.long)
        pair_index = torch.tensor(reserved_pairs, dtype=torch.long).T.contiguous()
        pair_idx = pair_index[0] * nums + pair_index[1]
        pair_x = store_dict['pair_x'][pair_idx]

        triples = torch.tensor(triples, dtype=torch.long).t().contiguous()

        store_dict.update({
            "diag_pos": diag_pos,
            "pair_index": pair_index,
            "pair_idx": pair_idx,
            "pair_x": pair_x,
            "triple_index": triples
        })
        data = K2FWLConnDistSpData(**store_dict)
        return data


def compute_triples(data: Data, threshold_1, threshold_2) -> Tensor:
    nums = data.num_nodes
    edges = data.edge_index.T.tolist()
    sage_graph = to_sagegraph(nums, edges)
    dist_dict = distances_all_pairs(sage_graph)
    dist_mat = [
        [
            dist_dict[a][b] if isinstance(dist_dict[a][b], int) else 65535
            for b in range(nums)
        ]
        for a in range(nums)
    ]
    # dist_mat = np.array(dist_mat, dtype=np.int32)

    comp_list, bcnodes_list = biconn_comp_decompose(sage_graph)

    def test_triple_dist(a, b, c):
        return dist_mat[a][b] + dist_mat[a][c] + dist_mat[c][b] <= threshold_1

    def test_pair_dist(a, b):
        return dist_mat[a][b] <= threshold_2

    # 2c3verts
    # pair0, pair1, pair2 = hyper_triples
    # pair0 = Index(a, b); pair1 = Index(a, c); pair2 = Index(c, b); a != b != c
    full_c2_3verts = chain.from_iterable(permutations(bcnodes, 3) for bcnodes in bcnodes_list)
    triples = [
        (a * nums + b, a * nums + c, c * nums + b)
        for a, b, c in full_c2_3verts if test_triple_dist(a, b, c)
    ]

    if threshold_2 > 0:
        for comp in comp_list:
            c1_2verts = list(permutations(comp, r=2))
            triples += [
                (a * nums + b, a * nums + a, a * nums + b)
                for a, b in c1_2verts if test_pair_dist(a, b)
            ]
            triples += [
                (a * nums + a, a * nums + b, b * nums + a)
                for a, b in c1_2verts if test_pair_dist(a, b)
            ]

    triples += [
        (a * nums + a, a * nums + a, a * nums + a)
        for a in range(nums)
    ]
    triples = list(sorted(triples))

    reserved_pairs = {v for triple in triples for v in triple}
    reserved_pairs = list(sorted(reserved_pairs))
    hash_to_idx = {v: idx for idx, v in enumerate(reserved_pairs)}
    reserved_pairs = [(v // nums, v % nums) for v in reserved_pairs]

    triples = [
        (hash_to_idx[a], hash_to_idx[b], hash_to_idx[c])
        for a, b, c in triples
    ]

    return reserved_pairs, triples


def biconn_comp_decompose(sage_graph) -> List[List[int]]:
    # num_nodes = sage_graph.num_verts()
    # Step 1: Split all connected components in the graph.
    nodes_list: List[List[int]] = sage_graph.connected_components(sort=False)
    comp_list = [sage_graph.subgraph(nodes) for nodes in nodes_list]

    # Step 2: Construct a Block-cut tree for each connected component
    bctree_list = [blocks_and_cuts_tree(comp) for comp in comp_list]
    # bctree.vertices()[0] = ('B', [nodes...])

    # Step 3: Split all biconnected components of size at least 3 for each connected component
    bcnodes_list = [
        bcnode[1]
        for bctree in bctree_list
        for bcnode in bctree.vertices()
        if bcnode[0] == "B" and len(bcnode[1]) > 2
    ]
    return nodes_list, bcnodes_list


if __name__ == "__main__":
    sage_graph = to_sagegraph(3, [(1, 2)])
    dist_dict = distances_all_pairs(sage_graph)
    print(dist_dict)
