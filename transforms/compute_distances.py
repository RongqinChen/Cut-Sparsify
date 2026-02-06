import numpy as np
import torch
from sage.all import Graph as SAGE_Graph
from sage.graphs.distances_all_pairs import floyd_warshall
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .hierarchy_embed import norm_hierarchy_embed


def to_sagegraph(num_nodes, edges):
    nodes = list(range(num_nodes))
    sagegraph = SAGE_Graph([nodes, edges], format="vertices_and_edges")
    return sagegraph


class RD(BaseTransform):
    def __init__(self, poly_dim, **kwargs) -> None:
        super().__init__()
        self.poly_dim = poly_dim

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        adj_sparse = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones((data.edge_index.size(1),)),
            (num_nodes, num_nodes),
            dtype=torch.float32,
        )
        adj = adj_sparse.to_dense()
        deg = torch.sum(adj, dim=1)
        deg[deg < 1] = 1
        D = torch.diag(deg)
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.diag(D)))
        L_norm = torch.eye(num_nodes) - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        # Compute the pseudo-inverse of the Laplacian matrix
        L_norm_pinv = torch.pinverse(L_norm)

        n = L_norm.shape[0]
        resistance_distances = torch.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                resistance_distances[i, j] = L_norm_pinv[i, i] + L_norm_pinv[j, j] - 2 * L_norm_pinv[i, j]
                resistance_distances[j, i] = resistance_distances[i, j]

        mat = torch.stack((torch.eye(num_nodes) - L_norm, resistance_distances), -1)
        mat = mat.flatten(0, 1)
        data["diag_x"] = torch.ones((num_nodes, 2))
        data["pair_x"] = mat

        data["log(1+deg)"] = torch.log(1 + adj.sum(1, keepdim=True))

        store_dict = dict(data.__dict__["_store"])
        if "diag_pos" not in store_dict:
            node_range = torch.arange(num_nodes)
            diag_pos = node_range * num_nodes + node_range
            data["diag_pos"] = diag_pos

        if "pair_index" not in store_dict:
            full_mat = torch.ones((data.num_nodes, data.num_nodes), dtype=torch.short)
            full_index = full_mat.nonzero(as_tuple=False).t()
            data["pair_index"] = full_index

        return data


class SPD(BaseTransform):
    def __init__(self, poly_dim, **kwargs) -> None:
        super().__init__()
        self.poly_dim = poly_dim
        self.max_distance = poly_dim

    def forward(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        edges = data.edge_index.T.tolist()
        sage_graph = to_sagegraph(num_nodes, edges)

        pair_dist_mat = np.ones((num_nodes, num_nodes), dtype=np.int16) * 32760

        ret_dict = floyd_warshall(sage_graph, paths=False, distances=True)
        for u, v_dist in ret_dict.items():
            for v, dist in v_dist.items():
                pair_dist_mat[u, v] = dist
                pair_dist_mat[v, u] = pair_dist_mat[u, v]

        dist = pair_dist_mat.flatten()
        dist = torch.from_numpy(dist)
        invert_dist = self.max_distance - dist
        dist_emb = norm_hierarchy_embed(invert_dist, self.max_distance)

        data["diag_x"] = torch.ones((num_nodes, self.max_distance)) / self.max_distance
        data["pair_x"] = dist_emb

        # compute row-normalized matrix
        adj_sparse = torch.sparse_coo_tensor(
            data.edge_index,
            torch.ones((data.edge_index.size(1),)),
            (num_nodes, num_nodes),
            dtype=torch.float32,
        )
        adj = adj_sparse.to_dense()
        deg = adj.sum(dim=1, keepdim=True)
        data["log(1+deg)"] = torch.log(1 + deg)

        store_dict = dict(data.__dict__["_store"])
        if "diag_pos" not in store_dict:
            node_range = torch.arange(num_nodes)
            diag_pos = node_range * num_nodes + node_range
            data["diag_pos"] = diag_pos

        if "pair_index" not in store_dict:
            full_mat = torch.ones((data.num_nodes, data.num_nodes), dtype=torch.short)
            full_index = full_mat.nonzero(as_tuple=False).t()
            data["pair_index"] = full_index

        return data
