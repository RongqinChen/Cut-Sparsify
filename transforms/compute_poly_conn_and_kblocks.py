import numpy as np
import torch
from sage.all import Graph as SAGE_Graph
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .compute_kblocks import compute_k_blocks
from .compute_polynomial import poly_method_dict
from .hierarchy_embed import norm_hierarchy_embed
from .v1_kcv_hierarchy import v1_kcv_hierarchy
from .v1_kcvdata import V1KCVData


def to_sagegraph(num_nodes, edges):
    nodes = list(range(num_nodes))
    sagegraph = SAGE_Graph([nodes, edges], format="vertices_and_edges")
    return sagegraph


class PolyConnAndKblock(BaseTransform):
    def __init__(self, max_kset_order=None, max_conn=None, poly_method=None, poly_dim=None) -> None:
        self.K = max_kset_order
        self.max_conn = max_conn
        self.poly_name = poly_method
        self.poly_fn = poly_method_dict[self.poly_name]
        self.poly_dim = poly_dim

    def __call__(self, data: Data):
        num_nodes = data.num_nodes
        edges = data.edge_index.T.tolist()

        sage_graph = to_sagegraph(num_nodes, edges)

        conn = np.zeros((num_nodes, num_nodes), dtype=np.int16)
        c0block = compute_k_blocks(sage_graph, self.K, None, conn)

        conn = torch.from_numpy(conn.flatten())
        conn = norm_hierarchy_embed(conn, self.max_conn)

        edge_index = data.edge_index
        num_nodes = data.num_nodes

        # compute row-normalized matrix
        adj_sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones((edge_index.size(1),)),
            (num_nodes, num_nodes),
            dtype=torch.float32,
        )
        adj = adj_sparse.to_dense()
        deg = adj.sum(dim=1, keepdim=True)
        data["log(1+deg)"] = torch.log1p(deg)

        if self.poly_name == "rrwp":
            deg_inv = 1.0 / deg
            deg_inv[deg_inv == float("inf")] = 0.0
            norm_adj = deg_inv * adj
        else:
            deg_ = deg.flatten()
            deg_sqrt_inv = torch.where(deg_ > 0, deg_ ** (-1 / 2), torch.zeros_like(deg_))
            deg_sqrt_inv = torch.diag_embed(deg_sqrt_inv)
            norm_adj = deg_sqrt_inv @ adj @ deg_sqrt_inv

        poly = self.poly_fn(norm_adj, self.poly_dim)
        poly = poly.flatten(0, 1)

        pair_x = torch.cat((conn, poly), dim=1)
        data["pair_x"] = pair_x

        diag_pos = torch.arange(num_nodes)
        diag_pos = diag_pos * num_nodes + diag_pos
        data["diag_x"] = pair_x[diag_pos, :]
        data["diag_pos"] = diag_pos

        attr_dict = v1_kcv_hierarchy(self.K, c0block)

        store_dict = dict(data.__dict__["_store"])
        store_dict.update(attr_dict)
        kcv_data = V1KCVData(**store_dict)
        return kcv_data
