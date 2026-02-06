import torch
from sage.all import Graph as SAGE_Graph
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .compute_pairconn import compute_pair_conn
from .compute_polynomial import poly_method_dict
from .hierarchy_embed import norm_hierarchy_embed


def to_sagegraph(num_nodes, edges):
    nodes = list(range(num_nodes))
    sagegraph = SAGE_Graph([nodes, edges], format="vertices_and_edges")
    return sagegraph


class ConnAndPoly(BaseTransform):
    def __init__(self, max_conn, poly_method, poly_dim) -> None:
        self.max_conn = max_conn
        self.poly_name = poly_method
        self.poly_fn = poly_method_dict[self.poly_name]
        self.poly_dim = poly_dim

    def __call__(self, data: Data):
        num_nodes = data.num_nodes
        edges = data.edge_index.T.tolist()
        sage_graph = to_sagegraph(num_nodes, edges)

        conn = compute_pair_conn(sage_graph)
        conn = torch.from_numpy(conn)
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

        poly = self.poly_fn(norm_adj, self.poly_dim)  # shape: [N, N, K]
        poly = poly.flatten(0, 1)

        pair_x = torch.cat((conn, poly), dim=1)
        data["pair_x"] = pair_x
        diag_pos = torch.arange(num_nodes)
        diag_pos = diag_pos * num_nodes + diag_pos
        data["diag_x"] = pair_x[diag_pos, :]

        full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
        pair_index = full_mat.nonzero(as_tuple=False).t()
        data["pair_index"] = pair_index
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.max_conn}_{self.poly_name}_{self.poly_dim}"
