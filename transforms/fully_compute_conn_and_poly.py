from typing import List

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from .hierarchy_embed import norm_hierarchy_embed
from .fully_compute_pairconn import FullyComputePairConn
from .compute_polynomial import poly_method_dict


class FullConnectivityPolynomial(BaseTransform):
    def __init__(self, max_conn, poly_method, poly_dim, **kwargs) -> None:
        super().__init__()
        self.max_conn = max_conn
        self.poly_dim = poly_dim
        self.compute_conn = FullyComputePairConn()
        self._poly_name = str.upper(poly_method)
        self.poly_method = poly_method_dict[poly_method]

    def __call__(self, data_list: List[Data], save_dir):
        self.compute_conn.load_or_compute(data_list, save_dir)
        poly_name = self._poly_name
        desc = f'Computing `FullConn{poly_name}(ConnDim{self.max_conn}, PolyDim={self.poly_dim})` ...'
        for idx in tqdm(range(len(data_list)), desc):
            data = data_list[idx]
            self._embed_one(idx, data)

        return data_list

    def _embed_one(self, gidx: int, data: Data):
        num_nodes = data.num_nodes
        conn = self.compute_conn.conn_list[gidx]
        conn = torch.from_numpy(conn)
        conn = norm_hierarchy_embed(conn, self.max_conn)

        # compute row-normalized matrix
        edge_index = data.edge_index
        adj_sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones((edge_index.size(1),)),
            (num_nodes, num_nodes),
            dtype=torch.float32,
        )
        adj = adj_sparse.to_dense()
        deg = adj.sum(dim=1, keepdim=True)
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float("inf")] = 0.0
        norm_adj = deg_inv * adj
        data["log(1+deg)"] = torch.log(1 + deg)

        poly = self.poly_method(norm_adj, self.poly_dim)
        poly = poly.flatten(0, 1)
        data["pair_x"] = torch.cat([conn, poly], 1)

        diag_pos = torch.arange(num_nodes)
        diag_pos = diag_pos * num_nodes + diag_pos
        data["diag_x"] = data["pair_x"][diag_pos, :]

        if "pair_index" not in data:
            full_mat = torch.ones((data.num_nodes, data.num_nodes), dtype=torch.short)
            full_index = full_mat.nonzero(as_tuple=False).t()
            data["pair_index"] = full_index

        self.compute_conn.conn_list[gidx] = None

    def __repr__(self) -> str:
        poly_name = self._poly_name
        return f"FullConn{poly_name}_{self.max_conn}_{self.max_dist}"
