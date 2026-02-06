import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class RWPOfNodeEdge(BaseTransform):
    """Computing Ramdom Walk Probabilities as Postional Encoding of Nodes and Edges"""

    def __init__(self, poly_dim) -> None:
        self.poly_dim = poly_dim

    def __call__(self, data: Data):
        num_nodes = data.num_nodes
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
        deg = adj.sum(dim=1)
        deg_inv = torch.where(deg > 0, 1 / deg, torch.zeros_like(deg))
        deg_inv = torch.diag_embed(deg_inv)
        norm_adj = deg_inv @ adj

        eye = torch.eye(norm_adj.size(0))
        rwp_list = [eye, norm_adj]
        while len(rwp_list) < self.poly_dim:
            pe = norm_adj @ rwp_list[-1]
            rwp_list.append(pe)

        rwps = torch.stack(rwp_list, dim=-1)  # shape: [N, N, K]
        rwps = rwps.flatten(0, 1)

        diag_pos = torch.arange(num_nodes)
        diag_pos = diag_pos * num_nodes + diag_pos
        data["node_enc"] = rwps[diag_pos, :]
        edge_idx = edge_index[0] * num_nodes + edge_index[1]
        data["edge_enc"] = rwps[edge_idx, :]
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.poly_dim}"
