import torch
from sage.all import Graph as SAGE_Graph
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from .compute_pairconn import compute_pair_conn
from .hierarchy_embed import norm_hierarchy_embed


def to_sagegraph(num_nodes, edges):
    nodes = list(range(num_nodes))
    sagegraph = SAGE_Graph([nodes, edges], format="vertices_and_edges")
    return sagegraph


class ConnRWPOfNodeEdge(BaseTransform):
    """Computing Local Connectivities and Ramdom Walk Probabilities as Postional Encoding of Nodes and Edges"""

    def __init__(self, max_conn, poly_dim) -> None:
        self.max_conn = max_conn
        self.poly_dim = poly_dim

    def __call__(self, data: Data):
        num_nodes = data.num_nodes
        edges = data.edge_index.T.tolist()
        sage_graph = to_sagegraph(num_nodes, edges)

        conn = compute_pair_conn(sage_graph)
        conn = torch.from_numpy(conn)
        data["pair_conn"] = conn

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

        conn = norm_hierarchy_embed(conn, self.max_conn)
        rwps = torch.stack(rwp_list, dim=-1)  # shape: [N, N, K]
        rwps = rwps.flatten(0, 1)
        pair_enc = torch.cat((conn, rwps), dim=1)

        diag_pos = torch.arange(num_nodes)
        diag_pos = diag_pos * num_nodes + diag_pos
        data["node_enc"] = pair_enc[diag_pos, :]
        edge_idx = edge_index[0] * num_nodes + edge_index[1]
        data["edge_enc"] = pair_enc[edge_idx, :]
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}_{self.max_conn}_{self.poly_name}_{self.poly_dim}"
