import torch
from math import comb
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def compute_bernstein_polynomial(norm_adj: torch.Tensor, poly_dim: int):
    K = poly_dim - 2
    eye = torch.eye(norm_adj.size(0))
    adj0 = eye + norm_adj
    adj1 = eye - norm_adj

    base1_list = [eye, adj0 / 2.0] + [None] * (K - 1)
    base2_list = [eye, adj1 / 2.0] + [None] * (K - 1)
    for k in range(2, K + 1):
        lidx, ridx = k // 2, k - k // 2
        base1_list[k] = base1_list[lidx] @ base1_list[ridx]
        base2_list[k] = base2_list[lidx] @ base2_list[ridx]

    bp_base_list = [base1_list[K - k] @ base2_list[k] for k in range(K + 1)]
    bp_coef_list = [comb(K, k) for k in range(K + 1)]
    poly_list = [bp_base_list[k] * bp_coef_list[k] for k in range(K + 1)]
    poly_list = [eye] + poly_list
    poly = torch.stack(poly_list, dim=-1)  # n x n x poly_dim
    return poly


def compute_chebyshev_polynomial(norm_adj: torch.Tensor, poly_dim: int):
    eye = torch.eye(norm_adj.size(0))
    poly_list = [eye, norm_adj]
    while len(poly_list) < poly_dim:
        poly = 2 * norm_adj @ poly_list[-1] - poly_list[-2]
        poly_list.append(poly)

    poly = torch.stack(poly_list, dim=-1)  # n x n x poly_dim
    return poly


def compute_mono_polynomial(norm_adj: torch.Tensor, poly_dim: int):
    eye = torch.eye(norm_adj.size(0))
    poly_list = [eye, norm_adj]
    while len(poly_list) < poly_dim:
        pe = norm_adj @ poly_list[-1]
        poly_list.append(pe)

    poly = torch.stack(poly_list, dim=-1)  # n x n x poly_dim
    return poly


poly_method_dict = {
    "bern": compute_bernstein_polynomial,
    "cheb": compute_chebyshev_polynomial,
    "rrwp": compute_mono_polynomial,
    "mono": compute_mono_polynomial,
}


class Polynomials(BaseTransform):
    def __init__(self, poly_method, poly_dim, **kwargs) -> None:
        super().__init__()
        self.poly_dim = poly_dim
        self.poly_name = poly_method
        self.poly_fn = poly_method_dict[poly_method]

    def forward(self, data: Data) -> Data:
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
        deg_ = deg.flatten()

        if self.poly_name == "rrwp":
            deg_inv = torch.where(deg_ > 0, 1 / deg_, torch.zeros_like(deg_))
            deg_inv = torch.diag_embed(deg_inv)
            norm_adj = deg_inv @ adj
        else:
            deg_sqrt_inv = torch.where(deg_ > 0, deg_ ** (-1 / 2), torch.zeros_like(deg_))
            deg_sqrt_inv = torch.diag_embed(deg_sqrt_inv)
            norm_adj = deg_sqrt_inv @ adj @ deg_sqrt_inv

        poly = self.poly_fn(norm_adj, self.poly_dim)
        data["pair_x"] = poly.flatten(0, 1)
        data["diag_x"] = torch.diagonal(poly).permute((1, 0))

        full_mat = torch.ones((data.num_nodes, data.num_nodes), dtype=torch.short)
        full_index = full_mat.nonzero(as_tuple=False).t()
        data["pair_index"] = full_index
        return data

    def __repr__(self) -> str:
        return f"{self.poly_name}_{self.poly_dim}"
