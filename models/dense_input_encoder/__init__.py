import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch

from utils import cfg
from models.input_encoder import edge_encoder_dict, node_encoder_dict, pe_encoder_dict


class DenseInputEncoder(nn.Module):

    def __init__(self, hidden_dim: int):
        super(DenseInputEncoder, self).__init__()
        self.node_encoder = node_encoder_dict[cfg.model.node_encoder](hidden_dim)
        self.edge_encoder = edge_encoder_dict[cfg.model.edge_encoder](hidden_dim)
        self.diag_encoder = pe_encoder_dict[cfg.model.pe_encoder](cfg.dataset.poly_dim, hidden_dim, False)
        self.pair_encoder = pe_encoder_dict[cfg.model.pe_encoder](cfg.dataset.poly_dim, hidden_dim, False)
        self.max_num_nodes = cfg.model.max_num_nodes

    def forward(self, batch: Batch) -> Batch:
        batch = self.node_encoder(batch)
        diag_h = self.diag_encoder(batch["diag_x"])

        node_h = batch["node_h"] + diag_h
        dense_node_h, dense_node_mask = to_dense_batch(node_h, batch["batch"], max_num_nodes=self.max_num_nodes)
        dense_node_h = dense_node_h.permute((0, 2, 1)).contiguous()  # B, C, N'
        dense_diag_h = torch.diag_embed(dense_node_h, dim1=-2, dim2=-1)

        batch = self.edge_encoder(batch)
        pair_h = self.pair_encoder(batch["pair_x"])
        dense_pair_h1 = to_dense_adj(batch["edge_index"], batch["batch"], batch["edge_h"], max_num_nodes=self.max_num_nodes)
        dense_pair_h2 = to_dense_adj(batch["pair_index"], batch["batch"], pair_h, max_num_nodes=self.max_num_nodes)
        # B, N', N', C
        dense_pair_h = dense_pair_h1 + dense_pair_h2
        dense_pair_h = dense_pair_h.permute((0, 3, 1, 2)).contiguous()

        batch["dense_pair_h"] = dense_pair_h + dense_diag_h
        batch["dense_node_mask"] = dense_node_mask
        return batch
