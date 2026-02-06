import torch
from torch import nn
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims


class OGBNodeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atom_embedding_list = nn.ModuleList()
        for dim in get_atom_feature_dims():
            emb = nn.Embedding(dim, hidden_dim)
            self.atom_embedding_list.append(emb)

    def forward(self, data):
        x_list = [
            emb(data.x[:, idx]) for idx, emb in enumerate(self.atom_embedding_list)
        ]
        x = torch.stack(x_list, 0).mean(0)
        data["node_h"] = x
        return data


class OGBEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.bond_embedding_list = nn.ModuleList()
        for dim in get_bond_feature_dims():
            emb = nn.Embedding(dim, hidden_dim)
            self.bond_embedding_list.append(emb)

    def forward(self, data):
        x_list = [
            emb(data.edge_attr[:, idx]) for idx, emb in enumerate(self.bond_embedding_list)
        ]
        x = torch.stack(x_list, 0).mean(0)
        data["edge_h"] = x
        return data
