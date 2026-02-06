import torch
from torch import nn


class DummyNodeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.encoder = nn.Embedding(num_embeddings=1, embedding_dim=out_dim)

    def forward(self, data):
        dummy_x = torch.zeros((data.num_nodes,), dtype=torch.long, device=data["edge_index"].device)
        data["node_h"] = self.encoder(dummy_x)
        return data


class DummyEdgeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.encoder = nn.Embedding(num_embeddings=1, embedding_dim=out_dim)

    def forward(self, data):
        dummy_x = data["edge_index"].new_zeros(data["edge_index"].size(1))
        data["edge_h"] = self.encoder(dummy_x)
        return data
