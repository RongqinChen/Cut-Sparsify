from torch import nn
from utils import cfg


class LinearNodeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.encoder = nn.Linear(cfg.dataset.node_attr_dim, out_dim)

    def forward(self, data):
        data["node_h"] = self.encoder(data.x)
        return data


class LinearEdgeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.encoder = nn.Linear(cfg.dataset.edge_attr_dim, out_dim)

    def forward(self, data):
        data["edge_h"] = self.encoder(data.edge_attr)
        return data
