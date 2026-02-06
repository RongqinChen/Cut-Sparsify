from torch import nn
from torch_geometric.data import Data, Batch

from utils import cfg


class IdentityNodeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, data: Data | Batch):
        data["node_h"] = data.x
        return data


class IdentityEdgeEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()

    def forward(self, data: Data | Batch):
        data["edge_h"] = data.edge_attr
        return data


class IdentityPEEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x
