"""
Model construction.
"""

from torch import nn

from utils import cfg


def make_model() -> nn.Module:
    r"""Make GNN model
    """
    from models.network import network_dict
    gnn = network_dict[cfg.model.name]()
    return gnn
