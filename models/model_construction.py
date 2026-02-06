"""
Model construction utilities for building GNN models.
"""

from torch import nn

from utils import cfg


def make_model() -> nn.Module:
    """Create and initialize a GNN model based on configuration.
    
    Returns:
        nn.Module: The instantiated GNN model specified in cfg.model.name.
    
    Raises:
        KeyError: If the model name is not found in network_dict.
    """
    from models.network import network_dict
    
    model_name = cfg.model.name
    if model_name not in network_dict:
        raise KeyError(f"Model '{model_name}' not found in network_dict. "
                      f"Available models: {list(network_dict.keys())}")
    
    gnn = network_dict[model_name]()
    print(f"Created model: {model_name}")
    print(gnn)
    
    return gnn
