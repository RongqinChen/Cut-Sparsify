"""
Output decoder modules for different graph learning tasks.

This module provides various decoder architectures for graph-level and node-level
prediction tasks including regression and classification.
"""

from typing import Dict, Type

from torch import nn
from torch_geometric.data import Batch

from utils import cfg


class BaseDecoder(nn.Module):
    """Base class for output decoders."""
    
    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass that adds predictions to the batch.
        
        Args:
            batch: Input batch containing node or graph embeddings.
            
        Returns:
            Batch with predictions added.
        """
        raise NotImplementedError


class GraphRegression(BaseDecoder):
    """Graph-level regression decoder with 3-layer MLP."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize graph regression decoder.
        
        Args:
            in_channels: Number of input features.
            out_channels: Number of output features.
        """
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, out_channels),
        )

    def forward(self, batch: Batch) -> Batch:
        """Perform graph-level regression prediction."""
        batch["graph_pred"] = self.regressor(batch["graph_h"])
        return batch


class GraphClassification(BaseDecoder):
    """Graph-level classification decoder with 3-layer MLP."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize graph classification decoder.
        
        Args:
            in_channels: Number of input features.
            out_channels: Number of output classes.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, out_channels),
        )

    def forward(self, batch: Batch) -> Batch:
        """Perform graph-level classification prediction."""
        batch["graph_pred"] = self.classifier(batch["graph_h"])
        return batch


class MLPGraphHead(BaseDecoder):
    """Graph-level prediction head with dropout and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize MLP graph head with configurable dropout.
        
        Args:
            in_channels: Number of input features.
            out_channels: Number of output features.
        """
        super().__init__()
        dropout_prob = cfg.model.output_drop_prob
        self.mlp = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(in_channels, out_channels),
        )

    def forward(self, batch: Batch) -> Batch:
        """Perform graph-level prediction with dropout regularization."""
        batch["graph_pred"] = self.mlp(batch["graph_h"])
        return batch


class NodeClassification(BaseDecoder):
    """Node-level classification decoder with 2-layer MLP."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize node classification decoder.
        
        Args:
            in_channels: Number of input features.
            out_channels: Number of output classes.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, out_channels),
        )

    def forward(self, batch: Batch) -> Batch:
        """Perform node-level classification prediction."""
        batch["node_pred"] = self.classifier(batch["node_h"])
        return batch


class NodeRegression(BaseDecoder):
    """Node-level regression decoder with 2-layer MLP."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize node regression decoder.
        
        Args:
            in_channels: Number of input features.
            out_channels: Number of output features.
        """
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, out_channels),
        )

    def forward(self, batch: Batch) -> Batch:
        """Perform node-level regression prediction."""
        batch["node_pred"] = self.regressor(batch["node_h"])
        return batch


# Registry mapping decoder names to their corresponding classes
output_decoder_dict: Dict[str, Type[BaseDecoder]] = {
    "graph_regression": GraphRegression,
    "graph_classification": GraphClassification,
    "mlpgraphhead": MLPGraphHead,
    "node_classification": NodeClassification,
    "node_regression": NodeRegression,
}

__all__ = ["output_decoder_dict", "BaseDecoder"]
