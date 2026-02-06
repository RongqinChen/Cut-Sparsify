"""
PPGN (Provably Powerful Graph Networks) with Positional Encoding.

This module implements a PPGN architecture for graph-level tasks.
"""

from typing import List

from torch import nn
from torch_geometric.data import Batch

from utils import cfg
from models.dense_input_encoder import DenseInputEncoder
from models.layer import layer_dict
from models.layer.ppgn_update import BlockUpdateLayer
from models.pooling import dense_pooling_dict
from models.output_decoder import output_decoder_dict


class PPGN(nn.Module):
    """PPGN implementation for graph neural networks.

    This model consists of four main components:
    1. Input encoding layer
    2. Multiple aggregation and update blocks
    3. Readout layer with jumping knowledge
    4. Output decoding layer

    All hyperparameters are retrieved from the global configuration (cfg.model).

    Args:
        Configuration is pulled from cfg.model with the following parameters:
        - hidden_dim (int): Hidden dimension size.
        - num_layers (int): Number of message passing layers.
        - mlp_depth (int): Depth of MLPs in each block.
        - pooling (str): Graph pooling method ('avg' or 'sum').
        - drop_prob (float): Dropout probability.
        - jk_mode (str): Jumping knowledge mode ('last', 'concat', or 'lstm').
        - task_type (str): Type of task (e.g., 'graph_classification', 'graph_regression').
        - num_tasks (int): Number of prediction tasks.
    """

    def __init__(self) -> None:
        super().__init__()

        # Extract configuration parameters
        hidden_dim: int = cfg.model.hidden_dim
        num_layers: int = cfg.model.num_layers
        mlp_depth: int = cfg.model.mlp_depth
        pooling: str = cfg.model.pooling
        drop_prob: float = cfg.model.drop_prob
        jumping_knowledge: str = cfg.model.jk_mode
        task_type: str = cfg.model.task_type
        num_tasks: int = cfg.model.num_tasks

        # Initialize dropout
        self.dropout = nn.Dropout(drop_prob)

        # Component 1: Input encoding
        self.input_encoder = DenseInputEncoder(hidden_dim)

        # Component 2: Aggregation and update blocks
        self.blocks = nn.ModuleList([
            BlockUpdateLayer(hidden_dim, mlp_depth, drop_prob)
            for _ in range(num_layers)
        ])

        # Component 3: Readout with jumping knowledge
        self.jk = layer_dict['jk'](jumping_knowledge, hidden_dim * 2, num_layers + 1)
        self.dense_pooling = dense_pooling_dict[pooling]()

        # Component 4: Output decoding
        self.output_decoder = output_decoder_dict[task_type](hidden_dim * 2, num_tasks)

        # Initialize model parameters
        self._reset_parameters()

    def alias(self) -> str:
        """Generate a descriptive alias for the model based on its configuration.

        Returns:
            str: Model alias in format 'PPGN_L{num_layers}D{hidden_dim}'.
        """
        return f"PPGN_L{cfg.model.num_layers}D{cfg.model.hidden_dim}"

    def _reset_parameters(self) -> None:
        """Reset all model parameters using Xavier initialization.

        Initializes Conv2d and Linear layers with Xavier normal initialization,
        and calls reset_parameters() for other modules that support it.
        """
        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.apply(_init_weights)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass through the PPGN model.

        Args:
            batch (Batch): Input batch containing graph data.

        Returns:
            Batch: Output batch with predictions stored in appropriate fields.
        """
        # Encode input features
        batch = self.input_encoder(batch)

        # Collect hidden states from all layers for jumping knowledge
        hidden_states: List = []
        for block in self.blocks:
            batch = block(batch)
            hidden_states.append(batch["dense_pair_h"])

        # Apply jumping knowledge if configured
        if self.jk is not None:
            batch["dense_pair_h"] = self.jk(hidden_states)

        # Pool node representations to graph-level
        batch = self.dense_pooling(batch)

        # Decode to final predictions
        batch = self.output_decoder(batch)

        return batch
