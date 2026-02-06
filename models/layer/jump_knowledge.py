from typing import List

import torch
from torch import Tensor, nn

from utils import cfg
from models.act import act_dict
from models.norms import norm_dict


class JumpingKnowledge(nn.Module):
    r"""Jumping Knowledge layer for aggregating multi-layer representations.
    
    Adapted from `torch_geometric.nn.models.jumping_knowledge`.
    
    Args:
        mode (str): The aggregation scheme to use. Supported modes:
            - "last": Use only the last layer's representation
            - "cat": Concatenate all layers and apply linear transformation
        dim_in (int): Input dimension of each layer
        num_convs (int): Number of convolutional layers to aggregate
        
    Raises:
        ValueError: If an unsupported mode is provided
    """
    
    def __init__(self, mode: str, dim_in: int, num_convs: int) -> None:
        super().__init__()
        self.mode = mode.lower()
        self.dim_in = dim_in
        self.num_convs = num_convs
        
        if self.mode == 'last':
            # No additional layers needed for 'last' mode
            pass
        elif self.mode == 'cat':
            # Concatenation mode: combine all layer outputs
            self.cat_lin = nn.Linear(dim_in * num_convs, dim_in)
            self.cat_norm = norm_dict[cfg.model.post_norm](dim_in)
            self.cat_act = act_dict[cfg.model.post_act]()
            self.cat_dropout = nn.Dropout(cfg.model.post_dropout)
        else:
            raise ValueError(
                f"Unsupported JumpingKnowledge mode: '{mode}'. "
                f"Supported modes are: 'last', 'cat'"
            )

    def forward(self, xs: List[Tensor]) -> Tensor:
        r"""Aggregate layer-wise representations.

        Args:
            xs (List[Tensor]): List of tensors containing layer-wise representations.
                Each tensor should have shape [batch_size, dim_in].

        Returns:
            Tensor: Aggregated representation with shape [batch_size, dim_in].
        """
        if self.mode == 'last':
            return xs[-1]
        
        elif self.mode == 'cat':
            # Concatenate all layer outputs along the feature dimension
            h = torch.cat(xs, dim=-1)  # [batch_size, num_convs * dim_in]
            h = self.cat_lin(h)
            h = self.cat_norm(h)
            h = self.cat_act(h)
            h = self.cat_dropout(h)
            return h
        
        else:
            # This should never be reached due to __init__ validation
            raise RuntimeError(f"Invalid mode encountered: {self.mode}")

    def __repr__(self) -> str:
        if self.mode == 'cat':
            return (
                f'{self.__class__.__name__}('
                f'mode={self.mode}, '
                f'dim_in={self.dim_in}, '
                f'num_convs={self.num_convs})'
            )
        return f'{self.__class__.__name__}(mode={self.mode})'
