import torch
from torch import nn
from torch_geometric.data import Batch


class BlockMLP(nn.Module):
    """Multi-layer perceptron with 1x1 convolutions, batch normalization, and dropout.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mlp_depth: Number of convolutional layers
        drop_prob: Dropout probability (default: 0.0)
    """
    
    def __init__(self, in_channels: int, out_channels: int, mlp_depth: int, drop_prob: float = 0.0):
        super().__init__()
        self.mlp_depth = mlp_depth
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First convolutional layer
        self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True))
        
        # Subsequent layers with normalization
        for _ in range(1, mlp_depth):
            self.norms.append(nn.BatchNorm2d(out_channels))
            self.convs.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after applying all layers
        """
        out = self.convs[0](x)
        
        for idx in range(1, len(self.convs)):
            out = self.norms[idx - 1](out)
            out = self.activation(out)
            out = self.dropout(out)
            out = self.convs[idx](out)

        return out


class BlockMatmulConv(nn.Module):
    """Matrix multiplication based convolution layer using two parallel MLPs.
    
    Args:
        channels: Number of channels
        mlp_depth: Depth of the MLP blocks
        drop_prob: Dropout probability (default: 0.0)
    """
    
    def __init__(self, channels: int, mlp_depth: int, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.mlp1 = BlockMLP(channels, channels, mlp_depth, drop_prob)
        self.mlp2 = BlockMLP(channels, channels, mlp_depth, drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with matrix multiplication and signed square root.
        
        Args:
            x: Input tensor of shape (B, H, N, N)
            
        Returns:
            Output tensor after signed square root transformation
        """
        mlp1_out = self.mlp1(x)
        mlp2_out = self.mlp2(x)
        mult = torch.matmul(mlp1_out, mlp2_out)
        
        # Signed square root: sqrt(relu(x)) - sqrt(relu(-x))
        out = torch.sqrt(torch.relu(mult)) - torch.sqrt(torch.relu(-mult))
        return out


class BlockUpdateLayer(nn.Module):
    """Update layer combining matrix convolution with residual connection.
    
    Args:
        channels: Number of channels
        mlp_depth: Depth of the MLP blocks
        drop_prob: Dropout probability
    """
    
    def __init__(self, channels: int, mlp_depth: int, drop_prob: float) -> None:
        super().__init__()
        self.matmul_conv = BlockMatmulConv(channels, mlp_depth, drop_prob)
        self.update = BlockMLP(channels * 2, channels, mlp_depth=2, drop_prob=drop_prob)

    def forward(self, batch: Batch) -> Batch:
        """Forward pass with residual connection.
        
        Args:
            batch: Input batch containing 'dense_pair_h' key
            
        Returns:
            Updated batch with modified 'dense_pair_h'
        """
        inputs = batch["dense_pair_h"]
        
        # Apply matmul convolution
        h = self.matmul_conv(inputs)
        
        # Concatenate with input and apply update with residual
        h = torch.cat([inputs, h], dim=1)
        h = self.update(h) + inputs
        
        batch["dense_pair_h"] = h
        return batch
