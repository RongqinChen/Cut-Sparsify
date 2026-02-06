import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    """
    Block of MLP layers with activation function after each (1x1 conv layers).
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mlp_depth (int): Number of MLP layers
        drop_prob (float): Dropout probability, default is 0.0
        activation_fn (callable): Activation function, default is ReLU
    """

    def __init__(self, in_channels, out_channels, mlp_depth, drop_prob=0.0, activation_fn=nn.functional.relu):
        super().__init__()
        self.activation = activation_fn
        self.dropout = nn.Dropout(drop_prob)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(mlp_depth):
            self.convs.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True))
            self.norms.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels

    def forward(self, inputs):
        """
        Forward pass through MLP block.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (N, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        out = self.convs[0](inputs)
        
        for idx in range(1, len(self.convs)):
            out = self.norms[idx](out)
            out = self.activation(out)
            out = self.dropout(out)
            out = self.convs[idx](out)

        return out


class SkipConnection(nn.Module):
    """
    Connects two input tensors with concatenation followed by 1x1 convolution.
    
    Args:
        in_channels (int): Total number of input channels (d1 + d2)
        out_channels (int): Number of output channels
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    def forward(self, in1, in2):
        """
        Forward pass through skip connection.
        
        Args:
            in1 (torch.Tensor): Earlier input tensor of shape (N, d1, H, W)
            in2 (torch.Tensor): Later input tensor of shape (N, d2, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


class RegularBlock(nn.Module):
    """
    Regular block with two parallel MLP routes.
    
    Takes input through 2 parallel MLP routes, multiplies the results using signed square root,
    and adds a skip-connection at the end to reduce dimension back to output_depth.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        mlp_depth (int): Depth of MLP layers
        drop_prob (float): Dropout probability, default is 0.0
    """

    def __init__(self, in_channels, out_channels, mlp_depth, drop_prob=0.0):
        super().__init__()
        self.out_channels = out_channels
        self.mlp1 = MlpBlock(in_channels, out_channels, mlp_depth, drop_prob)
        self.mlp2 = MlpBlock(in_channels, out_channels, mlp_depth, drop_prob)
        self.skip = SkipConnection(in_channels + out_channels, out_channels)

    def forward(self, inputs):
        """
        Forward pass through regular block.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (N, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        mlp1_out = self.mlp1(inputs)
        mlp2_out = self.mlp2(inputs)
        
        # Element-wise multiplication with signed square root normalization
        mult = torch.matmul(mlp1_out, mlp2_out)
        mult = torch.sqrt(torch.relu(mult)) - torch.sqrt(torch.relu(-mult))
        
        out = self.skip(in1=inputs, in2=mult)
        return out


class FullyConnected(nn.Module):
    """
    Fully connected layer with optional activation function.
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        activation_fn (callable, optional): Activation function, default is ReLU
    """
    
    def __init__(self, in_features, out_features, activation_fn=nn.functional.relu):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.activation = activation_fn

    def forward(self, inputs):
        """
        Forward pass through fully connected layer.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (N, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (N, out_features)
        """
        out = self.fc(inputs)
        if self.activation is not None:
            out = self.activation(out)

        return out
