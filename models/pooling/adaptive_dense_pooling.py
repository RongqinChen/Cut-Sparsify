import torch
from torch.nn import Module
from torch_geometric.data import Batch


class AdaptiveDiagOffdiagAvgPooling(Module):
    """Adaptive pooling layer that computes averaged diagonal and off-diagonal features."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass that computes average pooling over diagonal and off-diagonal elements.
        
        Args:
            batch: Input batch containing dense_node_mask and dense_pair_h
            
        Returns:
            batch: Updated batch with graph_h containing concatenated diagonal and off-diagonal averages
        """
        dense_node_mask = batch["dense_node_mask"]
        num_nodes = dense_node_mask.sum(dim=1, keepdim=True)
        
        # Create mask for valid pairs: (B, 1, N, N)
        mask_2d = dense_node_mask.unsqueeze(1).unsqueeze(-1) * dense_node_mask.unsqueeze(1).unsqueeze(2)
        
        # Apply mask to pair features
        pair_h = batch["dense_pair_h"] * mask_2d  # shape: (B, H, N, N)
        
        # Compute diagonal average
        diag_elements = torch.diagonal(pair_h, dim1=-2, dim2=-1)  # (B, H, N)
        diag_sum = diag_elements.sum(dim=2)  # (B, H)
        diag_avg = diag_sum / num_nodes
        
        # Compute off-diagonal average
        num_offdiag = num_nodes ** 2 - num_nodes
        if num_offdiag.min() == 0:
            # Handle edge case where there are no off-diagonal elements
            offdiag_avg = torch.zeros_like(diag_avg)
        else:
            total_sum = pair_h.sum(dim=[-1, -2])  # (B, H)
            offdiag_sum = total_sum - diag_sum
            offdiag_avg = offdiag_sum / num_offdiag
        
        # Concatenate diagonal and off-diagonal features
        batch["graph_h"] = torch.cat([diag_avg, offdiag_avg], dim=1)  # (B, 2H)
        return batch


class AdaptiveDiagOffdiagSumPooling(Module):
    """Adaptive pooling layer that computes summed diagonal and off-diagonal features."""
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch) -> Batch:
        """
        Forward pass that computes sum pooling over diagonal and off-diagonal elements.
        
        Args:
            batch: Input batch containing dense_node_mask and dense_pair_h
            
        Returns:
            batch: Updated batch with graph_h containing concatenated diagonal and off-diagonal sums
        """
        dense_node_mask = batch["dense_node_mask"]
        
        # Create mask for valid pairs: (B, 1, N, N)
        mask_2d = dense_node_mask.unsqueeze(1).unsqueeze(-1) * dense_node_mask.unsqueeze(1).unsqueeze(2)
        
        # Apply mask to pair features
        pair_h = batch["dense_pair_h"] * mask_2d  # shape: (B, H, N, N)
        
        # Compute diagonal sum
        diag_elements = torch.diagonal(pair_h, dim1=-2, dim2=-1)  # (B, H, N)
        diag_sum = diag_elements.sum(dim=2)  # (B, H)
        
        # Compute off-diagonal sum
        total_sum = pair_h.sum(dim=[-1, -2])  # (B, H)
        offdiag_sum = total_sum - diag_sum
        
        # Concatenate diagonal and off-diagonal features
        batch["graph_h"] = torch.cat([diag_sum, offdiag_sum], dim=1)  # (B, 2H)
        return batch
