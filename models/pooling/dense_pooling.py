import torch
from torch.nn import Module
from torch_geometric.data import Batch


class DiagOffdiagAvgPooling(Module):
    """
    Pooling layer that computes average of diagonal and off-diagonal elements separately.
    
    Given a dense pair representation of shape (B, H, N, N), this layer computes:
    - Average of diagonal elements
    - Average of off-diagonal elements
    And concatenates them to produce a graph-level representation of shape (B, 2H).
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch) -> Batch:
        """
        Args:
            batch: Batch object containing 'dense_pair_h' of shape (B, H, N, N)
        
        Returns:
            batch: Updated batch with 'graph_h' of shape (B, 2H)
        """
        pair_h = batch["dense_pair_h"]  # Shape: (B, H, N, N)
        batch_size, hidden_dim, num_nodes, _ = pair_h.shape
        
        # Extract and sum diagonal elements: (B, H)
        diag_elements = torch.diagonal(pair_h, dim1=-2, dim2=-1)  # (B, H, N)
        diag_sum = diag_elements.sum(dim=-1)  # (B, H)
        diag_avg = diag_sum / num_nodes
        
        # Compute off-diagonal average
        if num_nodes == 1:
            offdiag_avg = torch.zeros_like(diag_avg)
        else:
            total_sum = pair_h.sum(dim=(-2, -1))  # (B, H)
            offdiag_sum = total_sum - diag_sum
            num_offdiag_elements = num_nodes * num_nodes - num_nodes
            offdiag_avg = offdiag_sum / num_offdiag_elements
        
        # Concatenate diagonal and off-diagonal averages: (B, 2H)
        batch.graph_h = torch.cat([diag_avg, offdiag_avg], dim=1)
        return batch


class DiagOffdiagSumPooling(Module):
    """
    Pooling layer that computes sum of diagonal and off-diagonal elements separately.
    
    Given a dense pair representation of shape (B, H, N, N), this layer computes:
    - Sum of diagonal elements
    - Sum of off-diagonal elements
    And concatenates them to produce a graph-level representation of shape (B, 2H).
    """
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, batch: Batch) -> Batch:
        """
        Args:
            batch: Batch object containing 'dense_pair_h' of shape (B, H, N, N)
        
        Returns:
            batch: Updated batch with 'graph_h' of shape (B, 2H)
        """
        pair_h = batch["dense_pair_h"]  # Shape: (B, H, N, N)
        
        # Extract and sum diagonal elements: (B, H)
        diag_elements = torch.diagonal(pair_h, dim1=-2, dim2=-1)  # (B, H, N)
        diag_sum = diag_elements.sum(dim=-1)  # (B, H)
        
        # Compute off-diagonal sum
        total_sum = pair_h.sum(dim=(-2, -1))  # (B, H)
        offdiag_sum = total_sum - diag_sum
        
        # Concatenate diagonal and off-diagonal sums: (B, 2H)
        batch.graph_h = torch.cat([diag_sum, offdiag_sum], dim=1)
        return batch
