from torch.nn import Module
from torch_geometric.data import Batch
from torch_geometric.utils import scatter


class BaseGraphPooling(Module):
    """Base class for graph pooling operations."""
    
    def __init__(self, reduce_method: str) -> None:
        """
        Initialize the graph pooling module.
        
        Args:
            reduce_method: The reduction method ('mean', 'sum', or 'max')
        """
        super().__init__()
        self.reduce_method = reduce_method
    
    def _get_node_batch(self, batch: Batch):
        """Extract node batch indices from the batch object."""
        return batch.batch if hasattr(batch, 'batch') else batch['node'].batch
    
    def forward(self, batch: Batch) -> Batch:
        """
        Perform graph pooling operation.
        
        Args:
            batch: Input batch containing node features and batch indices
            
        Returns:
            Batch with graph-level representations stored in batch.graph_h
        """
        node_batch = self._get_node_batch(batch)
        batch.graph_h = scatter(
            batch.node_h, 
            node_batch, 
            dim=0, 
            dim_size=batch.num_graphs, 
            reduce=self.reduce_method
        )
        return batch


class GraphAvgPooling(BaseGraphPooling):
    """Graph pooling using average aggregation."""
    
    def __init__(self) -> None:
        super().__init__(reduce_method='mean')


class GraphSumPooling(BaseGraphPooling):
    """Graph pooling using sum aggregation."""
    
    def __init__(self) -> None:
        super().__init__(reduce_method='sum')


class GraphMaxPooling(BaseGraphPooling):
    """Graph pooling using max aggregation."""
    
    def __init__(self) -> None:
        super().__init__(reduce_method='max')
