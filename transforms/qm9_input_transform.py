import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class QM9InputTransform(BaseTransform):
    """QM9 input feature transformation."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data: Data) -> Data:
        data.edge_attr = torch.where(data.edge_attr == 1)[-1]
        data.label = data.y.clone()
        del data.y
        return data
