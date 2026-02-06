import torch

class NoneEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, data):
        return data
