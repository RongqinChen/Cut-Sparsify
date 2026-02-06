from torch.nn import Identity
from torch_geometric.nn.norm import BatchNorm, LayerNorm, InstanceNorm, GraphSizeNorm, PairNorm


norm_dict = {
    "batch": BatchNorm,
    "graphsize": GraphSizeNorm,
    "identity": Identity,
    "instance": InstanceNorm,
    "layer": LayerNorm,
    "pair": PairNorm,
}
