import torch
from typing import Optional
from torch_geometric.nn import Linear
from .jump_knowledge import JumpingKnowledge
from models.act import act_dict


layer_dict = {
    "linear": Linear,
    "jk": JumpingKnowledge,
}
