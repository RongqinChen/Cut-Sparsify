from torch import nn
from utils import cfg

"""
Generic Node and Edge encoders for datasets with node/edge features that
consist of only one type dictionary thus require a single nn.Embedding layer.

The number of possible Node and Edge types must be set by cfg options:
1) cfg.dataset.node_attr_dim
2) cfg.dataset.edge_attr_dim

In case of a more complex feature set, use a data-specific encoder.

These generic encoders can be used e.g. for:
* ZINC
cfg.dataset.node_attr_dim: 28
cfg.dataset.edge_attr_dim: 4

* AQSOL
cfg.dataset.node_attr_dim: 65
cfg.dataset.edge_attr_dim: 5


=== Description of the ZINC dataset ===
https://github.com/graphdeeplearning/benchmarking-gnns/issues/42
The node labels are atom types and the edge labels atom bond types.

Node labels:
'C': 0
'O': 1
'N': 2
'F': 3
'C H1': 4
'S': 5
'Cl': 6
'O -': 7
'N H1 +': 8
'Br': 9
'N H3 +': 10
'N H2 +': 11
'N +': 12
'N -': 13
'S -': 14
'I': 15
'P': 16
'O H1 +': 17
'N H1 -': 18
'O +': 19
'S +': 20
'P H1': 21
'P H2': 22
'C H2 -': 23
'P +': 24
'S H1 +': 25
'C H1 -': 26
'P H1 +': 27

Edge labels:
'NONE': 0
'SINGLE': 1
'DOUBLE': 2
'TRIPLE': 3


=== Description of the AQSOL dataset ===
Node labels:
'Br': 0, 'C': 1, 'N': 2, 'O': 3, 'Cl': 4, 'Zn': 5, 'F': 6, 'P': 7, 'S': 8, 'Na': 9, 'Al': 10,
'Si': 11, 'Mo': 12, 'Ca': 13, 'W': 14, 'Pb': 15, 'B': 16, 'V': 17, 'Co': 18, 'Mg': 19, 'Bi': 20, 'Fe': 21,
'Ba': 22, 'K': 23, 'Ti': 24, 'Sn': 25, 'Cd': 26, 'I': 27, 'Re': 28, 'Sr': 29, 'H': 30, 'Cu': 31, 'Ni': 32,
'Lu': 33, 'Pr': 34, 'Te': 35, 'Ce': 36, 'Nd': 37, 'Gd': 38, 'Zr': 39, 'Mn': 40, 'As': 41, 'Hg': 42, 'Sb':
43, 'Cr': 44, 'Se': 45, 'La': 46, 'Dy': 47, 'Y': 48, 'Pd': 49, 'Ag': 50, 'In': 51, 'Li': 52, 'Rh': 53,
'Nb': 54, 'Hf': 55, 'Cs': 56, 'Ru': 57, 'Au': 58, 'Sm': 59, 'Ta': 60, 'Pt': 61, 'Ir': 62, 'Be': 63, 'Ge': 64

Edge labels:
'NONE': 0, 'SINGLE': 1, 'DOUBLE': 2, 'AROMATIC': 3, 'TRIPLE': 4
"""


class TypeDictNodeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.Embedding(cfg.dataset.node_attr_dim, hidden_dim)

    def forward(self, data):
        data["node_h"] = self.encoder(data.x)
        return data


class TypeDictEdgeEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.encoder = nn.Embedding(cfg.dataset.edge_attr_dim, hidden_dim)

    def forward(self, data):
        data["edge_h"] = self.encoder(data.edge_attr)
        return data
