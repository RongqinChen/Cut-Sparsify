from torch import nn


class QM9NodeEncoder(nn.Module):
    def __init__(self, hidden_dim, num_types=10):
        super().__init__()
        self.z_emb = nn.Embedding(num_types, hidden_dim)
        self.x_lin = nn.Linear(11, hidden_dim)

    def forward(self, data):
        h1 = self.z_emb(data["z"])
        h2 = self.x_lin(data.x)
        data["node_h"] = h1 + h2
        return data


class QM9EdgeEncoder(nn.Module):
    def __init__(self, hidden_dim, num_types=4):
        super().__init__()
        self.emb = nn.Embedding(num_types, hidden_dim)

    def forward(self, data):
        data["edge_h"] = self.emb(data.edge_attr)
        return data
