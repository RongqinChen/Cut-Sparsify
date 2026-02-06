import torch
import torch_sparse
from torch import nn
from torch_scatter import scatter_add
from utils import cfg
from models.input_encoder import edge_encoder_dict, node_encoder_dict
from models.output_decoder import output_decoder_dict


class SPPGNLayer(nn.Module):
    """Single layer of the Simplified Provably Powerful Graph Network."""

    def __init__(self, mlp_depth: int, hidden_dim: int, use_sqrt: bool, drop_prob: float):
        super().__init__()
        self.mlp_depth = mlp_depth
        self.use_sqrt = use_sqrt
        self.hidden_dim = hidden_dim

        # Create MLP blocks for feature transformation
        self.mlp1 = self._create_mlp_block(mlp_depth, hidden_dim, hidden_dim, drop_prob)
        self.mlp2 = self._create_mlp_block(mlp_depth, hidden_dim, hidden_dim, drop_prob)
        self.upd = self._create_mlp_block(mlp_depth, hidden_dim * 2, hidden_dim, drop_prob)

    @staticmethod
    def _create_mlp_block(mlp_depth: int, in_dim: int, out_dim: int, drop_prob: float) -> nn.Sequential:
        """Create a standard MLP block with BatchNorm, ReLU, and Dropout."""
        layers = [nn.Linear(in_dim, out_dim)]
        for _ in range(mlp_depth):
            layers += [
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(out_dim, out_dim),
            ]

    def forward(self, data: dict) -> dict:
        idx0, idx1, idx2 = data["triple_index"]
        x2 = data["pair_h"]

        # Apply MLPs to pair features
        x2_1 = self.mlp1(x2)
        x2_2 = self.mlp2(x2)

        # Compute triple interactions and aggregate
        x3 = x2_1[idx1] * x2_2[idx2]
        x3_agg = scatter_add(x3, idx0, dim=0, dim_size=x2.size(0))

        # Apply signed square root if enabled
        if self.use_sqrt:
            x3_agg = torch.sqrt(torch.relu(x3_agg)) - torch.sqrt(torch.relu(-x3_agg))

        # Update pair features with residual connection
        h2 = torch.cat([x2, x3_agg], dim=-1)
        data["pair_h"] = self.upd(h2) + x2
        return data

    def extra_repr(self) -> str:
        return f"use_sqrt={self.use_sqrt}"


class SPPGN(nn.Module):
    """Subgraph Positional Pair Graph Network for graph-level predictions."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = cfg.model.hidden_dim
        self.num_layers = cfg.model.num_layers
        self.use_sqrt = True
        self.drop_prob = cfg.model.drop_prob
        self.pooling = cfg.model.pooling

        # Initialize layers
        self.init_layer = InitLayer()
        self.layers = nn.ModuleList([
            SPPGNLayer(self.hidden_dim, self.use_sqrt, self.drop_prob)
            for _ in range(self.num_layers)
        ])

        # Output decoder
        output_decoder_cls = output_decoder_dict[cfg.model.task_type]
        self.output_decoder = output_decoder_cls(self.hidden_dim * 2, cfg.model.num_tasks)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize model parameters."""
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif hasattr(module, "reset_parameters"):
                module.reset_parameters()

        self.apply(_init_weights)

    def forward(self, data: dict) -> dict:
        # Initialize node and pair features
        data = self.init_layer(data)

        # Apply SPPGN layers
        for layer in self.layers:
            data = layer(data)

        # Compute graph-level aggregations
        pair_h = data["pair_h"]
        diag_pos = data["diag_pos"]

        # Aggregate all pair features and diagonal (node) features
        agg_all_pairs = scatter_add(pair_h, data["pair_x_batch"], dim=0, dim_size=data.num_graphs)
        agg_diagonal = scatter_add(pair_h[diag_pos], data["batch"], dim=0, dim_size=data.num_graphs)
        agg_off_diagonal = agg_all_pairs - agg_diagonal

        # Apply pooling normalization
        graph_node_sizes = torch.diff(data.ptr).view(-1, 1)

        if self.pooling in {"avg", "mean"}:
            graph_pair_sizes = torch.diff(data["pair_x_ptr"]).view(-1, 1)
            diff_sizes = torch.clamp(graph_pair_sizes - graph_node_sizes, min=1)
            agg_off_diagonal = agg_off_diagonal / diff_sizes
            agg_diagonal = agg_diagonal / graph_node_sizes
        elif self.pooling == "sum_avg":
            agg_off_diagonal = agg_off_diagonal / graph_node_sizes

        # Combine aggregations and decode
        data["graph_h"] = torch.cat([agg_diagonal, agg_off_diagonal], dim=-1)
        data = self.output_decoder(data)
        return data


class InitLayer(nn.Module):
    """Initialization layer for encoding node, edge, and pair features."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_dim = cfg.model.hidden_dim
        self.poly_dim = cfg.dataset.poly_dim
        self.pe_encoder = cfg.model.pe_encoder

        # Initialize encoders
        self.node_encoder = node_encoder_dict[cfg.model.node_encoder](self.hidden_dim)
        self.edge_encoder = edge_encoder_dict[cfg.model.edge_encoder](self.hidden_dim)

        # Linear layers for polynomial features
        self.diag_lin = nn.Linear(self.poly_dim, self.hidden_dim, bias=False)
        self.pair_lin = nn.Linear(self.poly_dim, self.hidden_dim, bias=False)

        # Connectivity encoders if specified
        if "conn" in self.pe_encoder:
            self.conn_dim = int(self.pe_encoder.split("+")[0][4:])
            self.diag_conn = nn.Linear(self.conn_dim, self.hidden_dim, bias=False)
            self.pair_conn = nn.Linear(self.conn_dim, self.hidden_dim, bias=False)
        else:
            self.conn_dim = None

    def forward(self, data: dict) -> dict:
        # Encode node and edge features
        data = self.node_encoder(data)
        data = self.edge_encoder(data)

        # Process loop and pair features
        diag_x = data["diag_x"]
        pair_x = data["pair_x"]

        if self.conn_dim is None:
            node_value = self.diag_lin(diag_x)
            pair_value = self.pair_lin(pair_x)
        else:
            # Split connectivity and polynomial features
            diag_h = self.diag_lin(diag_x[:, self.conn_dim:])
            pair_h = self.pair_lin(pair_x[:, self.conn_dim:])

            # Process connectivity features with scaling
            diag_conn = self.diag_conn(diag_x[:, :self.conn_dim]) * (self.poly_dim / self.conn_dim)
            pair_conn = self.pair_conn(pair_x[:, :self.conn_dim]) * (self.poly_dim / self.conn_dim)

            node_value = diag_h + diag_conn
            pair_value = pair_h + pair_conn

        # Create diagonal indices for self-loops
        node_range = torch.arange(data.num_nodes, device=node_value.device)
        diag_index = torch.stack([node_range, node_range], dim=0)

        # Combine node features
        node_value = data["node_h"] + node_value

        # Coalesce pair, edge, and diagonal features into unified pair representation
        pair_index = data["pair_index"]
        edge_index = data["edge_index"]
        edge_value = data["edge_h"]

        _, pair_h = torch_sparse.coalesce(
            torch.cat([pair_index, edge_index, diag_index], dim=1),
            torch.cat([pair_value, edge_value, node_value], dim=0),
            data.num_nodes,
            data.num_nodes,
            op="add"
        )

        data["pair_h"] = pair_h
        return data
