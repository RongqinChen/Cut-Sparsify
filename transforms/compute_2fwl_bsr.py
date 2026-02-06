from itertools import permutations
from typing import Dict, List, Optional, Tuple

import torch
from networkx import Graph as NetworkXGraph
from sage.all import Graph as SageGraph
from sage.graphs.connectivity import blocks_and_cuts_tree as compute_block_cut_tree
from sage.graphs.connectivity import spqr_tree as compute_spqr_tree
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def decompose_graph_to_block_spqr(
    graph: SageGraph,
    stats: Optional[Dict] = None
) -> Dict:
    """
    Decompose graph into Block-SPQR components.

    Args:
        graph: A SageGraph instance to decompose
        stats: Optional dictionary to store decomposition statistics

    Returns:
        Dictionary containing vertices, edges, components, and node triples
    """
    num_nodes = graph.num_verts()
    num_edges = graph.num_edges()

    if stats is not None:
        stats.update({
            "num_nodes": num_nodes,
            "num_edges": num_edges * 2,
            "component_sizes": [],
            "block_sizes": [],
            "S_sizes": [],
            "P_sizes": [],
            "Q_sizes": [],
            "R_sizes": [],
        })

    component_node_lists = graph.connected_components(sort=False)
    components = [graph.subgraph(nodes) for nodes in component_node_lists]

    if stats is not None:
        stats["component_sizes"].extend([len(c.vertices()) for c in components])

    # Create bidirectional edge list
    edges = [(u, v) for u, v, _ in graph.edges()]
    edges.extend([(v, u) for u, v in edges])

    result = {
        "vertices": list(range(num_nodes)),
        "edges": edges,
        "connected_components": component_node_lists,
        "block_components": [],
        "S_components": [],
        "P_components": [],
        "Q_components": [],
        "R_components": [],
    }

    for component in components:
        decompose_component(component, result, stats)

    # Generate node triples
    triples = []
    for nodes in (result["S_components"] + result["R_components"]):
        triples.extend(permutations(nodes, 3))

    for u, v in result["edges"]:
        triples.extend([(u, v, v), (u, u, v), (u, v, u)])

    # Add self-loop triples
    triples.extend([(v, v, v) for v in range(num_nodes)])

    result["node_triples"] = triples
    return result


def decompose_component(
    component: SageGraph,
    result: Dict,
    stats: Optional[Dict] = None
) -> None:
    """
    Decompose connected component into blocks and SPQR components.

    Args:
        component: A connected component as SageGraph
        result: Dictionary to store decomposition results
        stats: Optional dictionary to store statistics
    """
    block_tree = compute_block_cut_tree(component)

    blocks = [
        component.subgraph(nodes)
        for node_type, nodes in block_tree.vertices()
        if node_type == "B" and len(nodes) > 2
    ]
    block_nodes = [block.vertices() for block in blocks]
    result["block_components"].extend(block_nodes)

    if stats is not None:
        stats["block_sizes"].extend([len(nodes) for nodes in block_nodes])

    for block in blocks:
        decompose_block(block, result, stats)


def decompose_block(
    block: SageGraph,
    result: Dict,
    stats: Optional[Dict] = None
) -> None:
    """
    Decompose biconnected block into SPQR components.

    Args:
        block: A biconnected block as SageGraph
        result: Dictionary to store SPQR component results
        stats: Optional dictionary to store component size statistics
    """
    spqr = compute_spqr_tree(block)

    for comp_type, comp_graph in spqr.vertices():
        nodes = list(comp_graph.vertices())
        result[f"{comp_type}_components"].append(nodes)

        if stats is not None:
            stats[f"{comp_type}_sizes"].append(len(nodes))


def convert_to_sage_graph(num_nodes: int, edges: List[Tuple[int, int]]) -> SageGraph:
    """
    Convert edge list to SAGE graph.

    Args:
        num_nodes: Number of nodes in the graph
        edges: List of edges as (source, target) tuples

    Returns:
        SageGraph instance
    """
    nodes = list(range(num_nodes))
    return SageGraph([nodes, edges], format="vertices_and_edges")


def compute_node_triples(data: Data) -> List[Tuple[int, int, int]]:
    """
    Compute node triples using Block-SPQR decomposition.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        List of node triples (3-tuples of node indices)
    """
    graph = convert_to_sage_graph(data.num_nodes, data.edge_index.T.tolist())
    result = decompose_graph_to_block_spqr(graph)
    return result["node_triples"]


class BSR2FWLData(Data):
    """Data class for 2-FWL GNNs with Block-SPQR decomposition."""

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, time=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def __inc__(self, key: str, value, *args, **kwargs):
        """Define increments for batching."""
        if key == "pair_index":         # [(v_i, v_j)]
            return self.num_nodes
        if key == "pair_x":
            return 0
        if key == "diag_pos":           # position of diagonal pairs
            return self["num_pairs"]
        if key == "triple_index":       # [(pair_i, pair_j, pair_k)]
            return self["num_pairs"]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        """Define concatenation dimensions for batching."""
        if key in ("pair_index", "triple_index"):
            return 1
        if key in ("pair_x", "diag_pos"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class BSR2FWLTransform(BaseTransform):
    """Transform for 2-FWL features with Block-SPQR decomposition."""

    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data: Data) -> BSR2FWLData:
        """
        Transform graph data with 2-FWL features.

        Args:
            data: Input PyTorch Geometric Data object

        Returns:
            BSR2FWLData object with transformed features
        """
        num_nodes = data.num_nodes

        # Compute node triples from Block-SPQR decomposition
        node_triples = compute_node_triples(data)

        # Map node triples to pair ID triples
        pair_id_triples = [
            (a * num_nodes + b, a * num_nodes + c, c * num_nodes + b)
            for a, b, c in node_triples
        ]

        # Extract unique pair IDs and create mapping
        preserved_pair_id = sorted({pair_id for triple in pair_id_triples for pair_id in triple})
        pair_id_to_pos_map = {pair_id: idx for idx, pair_id in enumerate(preserved_pair_id)}

        # Convert to tensors and filter data
        preserved_pair_id_t = torch.tensor(preserved_pair_id, dtype=torch.long)
        preserved_pair_x = data["pair_x"][preserved_pair_id_t, :]
        preserved_pair_index = data["pair_index"][:, preserved_pair_id_t]

        # Map pair ID triples to position-based triples
        pair_idx_triples = torch.tensor(
            [[pair_id_to_pos_map[pair_id] for pair_id in triple] for triple in pair_id_triples],
            dtype=torch.long
        )

        # Compute diagonal positions (self-pairs)
        diag_pos = torch.tensor(
            [pair_id_to_pos_map[v * num_nodes + v] for v in range(num_nodes)],
            dtype=torch.long
        )

        # Create new data object with updated features
        store = dict(data.__dict__["_store"])
        store.update({
            "num_pairs": preserved_pair_x.size(0),
            "pair_x": preserved_pair_x,
            "pair_index": preserved_pair_index,
            "diag_pos": diag_pos,
            "triple_index": pair_idx_triples.t(),
        })

        return BSR2FWLData(**store)


def run_tests():
    """Test Block-SPQR decomposition with a sample graph."""
    print("Testing compute_2fwl_bsr.py ...")

    nodes = list(range(10))
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
        (9, 0), (8, 1), (6, 1),
    ]

    nx_graph = NetworkXGraph()
    nx_graph.add_nodes_from(nodes)
    nx_graph.add_edges_from(edges)

    sage_graph = SageGraph(nx_graph)

    stats = {}
    result = decompose_graph_to_block_spqr(sage_graph, stats)

    print("Statistics:", stats)
    print(f"Number of node triples: {len(result['node_triples'])}")
    print(f"Number of components: {len(result['connected_components'])}")
    print(f"Number of blocks: {len(result['block_components'])}")


if __name__ == "__main__":
    run_tests()
