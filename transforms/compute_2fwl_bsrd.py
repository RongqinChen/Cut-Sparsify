"""
Block-SPQR decomposition and 2-FWL transform for graph neural networks.

This module provides functionality for decomposing graphs using Block-SPQR
decomposition and transforming graph data for 2-FWL (Folklore Weisfeiler-Leman) GNNs.
"""

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from networkx import Graph as NetworkXGraph
from sage.all import Graph as SageGraph
from sage.graphs.connectivity import blocks_and_cuts_tree as compute_block_cut_tree
from sage.graphs.connectivity import spqr_tree as compute_spqr_tree
from sage.graphs.distances_all_pairs import floyd_warshall
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


# Constants
MAX_DISTANCE = 32760
SPQR_COMPONENT_TYPES = ['S', 'P', 'Q', 'R']


def decompose_graph_to_block_spqr(
    graph: SageGraph,
    stats: Optional[Dict] = None
) -> Tuple[Dict, np.ndarray]:
    """
    Decompose a graph into Block-SPQR components.

    This function performs a hierarchical decomposition:
    1. Splits the graph into connected components
    2. Decomposes each component into biconnected blocks
    3. Further decomposes blocks into SPQR tree components

    Args:
        graph: A SageGraph instance to decompose
        stats: Optional dictionary to store decomposition statistics

    Returns:
        A tuple containing:
        - Dictionary with vertices, edges, components, and node triples
        - Pairwise distance matrix (numpy array)
    """
    num_nodes = graph.num_verts()
    num_edges = graph.num_edges()

    # Initialize statistics tracking
    if stats is not None:
        stats.update({
            "num_nodes": num_nodes,
            "num_edges": num_edges * 2,  # Bidirectional edges
            "component_sizes": [],
            "block_sizes": [],
            **{f"{comp_type}_sizes": [] for comp_type in SPQR_COMPONENT_TYPES}
        })

    # Extract connected components
    component_node_lists = graph.connected_components(sort=False)
    components = [graph.subgraph(nodes) for nodes in component_node_lists]

    if stats is not None:
        stats["component_sizes"].extend([len(c.vertices()) for c in components])

    # Create bidirectional edge list
    edges = [(u, v) for u, v, _ in graph.edges()]
    edges.extend([(v, u) for u, v in edges])

    # Initialize result structure
    result = {
        "vertices": list(range(num_nodes)),
        "edges": edges,
        "connected_components": component_node_lists,
        "block_components": [],
        **{f"{comp_type}_components": [] for comp_type in SPQR_COMPONENT_TYPES}
    }

    # Initialize pairwise distance matrix with maximum values
    pair_dist_mat = np.full((num_nodes, num_nodes), MAX_DISTANCE, dtype=np.int16)
    distance_dict = floyd_warshall(graph, paths=False, distances=True)
    for u, neighbors in distance_dict.items():
        for v, dist in neighbors.items():
            # Update with minimum distance (symmetric)
            pair_dist_mat[v, u] = pair_dist_mat[u, v] = dist

    # Decompose each connected component
    for component in components:
        _decompose_component(component, pair_dist_mat, result, stats)

    # Generate node triples from decomposition
    triples = _generate_node_triples(result, num_nodes)
    result["node_triples"] = triples
    return result, pair_dist_mat


def _decompose_component(
    component: SageGraph,
    pair_dist_mat: np.ndarray,
    result: Dict,
    stats: Optional[Dict] = None,
) -> None:
    """
    Decompose a connected component into biconnected blocks and SPQR components.

    Args:
        component: A connected component as SageGraph
        pair_dist_mat: Pairwise distance matrix to update
        result: Dictionary to store decomposition results
        stats: Optional dictionary to store statistics
    """
    block_tree = compute_block_cut_tree(component)

    # Extract biconnected blocks (filter out cut vertices and small blocks)
    blocks = [
        component.subgraph(nodes)
        for node_type, nodes in block_tree.vertices()
        if node_type == "B" and len(nodes) > 2
    ]
    
    block_nodes = [block.vertices() for block in blocks]
    result["block_components"].extend(block_nodes)

    if stats is not None:
        stats["block_sizes"].extend([len(nodes) for nodes in block_nodes])

    # Decompose each block into SPQR components
    for block in blocks:
        _decompose_block(block, pair_dist_mat, result, stats)


def _decompose_block(
    block: SageGraph,
    pair_dist_mat: np.ndarray,
    result: Dict,
    stats: Optional[Dict] = None
) -> None:
    """
    Decompose a biconnected block into SPQR tree components.

    SPQR tree components include:
    - S (Series): edges in series
    - P (Parallel): edges in parallel
    - Q (triconnected): single edges
    - R (Rigid): triconnected components

    Args:
        block: A biconnected block as SageGraph
        pair_dist_mat: Pairwise distance matrix to update
        result: Dictionary to store SPQR component results
        stats: Optional dictionary to store component size statistics
    """
    spqr = compute_spqr_tree(block)

    for comp_type, comp_graph in spqr.vertices():
        nodes = list(comp_graph.vertices())
        result[f"{comp_type}_components"].append(nodes)

        # Compute and update pairwise distances within component
        distance_dict = floyd_warshall(comp_graph, paths=False, distances=True)
        for u, neighbors in distance_dict.items():
            for v, dist in neighbors.items():
                # Update with minimum distance (symmetric)
                pair_dist_mat[u, v] = min(dist, pair_dist_mat[u, v])
                pair_dist_mat[v, u] = pair_dist_mat[u, v]

        if stats is not None:
            stats[f"{comp_type}_sizes"].append(len(nodes))


def _generate_node_triples(result: Dict, num_nodes: int) -> List[Tuple[int, int, int]]:
    """
    Generate node triples from decomposition results.

    Generates triples from:
    1. S and R components (all permutations of 3 nodes)
    2. Edges (creating specific triple patterns)
    3. Self-loops (diagonal triples)

    Args:
        result: Decomposition result dictionary
        num_nodes: Total number of nodes in the graph

    Returns:
        List of node triples as 3-tuples
    """
    triples = []

    # Add triples from S and R components
    for nodes in (result["S_components"] + result["R_components"]):
        triples.extend(permutations(nodes, 3))

    # Add edge-based triples
    for u, v in result["edges"]:
        triples.extend([(u, v, v), (u, u, v), (u, v, u)])

    # Add self-loop triples
    triples.extend([(v, v, v) for v in range(num_nodes)])

    return triples


def convert_to_sage_graph(num_nodes: int, edges: List[Tuple[int, int]]) -> SageGraph:
    """
    Convert an edge list to a SAGE graph.

    Args:
        num_nodes: Number of nodes in the graph
        edges: List of edges as (source, target) tuples

    Returns:
        SageGraph instance
    """
    nodes = list(range(num_nodes))
    return SageGraph([nodes, edges], format="vertices_and_edges")


def compute_node_triples(data: Data) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """
    Compute node triples using Block-SPQR decomposition.

    Args:
        data: PyTorch Geometric Data object containing graph structure

    Returns:
        A tuple containing:
        - List of node triples (3-tuples of node indices)
        - Pairwise distance matrix
    """
    graph = convert_to_sage_graph(data.num_nodes, data.edge_index.T.tolist())
    result, pair_dist_mat = decompose_graph_to_block_spqr(graph)
    return result["node_triples"], pair_dist_mat


class BSRD2FWLData(Data):
    """
    Data class for 2-FWL GNNs with Block-SPQR decomposition and distance constraints.

    This class extends PyTorch Geometric's Data class to support batching
    of pair-based and triple-based features.
    """

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, time=None, **kwargs):
        """Initialize BSRD2FWLData with standard graph attributes."""
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def __inc__(self, key: str, value, *args, **kwargs):
        """
        Define increment values for batching different attribute types.

        Args:
            key: Attribute name
            value: Attribute value

        Returns:
            Increment value for batching
        """
        if key == "pair_index":  # Node pair indices [(v_i, v_j)]
            return self.num_nodes
        if key == "pair_x":  # Pair features (no increment needed)
            return 0
        if key == "diag_pos":  # Diagonal pair positions
            return self["num_pairs"]
        if key == "triple_index":  # Triple indices [(pair_i, pair_j, pair_k)]
            return self["num_pairs"]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        """
        Define concatenation dimensions for batching different attribute types.

        Args:
            key: Attribute name
            value: Attribute value

        Returns:
            Dimension along which to concatenate
        """
        if key in ("pair_index", "triple_index"):
            return 1  # Concatenate along column dimension
        if key in ("pair_x", "diag_pos"):
            return 0  # Concatenate along row dimension
        return super().__cat_dim__(key, value, *args, **kwargs)


class BSRD2FWLTransform(BaseTransform):
    """
    Transform for generating 2-FWL features with Block-SPQR decomposition.

    This transform computes node triples based on Block-SPQR decomposition
    and filters them by distance threshold, then constructs pair-based features.
    """

    def __init__(self, threshold: float, **kwargs):
        """
        Initialize the transform.

        Args:
            threshold: Maximum distance threshold for including node pairs
        """
        super().__init__()
        self.threshold = threshold

    def forward(self, data: Data) -> BSRD2FWLData:
        """
        Transform graph data to include 2-FWL features.

        Args:
            data: Input PyTorch Geometric Data object

        Returns:
            BSRD2FWLData object with transformed features including:
            - pair_x: Features for node pairs
            - pair_index: Indices of node pairs
            - triple_index: Indices of node triples
            - diag_pos: Positions of diagonal (self) pairs
        """
        num_nodes = data.num_nodes

        # Compute node triples from Block-SPQR decomposition
        node_triples, pair_dist_mat = compute_node_triples(data)

        # Convert node triples to pair ID triples with distance filtering
        pair_id_triples = self._compute_filtered_pair_triples(
            node_triples, pair_dist_mat, num_nodes
        )

        # Extract unique pair IDs and create position mapping
        preserved_pair_ids = sorted(
            {pair_id for triple in pair_id_triples for pair_id in triple}
        )
        pair_id_to_pos = {pair_id: idx for idx, pair_id in enumerate(preserved_pair_ids)}

        # Filter and convert pair data to tensors
        preserved_pair_id_tensor = torch.tensor(preserved_pair_ids, dtype=torch.long)
        preserved_pair_x = data["pair_x"][preserved_pair_id_tensor, :]
        preserved_pair_index = data["pair_index"][:, preserved_pair_id_tensor]

        # Map pair ID triples to position-based indices
        pair_idx_triples = torch.tensor(
            [[pair_id_to_pos[pair_id] for pair_id in triple] 
             for triple in pair_id_triples],
            dtype=torch.long
        )

        # Compute diagonal positions (self-pairs: v_i, v_i)
        diag_pos = torch.tensor(
            [pair_id_to_pos[v * num_nodes + v] for v in range(num_nodes)],
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

        return BSRD2FWLData(**store)

    def _compute_filtered_pair_triples(
        self, 
        node_triples: List[Tuple[int, int, int]], 
        pair_dist_mat: np.ndarray, 
        num_nodes: int
    ) -> List[Tuple[int, int, int]]:
        """
        Convert node triples to pair ID triples with distance filtering.

        Args:
            node_triples: List of node triples (a, b, c)
            pair_dist_mat: Pairwise distance matrix
            num_nodes: Total number of nodes

        Returns:
            List of pair ID triples that satisfy distance constraints
        """
        pair_id_triples = []
        threshold = self.threshold

        for a, b, c in node_triples:
            # Check if all pairs in triple satisfy distance threshold
            if (pair_dist_mat[a, b] < threshold and 
                pair_dist_mat[a, c] < threshold and 
                pair_dist_mat[c, b] < threshold):
                
                # Convert to pair IDs: (a,b), (a,c), (c,b)
                pair_id_triple = (
                    a * num_nodes + b,
                    a * num_nodes + c,
                    c * num_nodes + b
                )
                pair_id_triples.append(pair_id_triple)

        return pair_id_triples


def run_tests():
    """Test Block-SPQR decomposition with a sample graph."""
    print("Testing Block-SPQR decomposition...")

    # Create a sample graph
    nodes = list(range(10))
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), 
        (6, 7), (7, 8), (8, 9), (9, 0), (8, 1), (6, 1),
    ]

    # Convert to SageGraph via NetworkX
    nx_graph = NetworkXGraph()
    nx_graph.add_nodes_from(nodes)
    nx_graph.add_edges_from(edges)
    sage_graph = SageGraph(nx_graph)

    # Perform decomposition with statistics tracking
    stats = {}
    result = decompose_graph_to_block_spqr(sage_graph, stats)

    # Print results
    print("\n=== Decomposition Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Decomposition Results ===")
    print(f"Number of node triples: {len(result['node_triples'])}")
    print(f"Number of connected components: {len(result['connected_components'])}")
    print(f"Number of blocks: {len(result['block_components'])}")
    
    for comp_type in SPQR_COMPONENT_TYPES:
        comp_count = len(result[f"{comp_type}_components"])
        print(f"Number of {comp_type} components: {comp_count}")


if __name__ == "__main__":
    run_tests()
