"""

Block-SPQR Tree Decomposition with Distance Filtering for 2-FWL Graph Neural Networks.

This module implements a graph transformation that combines Block-SPQR tree decomposition
with distance-based filtering to accelerate 2-dimensional Folklore Weisfeiler-Leman (2-FWL)
graph neural networks.
"""

from itertools import permutations
from typing import List, Tuple, Dict

import numpy as np
import torch
from sage.all import Graph as SageGraph
from sage.graphs.connectivity import blocks_and_cuts_tree, spqr_tree
from sage.graphs.distances_all_pairs import floyd_warshall
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


MAX_DISTANCE = 32760


def compute_distances(graph: SageGraph) -> np.ndarray:
    """Compute all-pairs shortest path distances."""
    n = graph.num_verts()
    dist = np.full((n, n), MAX_DISTANCE, dtype=np.int16)

    for u, neighbors in floyd_warshall(graph, paths=False, distances=True).items():
        for v, d in neighbors.items():
            dist[u, v] = dist[v, u] = d

    return dist


def get_sr_components(graph: SageGraph) -> List[SageGraph]:
    """Extract Series (S) and Rigid (R) components from SPQR tree."""
    sr_components = []

    for component_nodes in graph.connected_components(sort=False):
        component = graph.subgraph(component_nodes)
        block_tree = blocks_and_cuts_tree(component)

        for node_type, nodes in block_tree.vertices():
            if node_type == "B" and len(nodes) > 2:
                block = component.subgraph(nodes)
                tree = spqr_tree(block)

                for comp_type, comp_graph in tree.vertices():
                    if comp_type in ('S', 'R'):
                        sr_components.append(comp_graph)

    return sr_components


def update_distance_in_sr_components(dist: np.ndarray, sr_components: List[SageGraph]) -> np.ndarray:
    """Refine distance matrix using paths within SR components."""
    for comp_graph in sr_components:
        for u, neighbors in floyd_warshall(comp_graph, paths=False, distances=True).items():
            for v, d in neighbors.items():
                dist[u, v] = dist[v, u] = min(dist[u, v], d)

    return dist


def get_sr_triples(sr_components: List[SageGraph]) -> List[Tuple[int, int, int]]:
    """Generate ordered triples from SR component nodes."""
    triples = []

    for comp_graph in sr_components:
        comp_nodes = list(comp_graph.vertices())
        triples.extend(permutations(comp_nodes, 3))

    return triples


def get_edge_triples(graph: SageGraph) -> List[Tuple[int, int, int]]:
    """Generate triples from graph edges."""
    triples = []
    for u, v, _ in graph.edges():
        triples.extend([(u, v, v), (u, u, v), (u, v, u)])
        triples.extend([(v, u, u), (v, v, u), (v, u, v)])
    return triples


def get_self_triples(n: int) -> List[Tuple[int, int, int]]:
    """Generate self-loop triples for all nodes."""
    return [(i, i, i) for i in range(n)]


def build_graph_from_data(data: Data) -> SageGraph:
    """Convert PyTorch Geometric Data to SageMath Graph."""
    edges = data.edge_index.T.tolist()
    return SageGraph([list(range(data.num_nodes)), edges], format="vertices_and_edges")


def filter_triples_by_distance(
    node_triples: List[Tuple[int, int, int]],
    dist: np.ndarray,
    threshold: int
) -> List[Tuple[int, int, int]]:
    """Filter triples where all pairwise distances are below threshold."""
    filtered = []
    for a, b, c in node_triples:
        if dist[a, b] < threshold and dist[a, c] < threshold and dist[c, b] < threshold:
            filtered.append((a, b, c))
    return filtered


def convert_triples_to_pairs(triples: List[Tuple[int, int, int]], n: int) -> List[Tuple[int, int, int]]:
    """Convert node triples to pair-based representation."""
    return [(a * n + b, a * n + c, c * n + b) for a, b, c in triples]


def build_pair_mapping(pair_triples: List[Tuple[int, int, int]]) -> Tuple[List[int], Dict[int, int]]:
    """Create mapping between pair IDs and indices."""
    unique_pairs = sorted({p for triple in pair_triples for p in triple})
    pair_map = {p: i for i, p in enumerate(unique_pairs)}
    return unique_pairs, pair_map


def build_output_tensors(
    data: Data,
    unique_pairs: List[int],
    pair_map: Dict[int, int],
    pair_triples: List[Tuple[int, int, int]],
    n: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct tensors for 2-FWL message passing."""
    pair_ids = torch.tensor(unique_pairs, dtype=torch.long)
    pair_x = data["pair_x"][pair_ids, :]
    pair_index = data["pair_index"][:, pair_ids]

    triple_index = torch.tensor(
        [[pair_map[p] for p in triple] for triple in pair_triples],
        dtype=torch.long
    ).t()

    diag_pos = torch.tensor([pair_map[i * n + i] for i in range(n)], dtype=torch.long)

    return pair_x, pair_index, triple_index, diag_pos


class BSRD2FWLData(Data):
    """Custom Data class for 2-FWL with Block-SPQR decomposition."""

    def __inc__(self, key: str, value, *args, **kwargs):
        """Define increment behavior for batching."""
        if key == "pair_index":
            return self.num_nodes
        if key == "pair_x":
            return 0
        if key in ("diag_pos", "triple_index"):
            return self["num_pairs"]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        """Define concatenation dimension for batching."""
        if key in ("pair_index", "triple_index"):
            return 1
        if key in ("pair_x", "diag_pos"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class BSRD2FWLTransform(BaseTransform):
    """Transform for computing 2-FWL features with Block-SPQR decomposition."""

    def __init__(self, threshold: int, **kwargs):
        super().__init__()
        self.threshold = threshold

    def forward(self, data: Data) -> BSRD2FWLData:
        """Transform input graph to 2-FWL representation."""
        graph = build_graph_from_data(data)
        n = data.num_nodes

        sr_components = get_sr_components(graph)

        dist = compute_distances(graph)
        dist = update_distance_in_sr_components(dist, sr_components)

        spqr_triples = get_sr_triples(sr_components)
        edge_triples = get_edge_triples(graph)
        self_triples = get_self_triples(n)
        node_triples = spqr_triples + edge_triples + self_triples

        filtered_triples = filter_triples_by_distance(node_triples, dist, self.threshold)

        pair_triples = convert_triples_to_pairs(filtered_triples, n)

        unique_pairs, pair_map = build_pair_mapping(pair_triples)

        pair_x, pair_index, triple_index, diag_pos = build_output_tensors(
            data, unique_pairs, pair_map, pair_triples, n
        )

        output = dict(data._store)
        output.update({
            "num_pairs": len(unique_pairs),
            "pair_x": pair_x,
            "pair_index": pair_index,
            "diag_pos": diag_pos,
            "triple_index": triple_index,
        })

        return BSRD2FWLData(**output)



def run_tests():
    """Test Block-SPQR decomposition with a sample graph."""
    print("Testing Block-SPQR decomposition...")

    # Create a sample graph
    nodes = list(range(10))
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), 
        (6, 7), (7, 8), (8, 9), (9, 0), (8, 1), (6, 1),
    ]

    # Create SageGraph directly
    sage_graph = SageGraph([nodes, edges], format="vertices_and_edges")
    
    print(f"Graph has {sage_graph.num_verts()} vertices and {sage_graph.num_edges()} edges")
    
    # Test distance computation
    print("\nTesting distance computation...")
    dist = compute_distances(sage_graph)
    print(f"Distance matrix shape: {dist.shape}")
    print(f"Sample distances: dist[0,1]={dist[0,1]}, dist[0,5]={dist[0,5]}")
    
    # Test SR component extraction
    print("\nTesting SR component extraction...")
    sr_components = get_sr_components(sage_graph)
    print(f"Found {len(sr_components)} SR components")
    for i, comp in enumerate(sr_components):
        print(f"  Component {i}: {comp.num_verts()} vertices, {comp.num_edges()} edges")
    
    # Test distance refinement
    print("\nTesting distance refinement in SR components...")
    refined_dist = update_distance_in_sr_components(dist.copy(), sr_components)
    print(f"Distance matrix updated")
    
    # Test triple generation
    print("\nTesting triple generation...")
    sr_triples = get_sr_triples(sr_components)
    edge_triples = get_edge_triples(sage_graph)
    self_triples = get_self_triples(len(nodes))
    
    print(f"SR triples: {len(sr_triples)}")
    print(f"Edge triples: {len(edge_triples)}")
    print(f"Self triples: {len(self_triples)}")
    
    all_triples = sr_triples + edge_triples + self_triples
    print(f"Total triples: {len(all_triples)}")
    
    # Test triple filtering
    print("\nTesting triple filtering with threshold=5...")
    filtered = filter_triples_by_distance(all_triples, refined_dist, threshold=5)
    print(f"Filtered triples: {len(filtered)}")
    
    # Test pair conversion
    print("\nTesting pair conversion...")
    pair_triples = convert_triples_to_pairs(filtered, len(nodes))
    print(f"Pair triples: {len(pair_triples)}")
    
    # Test pair mapping
    print("\nTesting pair mapping...")
    unique_pairs, pair_map = build_pair_mapping(pair_triples)
    print(f"Unique pairs: {len(unique_pairs)}")
    
    # Test full transform with PyG Data
    print("\nTesting full BSRD2FWL transform...")
    num_nodes = len(nodes)
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    x = torch.randn(num_nodes, 16)
    
    # Create pair_x and pair_index for the test
    num_pairs = num_nodes * num_nodes
    pair_x = torch.randn(num_pairs, 32)
    pair_index = torch.stack([
        torch.arange(num_nodes).repeat_interleave(num_nodes),
        torch.arange(num_nodes).repeat(num_nodes)
    ])
    
    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        pair_x=pair_x,
        pair_index=pair_index
    )
    
    transform = BSRD2FWLTransform(threshold=5)
    transformed_data = transform(data)
    
    print(f"Transformed data:")
    print(f"  num_pairs: {transformed_data.num_pairs}")
    print(f"  pair_x shape: {transformed_data.pair_x.shape}")
    print(f"  pair_index shape: {transformed_data.pair_index.shape}")
    print(f"  triple_index shape: {transformed_data.triple_index.shape}")
    print(f"  diag_pos shape: {transformed_data.diag_pos.shape}")
    
    print("\nAll tests completed successfully!")


if __name__ == "__main__":
    run_tests()
