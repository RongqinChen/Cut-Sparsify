
"""
Block-SPQR decomposition for 2-FWL Graph Neural Networks.
"""

from itertools import permutations
from typing import Dict, List, Optional, Set, Tuple

import torch
from networkx import Graph as NetworkXGraph
from sage.all import Graph as SageGraph
from sage.graphs.connectivity import blocks_and_cuts_tree, spqr_tree
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


def create_bidirectional_edges(graph: SageGraph) -> List[Tuple[int, int]]:
    edges = [(u, v) for u, v, _ in graph.edges()]
    edges.extend([(v, u) for u, v in edges])
    return edges


def decompose_block_to_spqr(block: SageGraph, result: Dict, stats: Optional[Dict] = None) -> None:
    spqr = spqr_tree(block)
    for comp_type, comp_graph in spqr.vertices():
        nodes = list(comp_graph.vertices())
        result[f"{comp_type}_components"].append(nodes)
        if stats is not None:
            stats[f"{comp_type}_sizes"].append(len(nodes))


def decompose_component_to_blocks(component: SageGraph, result: Dict, stats: Optional[Dict] = None) -> None:
    block_tree = blocks_and_cuts_tree(component)
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
        decompose_block_to_spqr(block, result, stats)


def generate_node_triples(result: Dict, num_nodes: int) -> List[Tuple[int, int, int]]:
    triples = []
    for nodes in result["S_components"] + result["R_components"]:
        triples.extend(permutations(nodes, 3))
    for u, v in result["edges"]:
        triples.extend([(u, v, v), (u, u, v), (u, v, u)])
    triples.extend([(v, v, v) for v in range(num_nodes)])
    return triples


def decompose_graph_to_block_spqr(graph: SageGraph, stats: Optional[Dict] = None) -> Dict:
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
    
    edges = create_bidirectional_edges(graph)
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
        decompose_component_to_blocks(component, result, stats)
    
    result["node_triples"] = generate_node_triples(result, num_nodes)
    return result


def convert_to_sage_graph(num_nodes: int, edges: List[Tuple[int, int]]) -> SageGraph:
    nodes = list(range(num_nodes))
    return SageGraph([nodes, edges], format="vertices_and_edges")


def compute_node_triples(data: Data) -> List[Tuple[int, int, int]]:
    graph = convert_to_sage_graph(data.num_nodes, data.edge_index.T.tolist())
    result = decompose_graph_to_block_spqr(graph)
    return result["node_triples"]


def compute_pair_id_triples(node_triples: List[Tuple[int, int, int]], num_nodes: int) -> List[Tuple[int, int, int]]:
    return [(a * num_nodes + b, a * num_nodes + c, c * num_nodes + b) for a, b, c in node_triples]


def extract_unique_pair_ids(pair_id_triples: List[Tuple[int, int, int]]) -> List[int]:
    unique_ids: Set[int] = set()
    for triple in pair_id_triples:
        unique_ids.update(triple)
    return sorted(unique_ids)


def create_pair_id_mapping(preserved_pair_ids: List[int]) -> Dict[int, int]:
    return {pair_id: idx for idx, pair_id in enumerate(preserved_pair_ids)}


def map_pair_triples_to_positions(pair_id_triples: List[Tuple[int, int, int]], pair_id_to_pos: Dict[int, int]) -> torch.Tensor:
    return torch.tensor(
        [[pair_id_to_pos[pair_id] for pair_id in triple] for triple in pair_id_triples],
        dtype=torch.long
    )


def compute_diagonal_positions(num_nodes: int, pair_id_to_pos: Dict[int, int]) -> torch.Tensor:
    return torch.tensor([pair_id_to_pos[v * num_nodes + v] for v in range(num_nodes)], dtype=torch.long)


class BSR2FWLData(Data):
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, time=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "pair_index":
            return self.num_nodes
        if key == "pair_x":
            return 0
        if key == "diag_pos":
            return self["num_pairs"]
        if key == "triple_index":
            return self["num_pairs"]
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):
        if key in ("pair_index", "triple_index"):
            return 1
        if key in ("pair_x", "diag_pos"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


class BSR2FWLTransform(BaseTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data: Data) -> BSR2FWLData:
        num_nodes = data.num_nodes
        
        node_triples = compute_node_triples(data)
        pair_id_triples = compute_pair_id_triples(node_triples, num_nodes)
        preserved_pair_ids = extract_unique_pair_ids(pair_id_triples)
        pair_id_to_pos = create_pair_id_mapping(preserved_pair_ids)
        
        preserved_pair_id_t = torch.tensor(preserved_pair_ids, dtype=torch.long)
        preserved_pair_x = data["pair_x"][preserved_pair_id_t, :]
        preserved_pair_index = data["pair_index"][:, preserved_pair_id_t]
        
        pair_idx_triples = map_pair_triples_to_positions(pair_id_triples, pair_id_to_pos)
        diag_pos = compute_diagonal_positions(num_nodes, pair_id_to_pos)
        
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
    print("Testing compute_2fwl_bsr.py ...")
    
    nodes = list(range(10))
    edges = [
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
        (6, 7), (7, 8), (8, 9), (9, 0), (8, 1), (6, 1),
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
