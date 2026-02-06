from itertools import permutations
from typing import Dict, List, Optional

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
    """Decompose graph into Block-SPQR components."""
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

    edges = [(u, v) for u, v, _ in graph.edges()]
    edges += [(v, u) for u, v in edges]

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

    triples = []
    for nodes in (result["S_components"] + result["R_components"]):
        triples.extend(permutations(nodes, 3))

    for u, v in result["edges"]:
        triples.extend([(u, v, v), (u, u, v), (u, v, u)])

    for v in range(num_nodes):
        triples.extend([(v, v, v), (v, v, v), (v, v, v)])

    result["node_triples"] = triples
    return result


def decompose_component(
    component: SageGraph,
    result: Dict,
    stats: Optional[Dict] = None
) -> None:
    """Decompose connected component into blocks and SPQR components."""
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
    """Decompose biconnected block into SPQR components."""
    spqr = compute_spqr_tree(block)

    for comp_type, comp_graph in spqr.vertices():
        nodes = list(comp_graph.vertices())
        result[comp_type + "_components"].append(nodes)

    if stats is not None:
        for comp_type, comp_graph in spqr.vertices():
            stats[comp_type + "_sizes"].append(len(comp_graph.vertices()))


def convert_to_sage_graph(num_nodes: int, edges: List) -> SageGraph:
    """Convert edge list to SAGE graph."""
    nodes = list(range(num_nodes))
    return SageGraph([nodes, edges], format="vertices_and_edges")


def compute_pair_triples(data: Data) -> List:
    """Compute node triples using Block-SPQR decomposition."""
    graph = convert_to_sage_graph(data.num_nodes, data.edge_index.T.tolist())
    result = decompose_graph_to_block_spqr(graph)
    num_nodes = data.num_nodes
    pair_triples = [
        (a * num_nodes + b, a * num_nodes + c, c * num_nodes + b)
        for a, b, c in result["node_triples"]
    ]
    pair_triples = list(sorted(pair_triples))
    return pair_triples


class BSR2FWLData(Data):
    """Data class for 2-FWL GNNs with Block-SPQR decomposition."""

    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,
                 pos=None, time=None, **kwargs):
        super().__init__(x, edge_index, edge_attr, y, pos, time, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == "pair_index":         # [(v_i, v_j)]
            return self.num_nodes
        if key == "pair_x":
            return 0
        if key == "diag_pos":           # position of diagonal pairs
            return self.num_nodes**2
        if key == "triple_index":       # [(pair_i, pair_j, pair_k)]
            return self.num_nodes**2
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
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
        """Transform graph data with 2-FWL features."""
        num_nodes = data.num_nodes
        triples = compute_pair_triples(data)

        triple_tensor = torch.tensor(triples, dtype=torch.long).t().contiguous()

        indices = torch.arange(num_nodes, dtype=torch.long)
        diag_pos = indices * num_nodes + indices

        store = dict(data.__dict__["_store"])
        store.update({
            "diag_pos": diag_pos,
            "triple_index": triple_tensor,
        })

        if "pair_index" not in store:
            adj = torch.ones((num_nodes, num_nodes), dtype=torch.short)
            store["pair_index"] = adj.nonzero(as_tuple=False).t().contiguous()

        return BSR2FWLData(**store)


def run_tests():
    """Test Block-SPQR decomposition."""
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
    print("Decomposition result:", result)


if __name__ == "__main__":
    run_tests()
