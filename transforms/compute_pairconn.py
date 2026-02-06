"""
Module for computing pairwise connectivity in graphs using SPQR tree decomposition.

This module provides functions to analyze graph connectivity by decomposing graphs
into their connected components, blocks, and SPQR tree components.
"""

from itertools import combinations, permutations
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
from networkx import Graph as NX_Graph
from sage.all import Graph as SAGE_Graph
from sage.graphs.connectivity import blocks_and_cuts_tree, spqr_tree


def _update_connectivity_for_new_edges(edges: list, conn_mat: np.ndarray) -> None:
    """
    Update connectivity matrix by decrementing values for newly added edges.
    
    Args:
        edges: List of edges, where each edge is a tuple (u, v, label)
        conn_mat: Connectivity matrix to update
    """
    for edge in edges:
        if edge[2] is not None and 'new' in edge[2]:
            u, v = edge[0], edge[1]
            conn_mat[u, v] -= 1
            conn_mat[v, u] -= 1


def compute_pairconn_in_spqr_comp(
    label: str, 
    spqr_comp: SAGE_Graph, 
    conn_mat: np.ndarray
) -> None:
    """
    Compute pairwise connectivity within an SPQR tree component.
    
    Args:
        label: Component type - 'S' (series), 'P' (parallel), 'Q' (trivial), or 'R' (rigid)
        spqr_comp: SPQR tree component graph
        conn_mat: Connectivity matrix to update (modified in-place)
    """
    nodes = spqr_comp.vertices()
    edges = spqr_comp.edges()
    print(f"{label}: nodes={nodes}, edges={edges}")

    if label == "S":
        # S-component: cycle graph with 3+ vertices
        # All node pairs have connectivity 2
        for u, v in permutations(nodes, 2):
            conn_mat[u, v] += 2
        
        # Adjust for newly added edges
        _update_connectivity_for_new_edges(edges, conn_mat)

    elif label in ("P", "Q"):
        # P-component: dipole graph (multigraph with 2 vertices, 3+ edges)
        # Q-component: single real edge
        for edge in edges:
            if edge[2] is None:
                u, v = edge[0], edge[1]
                conn_mat[u, v] += 1
                conn_mat[v, u] += 1

    elif label == "R":
        # R-component: 3-connected graph (not cycle or dipole)
        nxg = NX_Graph()
        nxg.add_nodes_from(nodes)
        nxg.add_edges_from([(edge[0], edge[1]) for edge in edges])
        
        # Compute all-pairs node connectivity
        connectivity_dict = nx.connectivity.all_pairs_node_connectivity(nxg)
        
        # Update connectivity matrix
        # Note: connectivity_dict contains both (u,v) and (v,u), so we add once
        for u, neighbors in connectivity_dict.items():
            for v, conn_value in neighbors.items():
                conn_mat[u, v] += conn_value
        
        # Adjust for newly added edges
        _update_connectivity_for_new_edges(edges, conn_mat)


def compute_pairconn_in_block_comp(
    block_comp: SAGE_Graph, 
    conn_mat: np.ndarray, 
    cnt_dict: Optional[Dict] = None
) -> None:
    """
    Compute pairwise connectivity within a block component.
    
    Args:
        block_comp: Block component graph
        conn_mat: Connectivity matrix to update (modified in-place)
        cnt_dict: Optional dictionary to store component size statistics
    """
    # Build SPQR tree and process each component
    spqr_decomposition = spqr_tree(block_comp)
    
    for label, spqr_comp in spqr_decomposition.vertices():
        compute_pairconn_in_spqr_comp(label, spqr_comp, conn_mat)
    
    # Record component sizes if tracking is enabled
    if cnt_dict is not None:
        for label, spqr_comp in spqr_decomposition.vertices():
            size = len(spqr_comp.vertices())
            cnt_dict[f"{label}_sizes"].append(size)
    
    # Ensure minimum connectivity of 2 within the block
    block_nodes = block_comp.vertices()
    for u, v in combinations(block_nodes, 2):
        conn_mat[u, v] = max(conn_mat[u, v], 2)
        conn_mat[v, u] = conn_mat[u, v]


def compute_pair_conn_in_comp(
    comp: SAGE_Graph, 
    conn_mat: np.ndarray, 
    cnt_dict: Optional[Dict] = None
) -> None:
    """
    Compute pairwise connectivity within a connected component.
    
    Args:
        comp: Connected component graph
        conn_mat: Connectivity matrix to update (modified in-place)
        cnt_dict: Optional dictionary to store component size statistics
    """
    # Build block-cut tree
    bc_tree = blocks_and_cuts_tree(comp)
    
    # Extract non-trivial blocks (size > 2)
    block_comp_list = [
        comp.subgraph(bc[1])
        for bc in bc_tree.vertices()
        if bc[0] == "B" and len(bc[1]) > 2
    ]
    
    # Process each block
    for block_comp in block_comp_list:
        compute_pairconn_in_block_comp(block_comp, conn_mat, cnt_dict)
    
    # Record block sizes if tracking is enabled
    if cnt_dict is not None:
        block_sizes = [len(block.vertices()) for block in block_comp_list]
        cnt_dict["B_sizes"].extend(block_sizes)
    
    # Ensure minimum connectivity of 1 within the component
    comp_nodes = comp.vertices()
    for u, v in combinations(comp_nodes, 2):
        conn_mat[u, v] = max(conn_mat[u, v], 1)
        conn_mat[v, u] = conn_mat[u, v]


def compute_pair_conn(
    sage_graph: SAGE_Graph, 
    cnt_dict: Optional[Dict] = None
) -> np.ndarray:
    """
    Compute pairwise connectivity for all node pairs in a graph.
    
    Args:
        sage_graph: Input graph (SAGE Graph object)
        cnt_dict: Optional dictionary to store decomposition statistics
    
    Returns:
        Flattened connectivity matrix as 1D numpy array
    """
    num_nodes = sage_graph.num_verts()
    num_edges = sage_graph.num_edges()

    # Initialize statistics dictionary
    if cnt_dict is not None:
        cnt_dict.update({
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "C_sizes": [],  # Component sizes
            "B_sizes": [],  # Block sizes
            "S_sizes": [],  # Series component sizes
            "P_sizes": [],  # Parallel component sizes
            "Q_sizes": [],  # Trivial component sizes
            "R_sizes": [],  # Rigid component sizes
        })

    # Initialize connectivity matrix
    conn_mat = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    # Split graph into connected components
    nodes_list: List[List[int]] = sage_graph.connected_components(sort=False)
    comp_list = [sage_graph.subgraph(nodes) for nodes in nodes_list]
    
    if cnt_dict is not None:
        comp_sizes = [len(comp.vertices()) for comp in comp_list]
        cnt_dict["C_sizes"].extend(comp_sizes)

    # Compute connectivity for each component
    for comp in comp_list:
        compute_pair_conn_in_comp(comp, conn_mat, cnt_dict)

    # Set self-loop connectivity (maximum connectivity to other nodes)
    for v in range(num_nodes):
        conn_mat[v, v] = conn_mat[v, :].max()

    return conn_mat.flatten()


def _create_test_graph_and_visualize(
    nodes: range,
    edges: List[tuple],
    filepath: str,
    node_labels: Optional[Dict] = None,
    node_size: int = 20
) -> SAGE_Graph:
    """
    Helper function to create, visualize, and return a test graph.
    
    Args:
        nodes: Node range
        edges: List of edges
        filepath: Output file path for visualization
        node_labels: Optional custom node labels
        node_size: Node size for visualization
    
    Returns:
        SAGE Graph object
    """
    from .graph_drawio import draw_graph_drawio
    
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(nodes)
    nx_graph.add_edges_from(edges)
    
    pos = nx.kamada_kawai_layout(nx_graph)
    pos = {key: 200 * val for key, val in pos.items()}
    
    draw_graph_drawio(
        nx_graph, 
        pos, 
        node_labels=node_labels,
        node_size=node_size,
        filepath=filepath
    )
    
    return SAGE_Graph(nx_graph)


def _print_connectivity_matrix(pair_conn: np.ndarray, num_nodes: int) -> None:
    """
    Print connectivity matrix in a readable format.
    
    Args:
        pair_conn: Flattened connectivity matrix
        num_nodes: Number of nodes in the graph
    """
    print(f"Connectivity matrix shape: {pair_conn.shape}")
    for i in range(num_nodes):
        print(f"{i:02d}: ", end='')
        for j in range(i + 1):
            print(f"{j:02d}={pair_conn[i * num_nodes + j]}", end='  ')
        print()


def test1():
    """Test case 1: Basic graph structure."""
    edges = [(0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5)]
    sage_graph = _create_test_graph_and_visualize(
        range(6), edges, 'images/local_conn_test1.drawio'
    )
    
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def test1_1():
    """Test case 1.1: Graph with backward edge."""
    edges = [
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5), (2, 0)
    ]
    sage_graph = _create_test_graph_and_visualize(
        range(6), edges, 'images/local_conn_test1_1.drawio'
    )
    
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def test2():
    """Test case 2: Graph with additional edges."""
    edges = [
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5),
        (0, 4), (1, 2)
    ]
    sage_graph = _create_test_graph_and_visualize(
        range(6), edges, 'images/local_conn_test2.drawio'
    )
    
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def test3():
    """Test case 3: More connected graph."""
    edges = [
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5),
        (0, 4), (1, 2), (3, 4)
    ]
    sage_graph = _create_test_graph_and_visualize(
        range(6), edges, 'images/local_conn_test3.drawio'
    )
    
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def test4():
    """Test case 4: Larger graph structure."""
    edges = [
        (0, 1), (0, 2), (1, 4), (2, 3), (2, 4), (3, 5), (4, 5),
        (0, 4), (1, 2), (3, 4), (4, 6), (2, 7), (6, 7), (2, 6),
        (4, 7), (2, 5),
    ]
    sage_graph = _create_test_graph_and_visualize(
        range(8), edges, 'images/local_conn_test4.drawio'
    )
    
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def dodecahedron():
    """Test case: Dodecahedron graph."""
    from .graph_drawio import draw_graph_drawio
    
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(20))
    nx_graph.add_edges_from([
        (0, 1), (0, 4), (0, 5),
        (1, 2), (1, 6),
        (2, 3), (2, 7),
        (3, 4), (3, 8), (4, 9),
        (5, 12), (5, 13),
        (6, 13), (6, 14),
        (7, 10), (7, 14),
        (8, 10), (8, 11),
        (9, 11), (9, 12),
        (10, 15), (11, 16), (12, 17), (13, 18), (14, 19),
        (15, 16), (15, 19), (16, 17), (17, 18), (18, 19)
    ])
    
    # Create dodecahedron layout
    angles = [np.pi/2 - 2*np.pi/5 * i for i in range(5)]
    cos_vals = [np.cos(angle) for angle in angles]
    sin_vals = [-np.sin(angle) for angle in angles]
    
    pos = {}
    for i in range(5):
        pos[i] = (7 * cos_vals[i], 7 * sin_vals[i])
        pos[i+5] = (5 * cos_vals[i], 5 * sin_vals[i])
        pos[i+10] = (-5 * cos_vals[i], -5 * sin_vals[i])
        pos[i+15] = (-3 * cos_vals[i], -3 * sin_vals[i])
    
    pos = {key: 30 * np.array(val) for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath='images/dodecahedron.drawio')
    
    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def octagon():
    """Test case: Octagon with center node."""
    from .graph_drawio import draw_graph_drawio
    
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(9))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (8, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
    ])
    
    # Create octagon layout
    pos = {0: (0, 0)}
    radius = 30
    for i in range(1, 9):
        angle = 2 * np.pi / 8 * (i - 1)
        pos[i] = (radius * np.cos(angle), radius * np.sin(angle))
    
    node_labels = {i: '' for i in range(9)}
    draw_graph_drawio(
        nx_graph, pos, 
        node_labels=node_labels, 
        node_size=10, 
        filepath='images/octagon.drawio'
    )
    
    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def regular(num_nodes: int):
    """Test case: Regular polygon (cycle graph)."""
    from .graph_drawio import draw_graph_drawio
    
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from([
        (i, (i + 1) % num_nodes) for i in range(num_nodes)
    ])
    
    # Create regular polygon layout
    radius = 30
    pos = {}
    for i in range(num_nodes):
        angle = 2 * np.pi / num_nodes * i + np.pi / 2
        pos[i] = (radius * np.cos(angle), radius * np.sin(angle))
    
    node_labels = {i: '' for i in range(num_nodes)}
    draw_graph_drawio(
        nx_graph, pos, 
        node_labels=node_labels, 
        node_size=10, 
        filepath=f'images/regular_{num_nodes}.drawio'
    )
    
    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def x2():
    """Test case: Star-like graph with additional edges."""
    from .graph_drawio import draw_graph_drawio
    
    num_nodes = 9
    nx_graph = NX_Graph()
    nx_graph.add_nodes_from(range(num_nodes))
    nx_graph.add_edges_from([
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
        (7, 1), (8, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8),
    ])
    
    # Create radial layout
    radius = 30
    pos = {0: (0, 0)}
    for i in range(1, num_nodes):
        angle = 2 * np.pi / (num_nodes - 1) * i + np.pi / 2
        pos[i] = (radius * np.cos(angle), radius * np.sin(angle))
    
    draw_graph_drawio(
        nx_graph, pos, 
        node_size=10, 
        filepath=f'images/x2_{num_nodes}.drawio'
    )
    
    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def x():
    """Test case: Complex cycle structure."""
    edges = [
        (0, 1), (0, 6),
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), 
        (6, 7), (7, 8), (8, 9), (9, 10), (10, 1)
    ]
    sage_graph = _create_test_graph_and_visualize(
        range(11), edges, 'images/x.drawio'
    )
    
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


def x3():
    """Test case: Highly connected graph."""
    edges = [
        (0, 1), (0, 5), (0, 6),
        (1, 2), (1, 6), (1, 7),
        (2, 3), (2, 7),
        (3, 4), (3, 7),
        (4, 5), (4, 6), (4, 7),
        (5, 6),
    ]
    sage_graph = _create_test_graph_and_visualize(
        range(8), edges, 'images/x3.drawio'
    )
    
    cnt_dict = {}
    pair_conn = compute_pair_conn(sage_graph, cnt_dict)
    _print_connectivity_matrix(pair_conn, sage_graph.num_verts())


if __name__ == "__main__":
    # Run all test cases
    test1()
    test1_1()
    test2()
    test3()
    test4()
    dodecahedron()
    octagon()
    regular(6)
    x2()
    x3()
