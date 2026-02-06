from collections import defaultdict
from itertools import combinations, permutations
from typing import Dict, List

import networkx as nx
import numpy as np
from networkx import Graph as NX_Graph
from sage.all import Graph as SAGE_Graph
from sage.graphs.connectivity import blocks_and_cuts_tree, spqr_tree

from .kcvset import KCVSet


def compute_k_blocks(
    sage_graph: SAGE_Graph,
    max_kset_order=None,
    cnt_dict: dict = None,
    conn_mat: np.ndarray = None
):
    num_nodes = sage_graph.num_verts()

    if max_kset_order is not None:
        max_kset_order = min(max_kset_order, num_nodes)
    else:
        max_kset_order = num_nodes

    if max_kset_order is None:
        max_kset_order = num_nodes
    if conn_mat is None:
        conn_mat = np.zeros((num_nodes, num_nodes), dtype=np.int32)

    # Initialize c0set
    c0set = KCVSet(0, set(range(num_nodes)), list())
    kset_list_dict = defaultdict(list)
    kset_list_dict[0] = [c0set]

    # Find connected components (c1sets)
    c1vs_list: List[List[int]] = sage_graph.connected_components(sort=False)
    c1set_list = [KCVSet(1, set(c1vs), list()) for c1vs in c1vs_list]
    c0set.child_list.extend(c1set_list)
    kset_list_dict[1] = c1set_list

    # Find 2-connected components (c2sets)
    c2set_list = _find_2_blocks(sage_graph, c1set_list)
    if len(c2set_list) > 0:
        kset_list_dict[2] = c2set_list

        # Find higher-order blocks if needed
        if max_kset_order > 2:
            kc_pairs_dict: Dict[int, List] = defaultdict(list)  # {k: list of k-connected pairs}
            _find_3_blocks(sage_graph, kset_list_dict, conn_mat, kc_pairs_dict, max_kset_order)
            _find_higher_order_blocks(kc_pairs_dict, kset_list_dict)

    _update_remain_pair_conn(kset_list_dict, conn_mat)

    # Count structures if cnt_dict is provided
    if cnt_dict is not None:
        _count(sage_graph, kset_list_dict, cnt_dict)

    return c0set


def _find_2_blocks(sage_graph, c1set_list):
    c2set_list = []
    for c1set in c1set_list:
        c1_block = sage_graph.subgraph(c1set.verts)
        bc_tree = blocks_and_cuts_tree(c1_block)
        # vertex data (e.g., bc_tree.vertices()[0]): label, nodes
        for bc in bc_tree.vertices():
            if bc[0] == "B" and len(bc[1]) > 2:  # block could be a trivial edge graph
                c2set = KCVSet(2, set(bc[1]), list())
                c1set.child_list.append(c2set)

        c2set_list.extend(c1set.child_list)

    return c2set_list


def _find_3_blocks(sage_graph, kset_list_dict, conn_mat, kc_pairs_dict, max_K):
    for c2set in kset_list_dict[2]:
        c2block = sage_graph.subgraph(c2set.verts)
        a_spqr_tree = spqr_tree(c2block)
        # vertex data (e.g., a_spqr_tree.vertices()[0]): label, spqr_component
        for label, spqr_comp in a_spqr_tree.vertices():
            _compute_conn_in_spqr_comp(
                label, spqr_comp, c2set, conn_mat, kc_pairs_dict, max_K
            )
        if len(c2set.child_list) > 0:
            kset_list_dict[3].extend(c2set.child_list)


def _compute_conn_in_spqr_comp(
    label, spqr_comp, c2set, conn_mat, kc_pairs_dict,
    max_kset_order
):
    nodes = spqr_comp.vertices()
    edges = spqr_comp.edges()  # not directed

    # S: the associated graph is a cycle graph with three or more vertices and edges.
    if label == "S":
        for u, v in permutations(nodes, 2):
            conn_mat[u, v] += 2

        for edge in edges:  # undirected
            if edge[2] is not None and "new" in edge[2]:
                v, u = edge[0], edge[1]
                conn_mat[v, u] -= 1
                conn_mat[u, v] -= 1

    # P: the associated graph is a dipole graph, a multigraph with two vertices and three or more edges.
    # Q: the associated graph has a single real edge.
    elif label == "P" or label == "Q":
        # assert len(nodes) == 2
        u, v = nodes
        conn_mat[u, v] += 1
        conn_mat[v, u] += 1

    # R: the associated graph is a 3-connected graph that is not a cycle or dipole.
    elif label == "R":
        c3set = KCVSet(3, set(nodes), list())
        c2set.child_list.append(c3set)

        # Build NetworkX graph
        nxg = NX_Graph()
        nxg.add_nodes_from(nodes)
        nxg.add_edges_from([(edge[0], edge[1]) for edge in edges])
        ret_dict = nx.connectivity.all_pairs_node_connectivity(nxg)

        # both ret_dict[u][v] and ret_dict[v][u] are included in ret_dict,
        # so we only need to add once
        for u, v_conn in ret_dict.items():
            for v, conn in v_conn.items():
                conn_mat[u, v] += conn

                # update k-connected pairs
                for k in range(4, min(max_kset_order, conn) + 1):
                    kc_pairs_dict[k].append((u, v))

        for edge in edges:
            if edge[2] is not None and "new" in edge[2]:
                v, u = edge[0], edge[1]
                conn_mat[v, u] -= 1
                conn_mat[u, v] -= 1


def _find_higher_order_blocks(kc_pairs_dict, kset_list_dict):
    for k, pairs in kc_pairs_dict.items():
        components = _find_connected_components(pairs)
        k_nodes = [component for component in components if len(component) > k]
        if len(k_nodes) == 0:
            break
        kset_list_dict[k] += [KCVSet(k, vs, list()) for vs in k_nodes]


def _find_connected_components(pairs):
    # Step 1: Build adjacency list for the graph
    graph = defaultdict(set)
    for pair in pairs:
        u, v = pair
        graph[u].add(v)
        graph[v].add(u)

    # Step 2: Find connected components using DFS
    visited = set()
    components = []

    def dfs_iterative(start_node):
        """Perform iterative DFS to find all nodes in the same connected component."""
        stack = [start_node]
        component = set()
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                component.add(node)
                stack.extend(neighbor for neighbor in graph[node] if neighbor not in visited)
        return component

    for node in graph:
        if node not in visited:
            component = dfs_iterative(node)
            components.append(component)

    return components


def _update_remain_pair_conn(kset_list_dict, conn_mat):
    for c2set in kset_list_dict[2]:
        for u, v in combinations(c2set.verts, 2):
            conn_mat[u, v] = max(conn_mat[u, v], 2)
            conn_mat[v, u] = conn_mat[u, v]

    for c1set in kset_list_dict[1]:
        for u, v in combinations(c1set.verts, 2):
            conn_mat[u, v] = max(conn_mat[u, v], 1)
            conn_mat[v, u] = conn_mat[u, v]

    for v in range(conn_mat.shape[0]):
        conn_mat[v, v] = conn_mat[v, :].max()


def _count(sage_graph, kset_list_dict, cnt_dict):
    if cnt_dict is not None:
        cnt_dict['mk'] = len(kset_list_dict) - 1
        cnt_dict["num_nodes"] = sage_graph.num_verts()
        cnt_dict['num_edges'] = sage_graph.num_edges()

        for k in range(1, len(kset_list_dict)):
            kset_list = kset_list_dict[k]
            cnt_dict[f'num_{k}block'] = len(kset_list)
            cnt_dict[f'size_{k}block'] = [len(block) for block in kset_list]


def test(edges, name):
    from transforms.graph_drawio import draw_graph_drawio

    nx_graph = NX_Graph()
    nx_graph.add_edges_from(edges)
    pos = nx.spring_layout(nx_graph)

    # convert the position to drawio format
    pos = {key: val * 400 for key, val in pos.items()}
    draw_graph_drawio(nx_graph, pos, filepath=f"images/graph_having_{name}.drawio")

    sage_graph = SAGE_Graph(nx_graph)
    cnt_dict = {}
    c0block = compute_k_blocks(sage_graph, None, cnt_dict)
    print(c0block, cnt_dict['mk'])


if __name__ == "__main__":
    k3_block = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 4), (4, 3), (4, 5), (6, 7)]
    test(k3_block, "k3_block")

    k4_block = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (1, 3), (1, 4),
        (2, 3), (2, 4),
        (3, 5), (5, 4),
        (5, 6), (6, 7),
        (8, 9),
    ]
    test(k4_block, "k4_block")

    k5_block = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 8), (8, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        (4, 6), (6, 5),
        (6, 7),
    ]
    test(k5_block, "k5_block")
