import os
import time
from collections import Counter
from typing import List

import numpy as np
import torch
from sage.all import Graph as SAGE_Graph
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from .compute_pairconn import compute_pair_conn
from .hierarchy_embed import norm_hierarchy_embed


def to_sagegraph(num_nodes, edges):
    nodes = list(range(num_nodes))
    sagegraph = SAGE_Graph([nodes, edges], format="vertices_and_edges")
    return sagegraph


class FullyComputePairConn(BaseTransform):

    def __init__(self,) -> None:
        super().__init__()
        self.global_cnt_dict = None
        self.num_nodes_list = None
        self.conn_list: List[np.ndarray] = None

    def __call__(self, data_list: List[Data], save_dir) -> Data:
        self.load_or_compute(data_list, save_dir)
        self._embed_all(data_list)
        return data_list

    def __init_data__(self):
        MAX = 2**30 - 1
        self.global_cnt_dict = {
            "num_edges": [],
            "C_size_max": 0, "B_size_max": 0,
            "S_size_max": 0, "P_size_max": 0,
            "Q_size_max": 0, "R_size_max": 0,
            "C_size_min": MAX, "B_size_min": MAX,
            "S_size_min": MAX, "P_size_min": MAX,
            "Q_size_min": MAX, "R_size_min": MAX,
            "having_C": [], "having_B": [], "having_S": [],
            "having_P": [], "having_Q": [], "having_R": [],
            "conn_max": 0, "conn_min": MAX,
        }
        self._types = ["C", "B", "S", "P", "Q", "R"]
        self.num_nodes_list = []
        self.conn_list: List[np.ndarray] = []

    def load_or_compute(self, data_list: List[Data], save_dir) -> Data:
        self.__init_data__()
        if os.path.exists(f"{save_dir}/pair_conn.npy"):
            self._load_data(save_dir)
        else:
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/computing_conn_log.txt", "w") as logfile:
                self.compute_all(data_list, logfile)
                self._save_data(save_dir)

    def compute_all(self, data_list: List[Data], logfile):
        self.__init_data__()
        start_time = time.process_time()
        for idx in tqdm(range(len(data_list)), "Computing connectivity"):
            data = data_list[idx]
            num_nodes = data.num_nodes
            self.num_nodes_list.append(data.num_nodes)
            edges = data.edge_index.T.tolist()
            sage_graph = to_sagegraph(num_nodes, edges)
            cnt_dict = {}
            conn = compute_pair_conn(sage_graph, cnt_dict)
            self.conn_list.append(conn)
            self._update_global_cnt_dict(cnt_dict, conn)
            self._log_distribution(logfile, idx, cnt_dict, conn)

        end_time = time.process_time()
        print(f"Computing connectivity takes {end_time - start_time} seconds.", file=logfile)
        self._log_summary(logfile)

    def _embed_all(self, data_list: List[Data]):
        for idx in tqdm(range(len(data_list)), "Embedding connectivity"):
            data = data_list[idx]
            data_list[idx] = self._embed_one(idx, data)

    def _embed_one(self, gidx: int, data: Data):
        num_nodes = data.num_nodes
        conn = torch.from_numpy(self.conn_list[gidx], dtype=torch.float32)
        conn = norm_hierarchy_embed(conn, self.max_conn)
        data["pair_x"] = conn

        store_dict = dict(data.__dict__["_store"])
        if "diag_pos" not in store_dict:
            diag_pos = torch.arange(num_nodes)
            diag_pos = diag_pos * num_nodes + diag_pos
            data["diag_x"] = data["pair_x"][diag_pos, :]
            data["diag_pos"] = diag_pos

        if "pair_index" not in store_dict:
            full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
            pair_index = full_mat.nonzero(as_tuple=False).t()
            data["pair_index"] = pair_index

        self.conn_list[gidx] = None

    def _log_distribution(self, logfile, idx, cnt_dict, conn):
        print("idx:", idx, file=logfile)
        conn_d = _compute_distribution(conn.tolist())
        print(f"conn: {min(conn_d)}, {max(conn_d)}, {conn_d}", file=logfile)
        for key in self._types:
            sizes = cnt_dict[f"{key}_sizes"]
            if len(sizes) == 0:
                continue
            size_d = _compute_distribution(sizes)
            print(f"num_{key}: {len(sizes)}.", end="\t", file=logfile)
            print(f"{key}_sizes: {min(size_d)}, {max(size_d)}, {size_d}.", file=logfile)

    def _load_data(self, save_dir):
        conn_all = np.load(f"{save_dir}/pair_conn.npy")
        num_nodes_all = np.load(f"{save_dir}/num_nodes.npy")
        num_square = num_nodes_all**2
        cumsum = np.cumsum(num_square).tolist()[:-1]
        self.conn_list = np.split(conn_all, cumsum)
        for conn, square in zip(self.conn_list, num_square.tolist()):
            assert conn.shape[0] == square

    def _save_data(self, save_dir):
        conn_all = np.concatenate(self.conn_list)
        num_nodes_all = np.array(self.num_nodes_list)
        np.save(f"{save_dir}/pair_conn.npy", conn_all)
        np.save(f"{save_dir}/num_nodes.npy", num_nodes_all)

    def _log_summary(self, logfile):
        print("*" * 8, "summary", "*" * 8, file=logfile)
        nums = self.num_nodes_list
        print(f"nodes: {min(nums)}, {max(nums)}, {np.mean(nums):.2f}", file=logfile)
        g_cnt = self.global_cnt_dict
        nums = g_cnt["num_edges"]
        print(f"edges: {min(nums)}, {max(nums)}, {np.mean(nums):.2f}", file=logfile)
        print(f"conn: {g_cnt['conn_min']}, {g_cnt['conn_max']}", file=logfile)

        for key in self._types:
            gidxs = self.global_cnt_dict[f"having_{key}"]
            if len(gidxs) == 0:
                continue
            print(f"num_graphs_having_{key}: {len(gidxs)}", file=logfile)
            print(f"{key}_size_max: {self.global_cnt_dict[f'{key}_size_max']}", file=logfile)
            print(f"{key}_size_min: {self.global_cnt_dict[f'{key}_size_min']}", file=logfile)

        print("=" * 18, file=logfile)

    def _update_global_cnt_dict(self, cnt, conn):
        g_cnt = self.global_cnt_dict
        # g_cnt["num_nodes"].append(cnt["num_nodes"])
        g_cnt["num_edges"].append(cnt["num_edges"])
        g_cnt["conn_max"] = max(g_cnt["conn_max"], max(conn))
        g_cnt["conn_min"] = min(g_cnt["conn_min"], min(conn))

        gidx = len(g_cnt["num_edges"]) - 1
        for key in self._types:
            if len(cnt[f"{key}_sizes"]) == 0:
                continue
            g_cnt[f"{key}_size_max"] = max(max(cnt[f"{key}_sizes"]), g_cnt[f"{key}_size_max"])
            g_cnt[f"{key}_size_min"] = min(min(cnt[f"{key}_sizes"]), g_cnt[f"{key}_size_min"])
            g_cnt[f"having_{key}"].append(gidx)

    def __repr__(self) -> str:
        return "FullConn"


def _compute_distribution(values):
    cnt = dict(Counter(values))
    cnt = dict(sorted(cnt.items()))
    return cnt
