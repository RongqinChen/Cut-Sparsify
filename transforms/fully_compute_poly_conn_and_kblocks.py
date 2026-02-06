import os
import pickle as pkl
import time
from collections import Counter, defaultdict
from typing import List

import numpy as np
import torch
from sage.all import Graph as SAGE_Graph
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from tqdm import tqdm

from .compute_kblocks import compute_k_blocks
from .compute_polynomial import poly_method_dict
from .hierarchy_embed import norm_hierarchy_embed
from .v1_kcv_hierarchy import v1_kcv_hierarchy
from .v1_kcvdata import V1KCVData


def to_sagegraph(num_nodes, edges):
    nodes = list(range(num_nodes))
    sagegraph = SAGE_Graph([nodes, edges], format="vertices_and_edges")
    return sagegraph


class FullyComputePolyPairConnAndKBlocks(BaseTransform):

    def __init__(self, max_kset_order=None, max_conn=None, poly_method=None, poly_dim=None) -> None:
        super().__init__()
        self.MAX_NUM = 2**30 - 1
        self.K = max_kset_order
        self.c0block_list = None
        self.conn_list = None
        self.poly_dim = poly_dim
        self.global_cnt_dict = None
        self.max_conn = max_conn
        self.num_nodes_list = None
        self.poly_name = poly_method

    def __call__(self, data_list: List[Data], save_dir) -> Data:
        self.load_or_compute(data_list, save_dir)
        self._embed_all(data_list)
        return data_list

    def __init_data__(self):
        self.gmk = 0
        self.c0block_list = []
        self.conn_list = []
        self.num_nodes_list = []
        self.global_cnt_dict = defaultdict(list)

    def load_or_compute(self, data_list: List[Data], save_dir) -> Data:
        self.__init_data__()
        if os.path.exists(f"{save_dir}/c0block_list.K{self.K}.pkl"):
            self._load_data(save_dir)
        else:
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/computing_kblocks.K{self.K}.log", "w") as logfile:
                start_time = time.process_time()
                self.compute_all(data_list, logfile)
                self._save_data(save_dir)
                end_time = time.process_time()
                print(f"Preprocess takes {end_time - start_time} seconds.", file=logfile)

    def _load_data(self, save_dir):
        conn_all = np.load(f"{save_dir}/kblk.pair_conn.K{self.K}.npy")
        num_nodes_all = np.load(f"{save_dir}/kblk.num_nodes.K{self.K}.npy")
        num_square = num_nodes_all**2
        cumsum = np.cumsum(num_square).tolist()[:-1]
        self.conn_list = np.split(conn_all, cumsum)
        with open(f"{save_dir}/c0block_list.K{self.K}.pkl", "rb") as rbf:
            self.c0block_list = pkl.load(rbf)
        # for conn, square in zip(self.conn_list, num_square.tolist()):
        #     assert conn.shape[0] == square

        print('Loaded preprocessed k-blocks.')

    def compute_all(self, data_list: List[Data], logfile):
        self.__init_data__()
        start_time = time.process_time()
        for idx in tqdm(range(len(data_list)), "Computing k-blocks"):
            data = data_list[idx]
            num_nodes = data.num_nodes
            edges = data.edge_index.T.tolist()

            sage_graph = to_sagegraph(num_nodes, edges)
            self.num_nodes_list.append(num_nodes)

            cnt_dict = {}
            conn_mat = np.zeros((num_nodes, num_nodes), dtype=np.int16)
            c0block = compute_k_blocks(sage_graph, self.K, cnt_dict, conn_mat)

            self.c0block_list.append(c0block)
            pair_conn_flat = conn_mat.flatten()
            self.conn_list.append(pair_conn_flat)
            self._update_global_cnt_dict(cnt_dict, pair_conn_flat)
            self._log_distribution(logfile, idx, cnt_dict, pair_conn_flat)

        end_time = time.process_time()
        print(f"Computing KBlocks takes {end_time - start_time} seconds.", file=logfile)
        self._log_summary(logfile)

    def _embed_all(self, data_list: List[Data]):
        self.poly_dim = self.poly_dim - self.max_conn
        self.poly_fn = poly_method_dict[self.poly_name]

        for idx in tqdm(range(len(data_list)), "Generating KCVData"):
            data = data_list[idx]
            data_list[idx] = self._embed_one(idx, data)

    def _embed_one(self, gidx: int, data: Data):
        num_nodes = data.num_nodes
        conn = self.conn_list[gidx]
        conn = torch.from_numpy(conn)
        conn = norm_hierarchy_embed(conn, self.max_conn)

        edge_index = data.edge_index
        num_nodes = data.num_nodes
        # compute row-normalized matrix
        adj_sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones((edge_index.size(1),)),
            (num_nodes, num_nodes),
            dtype=torch.float32,
        )
        adj = adj_sparse.to_dense()
        deg = adj.sum(dim=1, keepdim=True)
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float("inf")] = 0.0
        norm_adj = deg_inv * adj
        data["log(1+deg)"] = torch.log(1 + deg)

        poly = self.poly_fn(norm_adj, self.poly_dim)
        poly = poly.flatten(0, 1)

        pair_x = torch.cat((conn, poly), dim=1)
        data["pair_x"] = pair_x

        diag_pos = torch.arange(num_nodes)
        diag_pos = diag_pos * num_nodes + diag_pos
        data["diag_x"] = pair_x[diag_pos, :]
        data["diag_pos"] = diag_pos

        c0block = self.c0block_list[gidx]
        attr_dict = v1_kcv_hierarchy(self.K, c0block)

        store_dict = dict(data.__dict__["_store"])
        store_dict.update(attr_dict)
        kcv_data = V1KCVData(**store_dict)
        self.conn_list[gidx] = None
        self.c0block_list[gidx] = None
        return kcv_data

    def _log_distribution(self, logfile, idx, cnt, conn):
        print("idx:\t\t\t", idx, file=logfile)
        conn_distri = _compute_distribution(conn.tolist())
        min_conn = min(conn_distri)
        max_conn = max(conn_distri)
        print(f"conn:\t\t\t {min_conn}, {max_conn}, {conn_distri}", file=logfile)
        for k in range(1, cnt["mk"] + 1):
            print(f"num_{k}block:\t\t", cnt[f"num_{k}block"], file=logfile)
            print(f"size_{k}block:\t", cnt[f"size_{k}block"], file=logfile)
            self.global_cnt_dict[f"size_{k}block"].extend(cnt[f"size_{k}block"])

    def _save_data(self, save_dir):
        conn_all = np.concatenate(self.conn_list)
        num_nodes_all = np.array(self.num_nodes_list)
        np.save(f"{save_dir}/kblk.pair_conn.K{self.K}.npy", conn_all)
        np.save(f"{save_dir}/kblk.num_nodes.K{self.K}.npy", num_nodes_all)
        with open(f"{save_dir}/c0block_list.K{self.K}.pkl", "wb") as wbf:
            pkl.dump(self.c0block_list, wbf)

    def _log_summary(self, logfile):
        g_cnt = self.global_cnt_dict
        print("*" * 8, "summary", "*" * 8, file=logfile)
        print("max_conn:\t", g_cnt["max_conn"], file=logfile)
        print("min_conn:\t", g_cnt["min_conn"], file=logfile)
        print("The maximum order of k-blocks", self.gmk, file=logfile)
        for k in range(1, self.gmk + 1):
            print(f"num graphs having {k}-blocks:\t", len(g_cnt[f"having_{k}block"]), file=logfile)

        for k in range(1, self.gmk + 1):
            print(f"max {k}-block size:\t", f'{max(g_cnt[f"size_{k}block"])}', file=logfile)
            sizes = g_cnt[f"size_{k}block"]
            cnt = {}
            for s in sizes:
                cnt[s] = cnt.get(s, 0) + 1
            print(f"{k}block size cnt:", ', '.join([f"{k}:{cnt[k]}" for k in sorted(cnt)]), file=logfile)

        print("=" * 18, file=logfile)

    def _update_global_cnt_dict(self, cnt, conn):
        g_cnt = self.global_cnt_dict
        g_cnt["num_nodes"].append(cnt["num_nodes"])
        g_cnt["num_edges"].append(cnt["num_edges"])

        gidx = len(g_cnt["num_nodes"]) - 1
        for k in range(1, cnt["mk"] + 1):
            g_cnt[f"num_{k}block"].append(cnt[f"num_{k}block"])
            g_cnt[f"size_{k}block"].extend(cnt[f"size_{k}block"])
            g_cnt[f"having_{k}block"].append(gidx)

        self.gmk = max(self.gmk, cnt["mk"])
        g_cnt["max_conn"] = max(g_cnt.get("max_conn", 0), max(conn))
        g_cnt["min_conn"] = min(g_cnt.get("min_conn", self.MAX_NUM), min(conn))


def _compute_distribution(values):
    cnt = dict(Counter(values))
    cnt = dict(sorted(cnt.items()))
    return cnt
