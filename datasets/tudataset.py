import os.path as osp
import shutil
import time
from typing import Any, Callable

from torch_geometric.data.separate import separate
from torch_geometric.datasets import TUDataset as PyG_TUDataset
from torch_geometric.io import fs
from tqdm import tqdm

from .kfold import get_idx_split


class TUDataset(PyG_TUDataset):
    def __init__(
        self, name: str, root: str,
        full_pre_transform: Callable[..., Any] | None = None,
        inmemory_transform: Callable[..., Any] | None = None,
        onthefly_transform: Callable[..., Any] | None = None,
        force_reload: bool = False
    ) -> None:
        super().__init__(root, name, onthefly_transform, force_reload=force_reload, cleaned=False)
        self._split = 'full'
        if full_pre_transform is not None:
            self._full_pre_process(full_pre_transform)
        if inmemory_transform is not None:
            self._inmemory_process(inmemory_transform)

    def get_feature_summary(self):
        summary = {
            "node_attr_dim": self._data.num_node_features,
            "edge_attr_dim": self._data.num_edge_features,
            "num_tasks": self._data.y.max().item() + 1,
        }
        if summary["num_tasks"] == 2:
            summary["num_tasks"] = 1
        return summary

    def download(self) -> None:
        url = self.cleaned_url if self.cleaned else self.url
        fs.cp(f'{url}/{self.name}.zip', self.raw_dir, extract=True)
        for filename in fs.ls(osp.join(self.raw_dir, self.name)):
            # fs.mv(filename, osp.join(self.raw_dir, osp.basename(filename)))  has bugs
            shutil.move(filename, osp.join(self.raw_dir, osp.basename(filename)))

        fs.rm(osp.join(self.raw_dir, self.name))

    def _seperate_data(self):
        if not hasattr(self, '_data_list') or self._data_list is None:
            self._data_list = [
                separate(self._data.__class__, self._data, idx, self.slices, decrement=False)
                for idx in range(self.len())
            ]
        return self._data_list

    def _full_pre_process(self, full_pre_transform):
        data_list = self._seperate_data()
        processed_dir = self.processed_dir + '/' + self._split
        self._data_list = full_pre_transform(data_list, processed_dir)

    def _inmemory_process(self, inmemory_transform):
        data_list = self._seperate_data()
        print(f"Computing {inmemory_transform}")
        time_start = time.perf_counter()
        data_list = [inmemory_transform(data) for data in tqdm(data_list)]
        time_elapsed = time.perf_counter() - time_start
        print(f"Took {time_elapsed:.2f}s.")
        self._data_list.clear()
        del self._data_list
        self._data_list = data_list

    def get_split(self, fold_idx):
        kfold_dir = f"{self.root}/{self.name}/10fold"
        train_idx, valid_idx = get_idx_split(fold_idx, kfold_dir, self._data.y, False)
        train_dataset = self[train_idx]
        valid_dataset = self[valid_idx]
        return train_dataset, valid_dataset
