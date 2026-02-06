import time
from typing import Any, Callable

import torch
from numpy.random import default_rng
from .pyg_pcqm4m import PygPCQM4Mv2Dataset
from torch_geometric.data.separate import separate
from tqdm import tqdm


class PCQM4Mv2(PygPCQM4Mv2Dataset):

    def __init__(
        self, name: str, root: str,
        full_pre_transform: Callable[..., Any] | None = None,
        inmemory_transform: Callable[..., Any] | None = None,
        onthefly_transform: Callable[..., Any] | None = None,
    ) -> None:
        super().__init__(root, transform=onthefly_transform)
        self.name = name
        if full_pre_transform is not None:
            self._full_pre_process(full_pre_transform)
        if inmemory_transform is not None:
            self._inmemory_process(inmemory_transform)

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

    def get_split(self):
        split_idx = self.get_idx_split2()

        rng = default_rng(seed=42)
        train_idx = rng.permutation(split_idx["train"].numpy())
        train_idx = torch.from_numpy(train_idx)

        # Leave out 150k graphs for a new validation set.
        valid_idx, train_idx = train_idx[:150000], train_idx[150000:]
        if self.name == "full":
            train_dataset = self[train_idx]
            valid_dataset = self[valid_idx]
            test_dataset = self[split_idx["valid"]]
            return train_dataset, valid_dataset, test_dataset
        elif self.name == "subset":
            # Further subset the training set for faster debugging.
            subset_ratio = 0.1
            subtrain_idx = train_idx[: int(subset_ratio * len(train_idx))]
            subvalid_idx = valid_idx[:50000]
            subtest_idx = split_idx["valid"]  # The original 'valid' as testing set.
            subtrain_dataset = self[subtrain_idx]
            subvalid_dataset = self[subvalid_idx]
            subtest_dataset = self[subtest_idx]
            return subtrain_dataset, subvalid_dataset, subtest_dataset
        elif self.name == "inference":
            test_dev_dataset = self[split_idx["test-dev"]]
            test_challenge_dataset = self[split_idx["test-challenge"]]
            return test_dev_dataset, test_challenge_dataset
