import time
from typing import Any, Callable

import torch
from torch_geometric.data.separate import separate
from torch_geometric.datasets import QM9 as PyG_QM9
from tqdm import tqdm

target_names = [
    "mu", "alpha", "HOMO", "LUMO", "Delta", "R^2", "ZPVE", "U_0", "U", "H", "G", "C_v", "12"
]

HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


class QM9(PyG_QM9):
    def __init__(
        self, root: str,
        full_pre_transform: Callable[..., Any] | None = None,
        inmemory_transform: Callable[..., Any] | None = None,
        onthefly_transform: Callable[..., Any] | None = None,
        force_reload: bool = False
    ) -> None:
        self._split = 'full'
        super().__init__(root, onthefly_transform, force_reload=force_reload)
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
