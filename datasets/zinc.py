import time
from pathlib import Path
from typing import Any, Callable, Optional, List

from torch_geometric.data import Data
from torch_geometric.data.separate import separate
from torch_geometric.datasets import ZINC as PyG_ZINC
from tqdm import tqdm


class ZINC(PyG_ZINC):
    """Extended ZINC dataset with additional preprocessing capabilities.
    
    Args:
        root: Root directory where the dataset should be saved.
        subset: If True, uses a subset of the dataset.
        split: Dataset split to use ('train', 'val', or 'test').
        full_pre_transform: Optional transform applied to the full dataset with disk caching.
        inmemory_transform: Optional transform applied to data in memory.
        onthefly_transform: Optional transform applied on-the-fly during data loading.
        force_reload: If True, forces reloading of the dataset.
    """
    
    def __init__(
        self,
        root: str,
        subset: bool = False,
        split: str = 'train',
        full_pre_transform: Optional[Callable[..., Any]] = None,
        inmemory_transform: Optional[Callable[..., Any]] = None,
        onthefly_transform: Optional[Callable[..., Any]] = None,
        force_reload: bool = False
    ) -> None:
        super().__init__(root, subset, split, onthefly_transform, force_reload=force_reload)
        self._split = split
        self._data_list: Optional[List[Data]] = None
        
        # Apply transforms in order
        if full_pre_transform is not None:
            self._apply_full_pre_transform(full_pre_transform)
        if inmemory_transform is not None:
            self._apply_inmemory_transform(inmemory_transform)

        # Use only the first feature dimension
        self._separate_data()
        for data in self._data_list:
            data.x = data.x[:, 0]

    def _separate_data(self) -> List[Data]:
        """Separates the batched data into individual graph objects.
        
        Returns:
            List of individual graph data objects.
        """
        if self._data_list is None:
            self._data_list = [
                separate(
                    cls=self._data.__class__,
                    batch=self._data,
                    idx=idx,
                    slice_dict=self.slices,
                    decrement=False
                )
                for idx in range(self.len())
            ]
        return self._data_list

    def _apply_full_pre_transform(self, full_pre_transform: Callable) -> None:
        """Applies preprocessing transform with disk caching.
        
        Args:
            full_pre_transform: Transform function to apply to the dataset.
        """
        data_list = self._separate_data()
        processed_dir = Path(self.processed_dir) / self._split
        self._data_list = full_pre_transform(data_list, str(processed_dir))

    def _apply_inmemory_transform(self, inmemory_transform: Callable) -> None:
        """Applies in-memory transform to all data objects.
        
        Args:
            inmemory_transform: Transform function to apply to each data object.
        """
        data_list = self._separate_data()
        print(f"Computing {inmemory_transform.__name__ if hasattr(inmemory_transform, '__name__') else inmemory_transform}")
        
        time_start = time.perf_counter()
        transformed_data = [inmemory_transform(data) for data in tqdm(data_list, desc="Transforming")]
        time_elapsed = time.perf_counter() - time_start
        
        print(f"Transformation completed in {time_elapsed:.2f}s")
        self._data_list = transformed_data
