from typing import Any, List, Optional, Sequence, Tuple, Union
import warnings
import torch.utils.data
from lightning.pytorch import LightningDataModule
from torch_geometric.data import Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.loader.dataloader import Collater

from utils import cfg


warnings.filterwarnings("ignore", ".*num_workers.*")


class MyCollater(Collater):
    def __init__(
        self, dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        if follow_batch is None:
            follow_batch = []
        if cfg.dataset.follow_batch is not None:
            follow_batch.extend(cfg.dataset.follow_batch)
        super().__init__(dataset, follow_batch, exclude_keys)

    def __call__(self, batch: List[Any]) -> Any:
        batched = super().__call__(batch)
        return batched


class MyDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset, batch_size, shuffle, drop_last=drop_last,
            collate_fn=MyCollater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )


class LightningData(LightningDataModule):
    def __init__(self, train_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset):
        super(LightningData, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def train_dataloader(self) -> MyDataLoader:
        return MyDataLoader(self.train_dataset, cfg.train.batch_size, True, True, num_workers=cfg.train.num_workers)

    def val_dataloader(self) -> MyDataLoader:
        return MyDataLoader(self.val_dataset, cfg.train.batch_size, False, num_workers=cfg.train.num_workers)

    def test_dataloader(self) -> MyDataLoader | None:
        return MyDataLoader(self.test_dataset, cfg.train.batch_size, False, num_workers=cfg.train.num_workers)


class TestOnValLightningData(LightningData):
    def val_dataloader(self) -> Tuple[MyDataLoader, MyDataLoader]:
        return (super().val_dataloader(), super().test_dataloader())

    def test_dataloader(self) -> Tuple[MyDataLoader, MyDataLoader]:
        return self.val_dataloader()
