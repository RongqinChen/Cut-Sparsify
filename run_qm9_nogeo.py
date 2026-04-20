import argparse
from collections import defaultdict
from datetime import datetime

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Timer

from datasets.qm9 import QM9, conversion, target_names
from evaluators import RegressionEvaluator
from lighting_interface import TestOnValLightningData, TestOnValLightningModel
from models.model_construction import make_model
from utils import create_trainer, load_cfg, log_final_results

torch.set_num_threads(16)

# Constants
NUM_QM9_TARGETS = 12
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 0.1
TEST_SPLIT_RATIO = 0.1


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for QM9 training.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train models on QM9 molecular property prediction dataset"
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        type=str,
        default="configs/qm9.grit.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--poly_method", type=str, help="Polynomial method to use")
    parser.add_argument("--poly_dim", type=int, help="Polynomial dimension")
    parser.add_argument("--layers", type=int, help="Number of layers in the model")
    parser.add_argument(
        "--target",
        type=int,
        default=-1,
        choices=list(range(NUM_QM9_TARGETS)),
        help="Target property to train on. -1 trains on all first 12 targets.",
    )
    parser.add_argument(
        "--local_log", action="store_true", help="Use log service locally"
    )
    return parser.parse_args()


class SetY:
    """Transform to normalize target labels using z-score normalization.

    Attributes:
        target: Index of the target property.
        mean: Mean value for normalization.
        std: Standard deviation for normalization.
    """

    def __init__(self, target: int, mean: float, std: float):
        self.target = target
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.y = (data["label"][:, self.target] - self.mean) / self.std
        return data


class QM9RegressionEvaluator(RegressionEvaluator):
    """Custom evaluator for QM9 regression that handles unit conversion.

    Attributes:
        std: Standard deviation used for denormalization.
        conversion: Conversion factor for the target property units.
    """

    def __init__(self, std: float, _conversion: float, **kwargs):
        super().__init__(**kwargs)
        self.std = std
        self.conversion = _conversion

    def __call__(self, preds: torch.Tensor, target: torch.Tensor):
        # Denormalize and convert units
        preds = preds * self.std / self.conversion
        target = target * self.std / self.conversion
        return super().__call__(preds, target)


def split_dataset(
    dataset, test_ratio: float = TEST_SPLIT_RATIO, val_ratio: float = VAL_SPLIT_RATIO
):
    """Split dataset into train, validation, and test sets.

    Args:
        dataset: Full dataset to split.
        test_ratio: Proportion of data for test set.
        val_ratio: Proportion of data for validation set.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    dataset = dataset.shuffle()
    total_size = len(dataset)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)

    test_dataset = dataset[:test_size]
    val_dataset = dataset[test_size:test_size + val_size]
    train_dataset = dataset[test_size + val_size:]

    return train_dataset, val_dataset, test_dataset


def compute_normalization_stats(train_dataset, target: int):
    """Compute mean and std for target normalization from training data.

    Args:
        train_dataset: Training dataset.
        target: Index of target property.

    Returns:
        Tuple of (mean, std).
    """
    y_list = [data["label"][0, target] for data in train_dataset]
    y_train = torch.stack(y_list, dim=0)
    mean = y_train.mean()
    std = y_train.std()
    return mean, std


def normalize_dataset(dataset, set_y_fn):
    """Apply normalization transform to all data in dataset.

    Args:
        dataset: Dataset to normalize.
        set_y_fn: Normalization function to apply.

    Returns:
        Dataset with normalized targets.
    """
    dataset._data_list = [set_y_fn(data) for data in dataset]
    return dataset


def train_single_target(
    target: int, dataset, cfg, timestamp: str, results_allruns: dict
):
    """Train model for a single target property.

    Args:
        target: Index of target property.
        dataset: Full QM9 dataset.
        cfg: Configuration object.
        timestamp: Timestamp string for logging.
        results_allruns: Dictionary to store results across runs.
    """
    # Set random seed for reproducibility
    seed_everything((target + 1) * 1000)

    # Split dataset
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Compute normalization statistics
    mean, std = compute_normalization_stats(train_dataset, target)

    # Create evaluator with proper conversions
    evaluator = QM9RegressionEvaluator(std, conversion[target])

    # Normalize datasets
    set_y_fn = SetY(target, mean, std)
    dataset = normalize_dataset(dataset, set_y_fn)
    train_dataset._data_list = dataset._data_list
    val_dataset._data_list = dataset._data_list
    test_dataset._data_list = dataset._data_list

    # Setup training components
    datamodule = TestOnValLightningData(train_dataset, val_dataset, test_dataset)
    criterion = torch.nn.L1Loss()
    model = TestOnValLightningModel(make_model(), criterion, evaluator)
    timer = Timer(duration=dict(weeks=4))

    # Train model
    target_name = f"{target:02d}.{target_names[target]}"
    trainer = create_trainer(timestamp, target_name, timer)
    trainer.fit(model, datamodule=datamodule)

    # Evaluate and log results
    result_dict = trainer.test(model, datamodule=datamodule, ckpt_path="best")[0]
    result_dict["avg_train_time_epoch"] = (
        timer.time_elapsed("train") / cfg.train.num_epochs
    )

    # Format result keys
    result_dict = {
        f"final/{key.replace('/', '_')}": val for key, val in result_dict.items()
    }

    # Store results
    for key, val in result_dict.items():
        results_allruns[key].append(val)

    log_final_results(model, results_allruns, 1)


def main():
    """Main training loop for QM9 dataset."""
    # Parse arguments and load configuration
    args = parse_args()
    cfg = load_cfg(args)

    # Determine which targets to train
    if args.target == -1:
        targets = list(range(NUM_QM9_TARGETS))
    else:
        targets = [args.target]

    # Load QM9 dataset
    dataset = QM9(
        cfg.dataset.path,
        cfg.dataset.full_pre_transform,
        cfg.dataset.inmemory_transform,
        cfg.dataset.onthefly_transform,
    )

    # Generate timestamp for logging
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")[2:]

    # Store results across all targets
    results_allruns = defaultdict(list)

    # Train on each target
    for target in targets:
        train_single_target(target, dataset, cfg, timestamp, results_allruns)


if __name__ == "__main__":
    main()
