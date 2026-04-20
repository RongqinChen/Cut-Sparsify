"""
Script to count triples in graph datasets using various transforms.
"""

import numpy as np
import torch
from tqdm import tqdm
from datasets import ZINC, QM9, TUDataset
from transforms.compute_2fwl import K2FWLTransform
from transforms.compute_2fwl_connsp import K2FWLConnSpTransform
from transforms.compute_2fwl_bsr import BSR2FWLTransform


def get_dataset(name: str) -> tuple:
    """
    Load dataset by name.

    Args:
        name: Dataset name (zinc, zincfull, qm9, ENZYMES, FRANKENSTEIN, NCI1, NCI109)

    Returns:
        Tuple of (train_set, valid_set, test_set)
    """
    valid_set = test_set = None

    if name == "zinc":
        train_set = ZINC("data/ZINC", subset=True, split="train")
        valid_set = ZINC("data/ZINC", subset=True, split="val")
        test_set = ZINC("data/ZINC", subset=True, split="test")
    elif name == "zincfull":
        train_set = ZINC("data/ZINC", subset=False, split="train")
        valid_set = ZINC("data/ZINC", subset=False, split="val")
        test_set = ZINC("data/ZINC", subset=False, split="test")
    elif name == "qm9":
        train_set = QM9("data/QM9")
    elif name in ["ENZYMES", "FRANKENSTEIN", "NCI1", "NCI109"]:
        train_set = TUDataset(name, "data/TUD")
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_set, valid_set, test_set


def get_transform(name: str):
    """
    Get transform by name.

    Args:
        name: Transform name (2fwl, cosp, sre)

    Returns:
        Transform instance
    """
    transform_map = {
        "2fwl": K2FWLTransform,  # Standard 3-WL/2-FWL
        "cosp": K2FWLConnSpTransform,  # Co-Sparsify
        "sre": BSR2FWLTransform,  # RSE-Sparsify
    }

    if name not in transform_map:
        raise ValueError(f"Unknown transform: {name}")

    return transform_map[name]()


def process_data(data, transform):
    """
    Process a single data sample with the given transform.

    Args:
        data: Input graph data
        transform: Transform to apply

    Returns:
        Processed data
    """
    num_nodes = data.num_nodes
    full_mat = torch.ones((num_nodes, num_nodes), dtype=torch.short)
    pair_index = full_mat.nonzero(as_tuple=False).t()

    data["pair_index"] = pair_index
    data["pair_x"] = torch.ones((num_nodes**2, 1))

    return transform(data)


def count_triples(transform, train_set, valid_set=None, test_set=None):
    """
    Count triples across train, validation, and test sets.

    Args:
        transform: Transform to apply
        train_set: Training dataset
        valid_set: Validation dataset (optional)
        test_set: Test dataset (optional)
    """
    valid_set = valid_set or []
    test_set = test_set or []

    triple_count = 0
    micro_avg = []
    total_molecules = len(train_set) + len(valid_set) + len(test_set)

    # Process all datasets
    for dataset in [train_set, valid_set, test_set]:
        for data in tqdm(dataset, desc="Processing graphs"):
            data = process_data(data, transform)
            micro_avg.append(data.triple_index.shape[1] / data.num_nodes)
            triple_count += data.triple_index.shape[1]

    # Calculate and print statistics
    avg_triples = triple_count / total_molecules if total_molecules > 0 else 0
    print(
        f"Average number of triples per molecule: {avg_triples:.2f} (micro avg: {np.mean(micro_avg):.2f})"
    )


def main(dataset_name: str, transform_name: str):
    """
    Main function to load dataset and count triples.

    Args:
        dataset_name: Name of the dataset
        transform_name: Name of the transform
    """
    train_set, valid_set, test_set = get_dataset(dataset_name)
    transform = get_transform(transform_name)
    count_triples(transform, train_set, valid_set, test_set)


if __name__ == "__main__":
    TRANSFORMS = ["2fwl", "cosp", "sre"]
    DATASETS = ["zinc", "zincfull", "qm9", "ENZYMES", "FRANKENSTEIN", "NCI1", "NCI109"]

    for transform_name in TRANSFORMS:
        for dataset_name in DATASETS:
            print(
                f"\nCounting all triples for {dataset_name} dataset with {transform_name} transform..."
            )
            try:
                main(dataset_name, transform_name)
            except Exception as e:
                print(
                    f"Error processing {dataset_name} with {transform_name}: {str(e)}"
                )
