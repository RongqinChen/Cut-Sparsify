import os.path as osp
from os import makedirs

from sklearn.model_selection import StratifiedKFold
from torch import LongTensor


def get_idx_split(fold_idx, kfold_dir, graph_labels, test_set=True):
    seed = 42
    if not osp.exists(kfold_dir):
        kfold_split(kfold_dir, seed, graph_labels, test_set)

    train_fpath = f"{kfold_dir}/train_idx-{fold_idx}.txt"
    with open(train_fpath, 'r') as rfile:
        train_idx = [int(line[:-1]) for line in rfile.readlines()]

    valid_fpath = f"{kfold_dir}/valid_idx-{fold_idx}.txt"
    with open(valid_fpath, 'r') as rfile:
        valid_idx = [int(line[:-1]) for line in rfile.readlines()]

    if not test_set:
        return train_idx, valid_idx

    test_fpath = f"{kfold_dir}/test_idx-{fold_idx}.txt"
    with open(test_fpath, 'r') as rfile:
        test_idx = [int(line[:-1]) for line in rfile.readlines()]

    return train_idx, valid_idx, test_idx


def kfold_split(kfold_dir, seed, graph_labels, test_set=True):
    makedirs(kfold_dir, exist_ok=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    idx_list = list(skf.split(graph_labels, graph_labels))
    for jdx in range(10):
        if not test_set:
            train_idxs, valid_idxs = idx_list[jdx - 1]
            train_idxs = [f'{idx}\n' for idx in train_idxs]
            train_idx_file = f'{kfold_dir}/train_idx-{jdx}.txt'
            with open(train_idx_file, 'w') as wf:
                wf.writelines(train_idxs)

            valid_idxs = [f'{idx}\n' for idx in valid_idxs]
            valid_idx_file = f'{kfold_dir}/valid_idx-{jdx}.txt'
            with open(valid_idx_file, 'w') as wf:
                wf.writelines(valid_idxs)

        else:
            dev_idxs, test_idxs = idx_list[jdx - 1]
            dev_labels = graph_labels[LongTensor(dev_idxs)]
            train_rdxs, valid_rdxs = next(skf.split(dev_labels, dev_labels))
            train_idxs = {dev_idxs[idx] for idx in train_rdxs}
            valid_idxs = {dev_idxs[idx] for idx in valid_rdxs}
            test_idxs = set(test_idxs)
            union_set = set(train_idxs) | set(valid_idxs) | set(test_idxs)
            assert len(union_set) == graph_labels.shape[0]
            assert len(train_idxs & valid_idxs) == 0
            assert len(train_idxs & test_idxs) == 0
            assert len(valid_idxs & test_idxs) == 0

            train_idxs = [f'{idx}\n' for idx in train_idxs]
            train_idx_file = f'{kfold_dir}/train_idx-{jdx}.txt'
            with open(train_idx_file, 'w') as wf:
                wf.writelines(train_idxs)

            valid_idxs = [f'{idx}\n' for idx in valid_idxs]
            valid_idx_file = f'{kfold_dir}/valid_idx-{jdx}.txt'
            with open(valid_idx_file, 'w') as wf:
                wf.writelines(valid_idxs)

            test_idxs = [f'{idx}\n' for idx in test_idxs]
            test_idx_file = f'{kfold_dir}/test_idx-{jdx}.txt'
            with open(test_idx_file, 'w') as wf:
                wf.writelines(test_idxs)
