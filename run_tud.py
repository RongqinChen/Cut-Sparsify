# To get all data logged automatically, import comet_ml before the following modules: torch, sklearn.
import comet_ml  # noqa
import argparse
from collections import defaultdict
from datetime import datetime

import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Timer

from datasets import TUDataset
from evaluators import BinaryClassificationEvaluator
from evaluators import MulticlassClassificationEvaluator
from lighting_interface import LightningData, LightningModel
from models.model_construction import make_model
from utils import load_cfg, log_final_results, make_logger_trainer
torch.set_num_threads(8)


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, input, target):
        if target.ndim == 1:
            return super().forward(input, target.float())
        else:
            return super().forward(input, target)


criterion_evaluator_dict = {
    'BZR': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'COX2': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'DHFR': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'Mutagenicity': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'NCI1': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'NCI109': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'PTC_FR': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'DD': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'PROTEINS_full': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'ENZYMES': (torch.nn.CrossEntropyLoss(), MulticlassClassificationEvaluator()),
    'FRANKENSTEIN': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'IMDB-MULTI': (torch.nn.CrossEntropyLoss(), MulticlassClassificationEvaluator()),
    'IMDB-BINARY': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'REDDIT-BINARY': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'COLLAB': (torch.nn.CrossEntropyLoss(), MulticlassClassificationEvaluator()),
    'DBLP_v1': (BCEWithLogitsLoss(), BinaryClassificationEvaluator()),
    'REDDIT-MULTI-5K': (torch.nn.CrossEntropyLoss(), MulticlassClassificationEvaluator()),
    'REDDIT-MULTI-12K': (torch.nn.CrossEntropyLoss(), MulticlassClassificationEvaluator()),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TUDataset')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='configs/tud.grit.yaml')
    parser.add_argument('--dataname', type=str, default='ENZYMES')
    parser.add_argument('--poly_method', type=str, default="rrwp")
    parser.add_argument('--poly_dim', type=int, default=12)
    parser.add_argument('--specified_run', type=int, nargs="+")
    parser.add_argument('--local_log', action="store_true", help='Use log service locally.')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args)

    # ======== load dataset ==============
    dataset = TUDataset(args.dataname, cfg.dataset.path, cfg.dataset.full_pre_transform,
                        cfg.dataset.inmemory_transform, cfg.dataset.onthefly_transform)
    feat_summary = dataset.get_feature_summary()
    cfg.dataset.node_attr_dim = feat_summary['node_attr_dim']
    cfg.dataset.edge_attr_dim = feat_summary['edge_attr_dim']
    if cfg.dataset.node_attr_dim == 0:
        cfg.model.node_encoder = 'dummy'
    if cfg.dataset.edge_attr_dim == 0:
        cfg.model.edge_encoder = 'dummy'
    cfg.model.num_tasks = feat_summary['num_tasks']

    specified_runs = range(cfg.train.runs) if args.specified_run is None else args.specified_run
    print("specified_runs", specified_runs)
    criterion, evaluator = criterion_evaluator_dict[args.dataname]

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")[2:]
    results_allruns = defaultdict(list)
    for ridx in specified_runs:
        train_set, valid_set = dataset.get_split(ridx)
        seed_everything((ridx + 1) * 1000)
        datamodule = LightningData(train_set, valid_set, valid_set)
        model = LightningModel(make_model(), criterion, evaluator)
        timer = Timer(duration=dict(weeks=4))
        logger, trainer = make_logger_trainer(timestamp, f"{args.dataname}_{ridx}", timer)
        trainer.fit(model, datamodule=datamodule)
        result_dict = trainer.test(model, datamodule=datamodule, ckpt_path="best")[0]
        result_dict["avg_train_time_epoch"] = timer.time_elapsed("train") / cfg.train.num_epochs
        result_dict = {f"final/{key.replace('/', '_')}": val for key, val in result_dict.items()}
        logger.log_metrics(result_dict)
        for key, val in result_dict.items():
            results_allruns[key].append(val)
        log_final_results(model, results_allruns, len(specified_runs))
        logger._experiment.end()


if __name__ == "__main__":
    main()
