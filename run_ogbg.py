# To get all data logged automatically, import comet_ml before the following modules: torch, sklearn.
import comet_ml  # noqa
import argparse
from collections import defaultdict
from datetime import datetime
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Timer

from datasets import OGBG
from ogb.graphproppred import Evaluator as OGB_Evaluator
from lighting_interface import TestOnValLightningData, TestOnValLightningModel
from models.model_construction import make_model
from utils import load_cfg, log_final_results, make_logger_trainer
torch.set_num_threads(8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='OGBG')
    parser.add_argument('--cfg', dest='cfg_file', type=str)
    parser.add_argument('--poly_method', type=str)
    parser.add_argument('--poly_dim', type=int)
    parser.add_argument('--pooling', type=str)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--drop_prob', type=float)
    parser.add_argument('--output_drop_prob', type=float)
    parser.add_argument('--specified_run', type=int, nargs="+")
    parser.add_argument('--local_log', action="store_true", help='Use log service locally.')
    return parser.parse_args()


class Evaluator(OGB_Evaluator):
    def __init__(self, name):
        super(Evaluator, self).__init__(name)

    def __call__(self, preds, target):
        result_dict = self.eval({'y_true': target, 'y_pred': preds})
        result_dict['metric'] = result_dict[self.eval_metric]
        return result_dict


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
    def forward(self, inputs, targets):
        is_labeled = targets == targets  # Filter our nans.
        targets = targets.float()
        if not torch.all(is_labeled):
            inputs = inputs[is_labeled]
            targets = targets[is_labeled]
        return super().forward(inputs, targets)


def main():
    args = parse_args()
    cfg = load_cfg(args)

    # ======== load dataset ==============
    dataset = OGBG(cfg.dataset.dataname, cfg.dataset.path, cfg.dataset.full_pre_transform,
                   cfg.dataset.inmemory_transform, cfg.dataset.onthefly_transform)
    train_set, valid_set, test_set = dataset.train_dataset(), dataset.valid_dataset(), dataset.test_dataset()

    specified_runs = range(cfg.train.runs) if args.specified_run is None else args.specified_run
    print("specified_runs", specified_runs)
    evaluator = Evaluator(cfg.dataset.dataname)
    print("eval_metric", evaluator.eval_metric)
    if evaluator.eval_metric == 'rmse':
        criterion = torch.nn.MSELoss()
        cfg.log.monitor_mode = 'min'
    else:
        cfg.log.monitor_mode = 'max'
        criterion = BCEWithLogitsLoss()
    cfg.log.monitor = f"val/{evaluator.eval_metric}"

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")[2:]
    results_allruns = defaultdict(list)
    for ridx in specified_runs:
        seed_everything((ridx + 1) * 1000)
        datamodule = TestOnValLightningData(train_set, valid_set, test_set)
        model = TestOnValLightningModel(make_model(), criterion, evaluator)
        timer = Timer(duration=dict(weeks=4))
        mcfg = cfg.model
        run_label = f"{ridx}-{mcfg.pooling}-{mcfg.drop_prob}-{mcfg.output_drop_prob}"
        logger, trainer = make_logger_trainer(timestamp, run_label, timer)
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
