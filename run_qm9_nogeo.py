# To get all data logged automatically, import comet_ml before the following modules: torch, sklearn.
import comet_ml  # noqa
import argparse
from collections import defaultdict
from datetime import datetime
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Timer

from datasets.qm9 import QM9, target_names, conversion
from evaluators import RegressionEvaluator
from lighting_interface import TestOnValLightningData, TestOnValLightningModel
from models.model_construction import make_model
from utils import load_cfg, log_final_results, make_logger_trainer
torch.set_num_threads(16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='QM9')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='configs/qm9.grit.yaml')
    parser.add_argument('--poly_method', type=str)
    parser.add_argument('--poly_dim', type=int)
    parser.add_argument('--layers', type=int)
    parser.add_argument("--target", type=int, default=-1, choices=list(range(12)),
                        help="Train target. -1 for all first 12 targets.")
    parser.add_argument('--local_log', action="store_true", help='Use log service locally.')
    return parser.parse_args()


class SetY(object):
    def __init__(self, target, mean, std):
        super().__init__()
        self.target = target
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data.y = (data["label"][:, self.target] - self.mean) / self.std
        return data


class QM9RegressionEvaluator(RegressionEvaluator):
    def __init__(self, std, _conversion, **kwargs):
        super().__init__(**kwargs)
        self.std = std
        self.conversion = _conversion

    def __call__(self, preds, target):
        preds = preds * self.std / self.conversion
        target = target * self.std / self.conversion
        return super(QM9RegressionEvaluator, self).__call__(preds, target)


def main():
    args = parse_args()
    cfg = load_cfg(args)
    if args.target == -1:
        targets = list(range(12))
    else:
        targets = [args.target]

    # ============== load dataset ==============
    dataset = QM9(cfg.dataset.path, cfg.dataset.full_pre_transform,
                  cfg.dataset.inmemory_transform, cfg.dataset.onthefly_transform)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")[2:]
    criterion = torch.nn.L1Loss()
    results_allruns = defaultdict(list)
    for target in targets:
        seed_everything((target + 1) * 1000)
        args.target = target
        tenprecent = int(len(dataset) * 0.1)
        dataset = dataset.shuffle()
        train_dataset = dataset[2 * tenprecent:]
        test_dataset = dataset[:tenprecent]
        val_dataset = dataset[tenprecent:2 * tenprecent]

        y_list = [data["label"][0, target] for data in train_dataset]
        y_train = torch.stack(y_list, 0)
        mean, std = y_train.mean(), y_train.std()
        evaluator = QM9RegressionEvaluator(std, conversion[target])

        set_y_fn = SetY(target, mean, std)
        dataset._data_list = [set_y_fn(data) for data in dataset]
        train_dataset._data_list = dataset._data_list
        test_dataset._data_list = dataset._data_list
        val_dataset._data_list = dataset._data_list

        datamodule = TestOnValLightningData(train_dataset, val_dataset, test_dataset)
        model = TestOnValLightningModel(make_model(), criterion, evaluator)
        timer = Timer(duration=dict(weeks=4))
        logger, trainer = make_logger_trainer(timestamp, f"{target:02d}.{target_names[target]}", timer)
        trainer.fit(model, datamodule=datamodule)
        result_dict = trainer.test(model, datamodule=datamodule, ckpt_path="best")[0]
        result_dict["avg_train_time_epoch"] = timer.time_elapsed("train") / cfg.train.num_epochs
        result_dict = {f"final/{key.replace('/', '_')}": val for key, val in result_dict.items()}
        logger.log_metrics(result_dict)
        for key, val in result_dict.items():
            results_allruns[key].append(val)
        log_final_results(model, results_allruns, 1)
        logger._experiment.end()


if __name__ == "__main__":
    main()
