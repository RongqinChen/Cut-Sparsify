import argparse
from collections import defaultdict
from datetime import datetime

import swanlab
import torch
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Timer
from ogb.lsc import PCQM4Mv2Evaluator

from datasets import PCQM4Mv2
from evaluators import RegressionEvaluator
from lighting_interface import TestOnValLightningData, TestOnValLightningModel
from models.model_construction import make_model
from utils import load_cfg, log_final_results, make_logger_trainer
torch.set_num_threads(8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PCQM4Mv2')
    parser.add_argument('--cfg', dest='cfg_file', type=str)
    parser.add_argument('--poly_method', type=str)
    parser.add_argument('--poly_dim', type=int)
    parser.add_argument('--specified_run', type=int, nargs="+")
    parser.add_argument('--local_log', action="store_true", help='Use log service locally.')
    return parser.parse_args()


class Evaluator(PCQM4Mv2Evaluator):
    def __init__(self, ):
        super().__init__()
        self.regression_evaluator = RegressionEvaluator()

    def __call__(self, preds, target):
        input_dict = {"y_true": target, "y_pred": preds}
        result_dict = self.eval(input_dict)
        result_dict.update(self.regression_evaluator(preds, target))
        return result_dict


def main():
    args = parse_args()
    cfg = load_cfg(args)

    # ======== load dataset ==============
    dataset = PCQM4Mv2('full', cfg.dataset.path, cfg.dataset.full_pre_transform,
                       cfg.dataset.inmemory_transform, cfg.dataset.onthefly_transform)
    specified_runs = range(cfg.train.runs) if args.specified_run is None else args.specified_run
    print("specified_runs", specified_runs)
    evaluator = Evaluator()
    criterion = torch.nn.L1Loss()

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")[2:]
    results_allruns = defaultdict(list)
    for ridx in specified_runs:
        seed_everything((ridx + 1) * 1000)
        datamodule = TestOnValLightningData(*dataset.get_split())
        model = TestOnValLightningModel(make_model(), criterion, evaluator)
        timer = Timer(duration=dict(weeks=4))
        logger, trainer = make_logger_trainer(timestamp, ridx, timer, enable_progress_bar=True)
        trainer.fit(model, datamodule=datamodule)
        result_dict = trainer.test(model, datamodule=datamodule, ckpt_path="best")[0]
        result_dict["avg_train_time_epoch"] = timer.time_elapsed("train") / cfg.train.num_epochs
        result_dict = {f"final/{key.replace('/', '_')}": val for key, val in result_dict.items()}
        logger.log_metrics(result_dict)
        for key, val in result_dict.items():
            results_allruns[key].append(val)
        log_final_results(model, results_allruns, len(specified_runs))
        swanlab.finish()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
