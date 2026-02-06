import math
from collections import defaultdict
from typing import Dict

import torch
from lightning.pytorch import LightningModule
from torch import Tensor, nn
from torch_geometric.data import Batch, Data

from utils import cfg


class LightningModel(LightningModule):
    def __init__(self, net: nn.Module, criterion, evaluator):
        super(LightningModel, self).__init__()
        self.net = net
        self.criterion = criterion
        self.evaluator = evaluator
        self.step_results = defaultdict(list)
        self.epoch_results = {}
        self._reset_parameters()

    def _reset_parameters(self):
        def _init_weights(m):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        self.net.apply(_init_weights)

    def step(self, split, batch: Data | Batch, batch_idx: Tensor):
        batch = self.net(batch)
        if 'node' in cfg.model.task_type:
            preds, target = batch["node_pred"], batch["y"]
        else:
            preds, target = batch["graph_pred"], batch["y"]

        if target.ndim == 1 and preds.size(1) == 1:
            preds = preds.squeeze(dim=1)
        loss = self.criterion(preds, target)
        self.step_results[f'{split}_loss'].append(loss.detach().to('cpu', non_blocking=True))
        self.step_results[f'{split}_pred'].append(preds.detach().to('cpu', non_blocking=True))
        self.step_results[f'{split}_target'].append(target.detach().to('cpu', non_blocking=True))
        return loss

    def training_step(self, batch: Data | Batch, batch_idx: Tensor) -> Dict:
        return self.step('train', batch, batch_idx)

    def validation_step(self, batch: Data | Batch, batch_idx: Tensor) -> Dict:
        return self.step('val', batch, batch_idx)

    def test_step(self, batch: Data | Batch, batch_idx: Tensor) -> Dict:
        return self.step('test', batch, batch_idx)

    def _compute_metric(self, split) -> Dict:
        if len(self.step_results[f'{split}_pred']) == 0:
            return
        all_loss = torch.stack(self.step_results[f'{split}_loss'])
        all_preds = torch.cat(self.step_results[f'{split}_pred'])
        all_target = torch.cat(self.step_results[f'{split}_target'])
        results: dict = self.evaluator(all_preds, all_target)
        results.update({"loss_avg": all_loss.mean().item(), "loss_std": all_loss.std().item()})
        results = {f"{split}/{key}": val for key, val in results.items()}
        self.epoch_results.update(results)

    def on_validation_epoch_end(self):
        self._compute_metric('val')

    def on_test_epoch_start(self):
        self.epoch_results.clear()
        for vlist in self.step_results.values():
            vlist.clear()

    def on_test_epoch_end(self):
        self._compute_metric('test')
        self.log_dict(self.epoch_results, prog_bar=False)

    def on_train_epoch_start(self):
        self.epoch_results.clear()
        for vlist in self.step_results.values():
            vlist.clear()

    def on_train_epoch_end(self):
        self._compute_metric('train')
        self.log_dict(self.epoch_results, prog_bar=False)

    def _optimizer(self) -> torch.optim.Optimizer:
        if hasattr(torch.optim, cfg.train.optimizer):
            Optimizer = getattr(torch.optim, cfg.train.optimizer)
            return Optimizer(params=self.net.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        else:
            raise ValueError(f"{cfg.train.optimizer} is not a valid optimizer.")

    def _scheduler_cosine_with_warmup(self, optimizer):
        nwe = cfg.train.num_warmup_epochs

        def lr_lambda(current_step, num_cycles: float = 0.5):
            if current_step < nwe:
                return max(1e-6, float(current_step) / float(max(1, nwe)))

            progress = float(current_step - nwe) / float(max(1, cfg.train.num_epochs - nwe))
            return max(cfg.train.min_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)

    def _scheduler_reducelr_on_plateau(self, optimizer):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=float(self.args.factor),
            patience=float(self.args.patience), min_lr=float(self.args.min_lr)
        )

    def configure_optimizers(self) -> Dict:
        optimizer = self._optimizer()
        if cfg.train.scheduler == "cosine_with_warmup":
            scheduler = self._scheduler_cosine_with_warmup(optimizer)
            return [optimizer], [scheduler]
        if cfg.train.scheduler == "reducelr_on_plateau":
            scheduler = self._scheduler_reducelr_on_plateau(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler, "monitor": cfg.monitor.monitor,
                    "frequency": 1, "interval": "epoch"
                },
            }
        else:
            scheduler = self._scheduler_cosine_with_warmup(optimizer)
            return [optimizer], [scheduler]


class TestOnValLightningModel(LightningModel):
    r"""Given a preset evaluation interval, run test dataset during validation
        to have a snoop on test performance every args.test_eval_interval epochs during training.
    """

    def __init__(self, model: nn.Module, criterion, evaluator):
        super(TestOnValLightningModel, self).__init__(model, criterion, evaluator)
        self.test_eval_still = cfg.train.num_warmup_epochs

    def validation_step(self, batch: Data | Batch, batch_idx: Tensor, dataloader_idx: int) -> Dict:
        if dataloader_idx == 0:
            return super(TestOnValLightningModel, self).validation_step(batch, batch_idx)
        else:
            # only do validation when reaching the predefined epoch.
            if self.test_eval_still != 0:
                return {'loader_idx': dataloader_idx}
            return super(TestOnValLightningModel, self).test_step(batch, batch_idx)

    def on_validation_epoch_end(self):
        self._compute_metric('val')
        if self.test_eval_still == 0:
            self._compute_metric('test')
            self.test_eval_still = cfg.train.test_eval_interval
        else:
            self.test_eval_still = self.test_eval_still - 1

    def on_test_epoch_start(self):
        # set test validation interval to zero to performance test dataset validation.
        self.test_eval_still = 0
        super(TestOnValLightningModel, self).on_test_epoch_start()

    def test_step(self, batch: Data | Batch, batch_idx: Tensor, dataloader_idx: int) -> Dict:
        results = self.validation_step(batch, batch_idx, dataloader_idx)
        return results

    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
        self.log_dict(self.epoch_results, prog_bar=False)
