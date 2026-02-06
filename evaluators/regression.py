import numpy as np
from torch import Tensor

from torchmetrics.functional.regression.mae import mean_absolute_error
from torchmetrics.functional.regression.mse import mean_squared_error
# from torchmetrics.functional.regression.pearson import pearson_corrcoef
from torchmetrics.functional.regression.r2 import r2_score
from torchmetrics.functional.regression.spearman import spearman_corrcoef


def reformat(val):
    return round(float(val), 5)


class RegressionEvaluator():
    def __init__(self):
        pass

    def __call__(self, preds: Tensor, target: Tensor):
        if preds.ndim > 1:
            preds = preds.squeeze(1)
        if target.ndim > 1:
            target = target.squeeze(1)

        result_dict = {
            'mae': reformat(mean_absolute_error(preds, target).item()),
            'rmse': reformat(np.sqrt(mean_squared_error(preds, target).item())),
            # 'pearson': reformat(pearson_corrcoef(preds, target).item()),
            'r2': reformat(r2_score(preds, target).item()),
            'spearman': reformat(spearman_corrcoef(preds, target).item()),
        }
        return result_dict


class MultilabelRegressionEvaluator():
    def __init__(self):
        pass

    def __call__(self, preds: Tensor, target: Tensor):
        if preds.ndim > 1:
            preds = preds.T.flatten()
        if target.ndim > 1:
            target = target.T.flatten()

        result_dict = {
            'mae': reformat(mean_absolute_error(preds, target).item()),
            'rmse': reformat(np.sqrt(mean_squared_error(preds, target).item())),
            # 'pearson': reformat(pearson_corrcoef(preds, target).item()),
            'r2': reformat(r2_score(preds, target).item()),
            'spearman': reformat(spearman_corrcoef(preds, target).item()),
        }
        return result_dict


__all__ = [RegressionEvaluator]
