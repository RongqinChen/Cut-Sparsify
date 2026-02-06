import numpy as np
import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score, average_precision_score


"""
Evaluation functions from OGB.
https://github.com/snap-stanford/ogb/blob/master/ogb/graphproppred/evaluate.py
"""


def reformat(val):
    return round(float(val), 5)


def eval_acc(y_true: np.ndarray, y_pred):
    '''
        compute accuracy score averaged over samples
    '''
    if y_true.ndim == 1:
        labeled = y_true == y_true
        correct = y_true[labeled] == y_pred[labeled]
        results = {'acc': float(np.sum(correct)) / len(correct)}
    else:
        acc_list = []
        for i in range(y_true.shape[1]):
            labeled = y_true[:, i] == y_true[:, i]
            correct = y_true[labeled, i] == y_pred[labeled, i]
            acc_list.append(float(np.sum(correct)) / len(correct))

        results = {'acc': reformat(np.average(acc_list))}
    return results


def eval_auroc(y_true, y_pred):
    '''
        compute ROC-AUC averaged across tasks
    '''
    if y_true.ndim == 1:
        labeled = y_true == y_true
        try:
            results = {'auroc': roc_auc_score(y_true[labeled], y_pred[labeled])}
        except Exception as e:
            print(e)
            results = {'auroc': 0.}
    else:
        auroc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                labeled = y_true[:, i] == y_true[:, i]
                auroc_list.append(roc_auc_score(y_true[labeled, i], y_pred[labeled, i]))

        if len(auroc_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute ROC-AUC.')

        results = {'auroc': reformat(np.average(auroc_list))}
    return results


def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''
    if y_true.ndim == 1:
        labeled = y_true == y_true
        try:
            results = {'ap': average_precision_score(y_true[labeled], y_pred[labeled])}
        except Exception as e:
            print(e)
            results = {'ap': 0.}
    else:
        ap_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                # ignore nan values
                labeled = y_true[:, i] == y_true[:, i]
                ap = average_precision_score(y_true[labeled, i], y_pred[labeled, i])
                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')

        results = {'ap': reformat(np.average(ap_list))}
    return results


def eval_F1(y_true, y_pred):
    '''
        compute F1 score averaged over samples
    '''

    precision_list = []
    recall_list = []
    f1_list = []

    for ref, p in zip(y_true, y_pred):
        label = set(ref)
        prediction = set(p)
        true_positive = len(label.intersection(prediction))
        false_positive = len(prediction - label)
        false_negative = len(label - prediction)

        if true_positive + false_positive > 0:
            precision = true_positive / (true_positive + false_positive)
        else:
            precision = 0

        if true_positive + false_negative > 0:
            recall = true_positive / (true_positive + false_negative)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    results = {
        'precision': reformat(np.average(precision_list)),
        'recall': reformat(np.average(recall_list)),
        'F1': reformat(np.average(f1_list)),
    }
    return results


class OGBClassificationEvaluator():
    def __init__(self):
        pass

    def __call__(self, preds: Tensor, target: Tensor):
        if preds.ndim > 1:
            preds = preds.squeeze(1)
        if target.ndim > 1:
            target = target.squeeze(1)

        preds, target = preds.numpy(), target.numpy()
        result_dict = {}
        result_dict.update(eval_acc(target, preds))
        result_dict.update(eval_auroc(target, preds))
        result_dict.update(eval_ap(target, preds))
        # result_dict.update(eval_F1(target, preds))
        return result_dict


__all__ = [OGBClassificationEvaluator]
