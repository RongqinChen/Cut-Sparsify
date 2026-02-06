import warnings
from torch import Tensor

from torchmetrics.functional.classification.accuracy import (
    binary_accuracy, multiclass_accuracy, multilabel_accuracy
)
from torchmetrics.functional.classification.auroc import (
    binary_auroc, multiclass_auroc, multilabel_auroc
)
from torchmetrics.functional.classification.average_precision import (
    binary_average_precision, multiclass_average_precision, multilabel_average_precision
)
from torchmetrics.functional.classification.f_beta import (
    binary_f1_score, multiclass_f1_score, multilabel_f1_score,
)
from torchmetrics.functional.classification.precision_recall import (
    binary_precision, binary_recall,
    multiclass_precision, multiclass_recall,
    multilabel_precision, multilabel_recall,
)
from torchmetrics.functional.classification.specificity import (
    binary_specificity, multiclass_specificity, multilabel_specificity,
)
# from torchmetrics.functional.classification.confusion_matrix import (
#     binary_confusion_matrix, multiclass_confusion_matrix, multilabel_confusion_matrix
# )

warnings.filterwarnings("ignore", "No positive samples.*")


def reformat(val):
    return round(float(val), 5)


class BinaryClassificationEvaluator():
    def __init__(self):
        pass

    def __call__(self, preds: Tensor, target: Tensor):
        if preds.ndim > 1:
            preds = preds.squeeze(1)
        if target.ndim > 1:
            target = target.squeeze(1)

        result_dict = {
            'acc': reformat(binary_accuracy(preds, target).item()),
            'auroc': reformat(binary_auroc(preds.cuda(), target.cuda()).item()),
            'ap': reformat(binary_average_precision(preds, target).item()),
            'f1': reformat(binary_f1_score(preds, target).item()),
            'precision': reformat(binary_precision(preds, target).item()),
            'recall': reformat(binary_recall(preds, target).item()),
            'specificity': reformat(binary_specificity(preds, target).item()),
            # 'confusion': ",".join(map(str, map(round, binary_confusion_matrix(preds, target).flatten().tolist()))),
        }
        return result_dict


class MulticlassClassificationEvaluator():
    def __init__(self):
        pass

    def __call__(self, preds: Tensor, target: Tensor):
        if target.ndim > 1:
            target = target.squeeze(1)
        # num_classes = preds.size(1)
        target_int = target.int()
        predict_int = preds.argmax(1)
        correct = target_int == predict_int
        acc = correct.sum() / preds.size(0)
        result_dict = {
            'acc': reformat(acc),
            # 'auroc': reformat(multiclass_auroc(preds.cuda(), target_int.cuda(), num_classes).item()),
            # 'ap': reformat(multiclass_average_precision(preds, target_int, num_classes).item()),
            # 'f1': reformat(multiclass_f1_score(preds, target_int, num_classes).item()),
            # 'precision': reformat(multiclass_precision(preds, target_int, num_classes).item()),
            # 'recall': reformat(multiclass_recall(preds, target_int, num_classes).item()),
            # 'specificity': reformat(multiclass_specificity(preds, target_int, num_classes).item()),
            # 'confusion': ",".join(map(str, map(round, multiclass_confusion_matrix(preds, target, num_classes).flatten().tolist()))),
        }
        return result_dict


class MultilabelClassificationEvaluator():
    def __init__(self):
        pass

    def __call__(self, preds: Tensor, target: Tensor):
        if target.ndim > 1:
            target = target.squeeze(1)
        num_classes = preds.size(1)
        target_int = target.int()
        result_dict = {
            'acc': reformat(multilabel_accuracy(preds, target_int, num_classes).item()),
            'auroc': reformat(multilabel_auroc(preds.cuda(), target_int.cuda(), num_classes).item()),
            'ap': reformat(multilabel_average_precision(preds, target_int, num_classes).item()),
            'f1': reformat(multilabel_f1_score(preds, target_int, num_classes).item()),
            'precision': reformat(multilabel_precision(preds, target_int, num_classes).item()),
            'recall': reformat(multilabel_recall(preds, target_int, num_classes).item()),
            'specificity': reformat(multilabel_specificity(preds, target_int, num_classes).item()),
            # 'confusion': ",".join(map(str, map(round, multilabel_confusion_matrix(preds, target, num_classes).flatten().tolist()))),
        }
        return result_dict


__all__ = [BinaryClassificationEvaluator, MulticlassClassificationEvaluator, MultilabelClassificationEvaluator]
