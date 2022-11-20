# -*- coding: utf-8 -*-
# @Time     : 2022/11/17 14:57
# @File     : modules.py
# @Author   : Zhou Hang
# @Email    : zhouhang@idataway.com
# @Software : Python 3.7
# @About    :
import numpy as np
from transformers.trainer_utils import EvalLoopOutput, PredictionOutput
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from sklearn.metrics import precision_recall_fscore_support

from .functional import sigmoid

__all__ = ['MetricsForGlobalPointerTask',
           'MetricsForBiaffineTask',
           'MetricsForCommonTextClassificationTask']

Output = TypeVar('Output', EvalLoopOutput, PredictionOutput)


def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")


class Metrics(object):

    def forward(self, output: Output) -> Dict:
        raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

    def __call__(self, output: Output) -> Dict:
        return self.forward(output=output)


class MetricsForBiaffineTask(Metrics):
    # https://github.com/suolyer/PyTorch_BERT_Biaffine_NER/blob/main/model/metrics/metrics.py
    def __init__(self):
        super(MetricsForBiaffineTask, self).__init__()

    def forward(self, output: Output) -> Dict:
        predictions, labels = output.predictions, output.label_ids

        print(predictions.shape)
        print(labels.shape)

        predictions = np.argmax(predictions, axis=-1)
        batch_size, seq_len, hidden = labels.shape
        predictions = predictions.reshape((batch_size, seq_len, hidden))
        predictions = predictions.reshape(-1).astype(np.float64)
        labels = labels.reshape(-1).astype(np.float64)
        ones = np.ones_like(predictions)
        zero = np.zeros_like(predictions)

        y_pred = np.where(predictions < 1, zero, ones)
        ones = np.ones_like(labels)
        zero = np.zeros_like(labels)
        y_true = np.where(labels < 1, zero, ones)

        corr = np.equal(predictions, labels).astype(np.float64)
        corr = np.multiply(corr, y_true)
        recall = np.sum(corr) / (np.sum(y_true) + 1e-8)
        precision = np.sum(corr) / (np.sum(y_pred) + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)

        return {
            "precision": precision,
            "recall": recall,
            "F1_score": f1,
            "Combined_score": np.mean([recall, precision, f1]).item(),
        }


class MetricsForGlobalPointerTask(Metrics):
    def __init__(self):
        super(MetricsForGlobalPointerTask, self).__init__()

    def forward(self, output: Output) -> Dict:
        predictions, labels = output.predictions, output.label_ids
        y_pred, y_true = [], []
        for b, l, start, end in zip(*np.where(predictions > 0)):
            y_pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(labels > 0)):
            y_true.append((b, l, start, end))

        y_pred = set(y_pred)
        y_true = set(y_true)
        true_positives = len(y_pred & y_true)
        true_positives_and_false_positives = len(y_pred)
        true_positives_and_false_negatives = len(y_true)

        if true_positives_and_false_positives == 0:
            precision = .0
        else:
            precision = true_positives / true_positives_and_false_positives
        if true_positives_and_false_negatives == 0:
            recall = .0
        else:
            recall = true_positives / true_positives_and_false_negatives
        if true_positives_and_false_positives + true_positives_and_false_negatives == 0:
            f1 = .0
        else:
            f1 = 2 * true_positives / (true_positives_and_false_positives + true_positives_and_false_negatives)

        return {
            "precision": precision,
            "recall": recall,
            "F1_score": f1,
            "Combined_score": np.mean([precision, recall, f1]).item()
        }


class MetricsForCommonTextClassificationTask(Metrics):
    def __init__(self, average='macro', multi_label=False, mark_line=.5, use_sigmoid=True):
        super(MetricsForCommonTextClassificationTask, self).__init__()
        self.multi_label = multi_label
        self.mark_line = mark_line
        self.average = average
        self.use_sigmoid = use_sigmoid

    def _mark_label(self, predictions):
        if self.multi_label:
            if self.use_sigmoid:
                predictions = sigmoid(predictions)
            return (predictions > self.mark_line).astype('int32')
        else:
            return np.argmax(predictions, axis=1)

    def forward(self, output: Output) -> Dict:
        predictions, labels = output.predictions, output.label_ids
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        predictions = self._mark_label(predictions=predictions)
        p, r, f1, _ = precision_recall_fscore_support(y_pred=predictions, y_true=labels, average=self.average)
        result = {"Precision": p,
                  "Recall": r,
                  "F1_score": f1,
                  "combined_score": np.mean([p, r, f1]).item()}
        return result
