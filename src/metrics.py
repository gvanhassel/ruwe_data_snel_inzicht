# from typing import Callable, Dict, Iterator, List, Optional, Tuple, Protocol, str
import torch

Tensor = torch.Tensor

class BinaryAccuracy:
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        """
        yhat is expected to be a vector with d dimensions.
        The highest values in the vector corresponds with
        the correct class.
        """
        yhat = (yhat > 0.5).float()
        correct = (yhat == y).float().sum()

        return correct / y.shape[0]


class Recall:
    def __repr__(self) -> str:
        return "Recall Metric"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        yhat = (yhat > 0.5).float()
        true_positives = torch.sum((y == 1) & (yhat == 1))
        false_negatives = torch.sum((y == 1) & (yhat == 0))
        try:
            return true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            return 0

class Precision:
    def __repr__(self) -> str:
        return "Precision Metric"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        yhat = (yhat > 0.5).float()
        true_positives = torch.sum((y == 1) & (yhat == 1))
        false_positives = torch.sum((y == 0) & (yhat == 1))
        try:
             return true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            return 0

class F1Score:
    def __repr__(self) -> str:
        return "F1 Score Metric"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        precision_metric = Precision()
        recall_metric = Recall()
        precision_score = precision_metric(y, yhat)
        recall_score = recall_metric(y, yhat)
        try:
            return 2 * (precision_score * recall_score) / (precision_score + recall_score)
        except ZeroDivisionError:
            return 0