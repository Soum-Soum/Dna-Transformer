import evaluate
import numpy as np
import torch
from transformers import EvalPrediction

from adn.data.metadata import Metadata

accuracy_metric = evaluate.load("accuracy")


class MetricCalculator:

    def __init__(self, metadata: Metadata):
        self.metadata = metadata

    def _get_predictions_ids(self, logits) -> torch.Tensor:
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits)
        return torch.argmax(logits, dim=-1)

    def compute_final_accuracy(self, predictions: EvalPrediction) -> dict:
        logits, labels = predictions
        predictions_ids = self._get_predictions_ids(logits)
        return accuracy_metric.compute(
            predictions=predictions_ids, references=labels[:, 0]
        )

    def compute_family_accuracy(self, predictions: EvalPrediction) -> dict:
        logits, labels = predictions
        predictions_ids = self._get_predictions_ids(logits)
        label_id_to_family_id = self.metadata.label_id_to_family_id

        predicted_family_ids = torch.tensor(
            [label_id_to_family_id[pred_id.item()] for pred_id in predictions_ids]
        )
        family_ids = labels[:, 1]

        return accuracy_metric.compute(
            predictions=predicted_family_ids, references=family_ids
        )

    def compute_metrics(self, predictions: EvalPrediction) -> dict:
        final_acc = self.compute_final_accuracy(predictions)
        family_acc = self.compute_family_accuracy(predictions)
        return {
            "final_accuracy": final_acc["accuracy"],
            "family_accuracy": family_acc["accuracy"],
        }
