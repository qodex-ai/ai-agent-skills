"""
Evaluation Metrics for Fine-Tuned Models

Calculate perplexity, task-specific metrics, and quality scores.
"""

import torch
import math
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_perplexity(model, eval_dataset):
    """Calculate perplexity on evaluation dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in eval_dataset:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].shape[0]
            total_tokens += batch["input_ids"].shape[0]

    perplexity = math.exp(total_loss / total_tokens)
    return perplexity


def evaluate_task_metrics(predictions, ground_truth, average="weighted"):
    """Calculate standard ML metrics."""
    metrics = {
        "accuracy": accuracy_score(ground_truth, predictions),
        "precision": precision_score(ground_truth, predictions, average=average, zero_division=0),
        "recall": recall_score(ground_truth, predictions, average=average, zero_division=0),
        "f1": f1_score(ground_truth, predictions, average=average, zero_division=0),
    }
    return metrics


def print_evaluation_report(metrics):
    """Print formatted evaluation report."""
    print("\nEvaluation Metrics:")
    print("=" * 40)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            print(f"{metric_name:.<20} {metric_value:.4f}")
        else:
            print(f"{metric_name:.<20} {metric_value}")
    print("=" * 40)


class HumanEvaluationCriteria:
    """Framework for human evaluation of model outputs."""

    CRITERIA = {
        "relevance": "Does the response address the query?",
        "coherence": "Is the response logically consistent?",
        "factuality": "Are the facts accurate?",
        "helpfulness": "Would this help the user?",
    }

    @staticmethod
    def score_response(response: str, query: str, criteria: str = "relevance") -> int:
        """Score response on a scale of 1-5 for given criteria."""
        # Placeholder - implement with your evaluation logic
        return 3


if __name__ == "__main__":
    # Example usage
    sample_predictions = [0, 1, 1, 0, 1]
    sample_ground_truth = [0, 1, 0, 0, 1]

    metrics = evaluate_task_metrics(sample_predictions, sample_ground_truth)
    print_evaluation_report(metrics)
