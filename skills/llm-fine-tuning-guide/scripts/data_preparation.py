"""
Data Preparation Utilities for Fine-Tuning

Handles dataset validation, augmentation, and splitting.
"""

from sklearn.model_selection import train_test_split
from collections import Counter
import textwrap


class DatasetValidator:
    """Validates dataset quality before training."""

    def validate_dataset(self, data):
        """Check for common data quality issues."""
        issues = {
            "empty_samples": 0,
            "duplicates": 0,
            "outliers": 0,
            "imbalance": {},
        }

        # Check for empty samples
        for sample in data:
            if not sample.get("text"):
                issues["empty_samples"] += 1

        # Check for duplicates
        texts = [s.get("text") for s in data]
        issues["duplicates"] = len(texts) - len(set(texts))

        # Check for length outliers
        lengths = [len(t.split()) for t in texts if t]
        if lengths:
            mean_length = sum(lengths) / len(lengths)
            issues["outliers"] = sum(1 for l in lengths if l > mean_length * 3)

        return issues

    def print_issues(self, issues):
        """Print formatted validation report."""
        print("Dataset Validation Report:")
        print(f"  Empty samples: {issues['empty_samples']}")
        print(f"  Duplicates: {issues['duplicates']}")
        print(f"  Outliers: {issues['outliers']}")


def create_splits(data, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """Create train/validation/test splits."""
    train_data, temp_data = train_test_split(
        data, train_size=train_size, random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)
    val_data, test_data = train_test_split(
        temp_data, train_size=val_ratio, random_state=random_state
    )

    return train_data, val_data, test_data


def augment_data(data, augmentation_strategy="synonym"):
    """Simple data augmentation strategies."""
    augmented = []

    for sample in data:
        augmented.append(sample)

        if augmentation_strategy == "paraphrase":
            # Add paraphrased version (requires external library)
            pass
        elif augmentation_strategy == "shuffle":
            # Shuffle word order (except first/last few words)
            if "text" in sample:
                words = sample["text"].split()
                if len(words) > 4:
                    import random

                    shuffled = words[:2] + random.sample(words[2:-2], len(words) - 4) + words[-2:]
                    augmented.append({"text": " ".join(shuffled), "label": sample.get("label")})

    return augmented


def format_for_instruction_tuning(data, template=None):
    """Format data for instruction fine-tuning."""
    if template is None:
        template = """Below is an instruction that describes a task, paired with an input that provides further context.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    formatted_data = []
    for sample in data:
        formatted_text = template.format(
            instruction=sample.get("instruction", ""),
            input=sample.get("input", ""),
            output=sample.get("output", ""),
        )
        formatted_data.append({"text": formatted_text})

    return formatted_data


if __name__ == "__main__":
    # Example usage
    validator = DatasetValidator()
    sample_data = [
        {"text": "This is a test sample", "label": 1},
        {"text": "This is another test", "label": 0},
    ]
    issues = validator.validate_dataset(sample_data)
    validator.print_issues(issues)
