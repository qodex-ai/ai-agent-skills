# LLM Fine-Tuning Guide - Code Structure

This skill uses supporting Python files to keep documentation lean and maintainable.

## Directory Structure

```
llm-fine-tuning-guide/
├── SKILL.md                    # Main documentation (concepts, best practices)
├── README.md                   # This file
├── examples/                   # Implementation examples
│   ├── full_fine_tuning.py     # Full parameter fine-tuning
│   ├── lora_fine_tuning.py     # LoRA implementation
│   └── qlora_fine_tuning.py    # QLoRA (single GPU)
└── scripts/                    # Utility modules
    ├── data_preparation.py     # Dataset validation, augmentation, splitting
    └── evaluation_metrics.py   # Perplexity, task metrics, evaluation utils
```

## Running Examples

### 1. Full Fine-Tuning
```bash
python examples/full_fine_tuning.py
```
Updates all model parameters. Requires powerful GPU.

### 2. LoRA (Recommended)
```bash
python examples/lora_fine_tuning.py
```
Parameter-efficient, ~99% fewer trainable parameters.

### 3. QLoRA (Single Consumer GPU)
```bash
python examples/qlora_fine_tuning.py
```
Quantized LoRA for 7B models on 24GB GPU.

## Using the Utilities

### Data Preparation
```python
from scripts.data_preparation import DatasetValidator, create_splits, augment_data

validator = DatasetValidator()
issues = validator.validate_dataset(your_data)
validator.print_issues(issues)

train, val, test = create_splits(your_data)
```

### Evaluation Metrics
```python
from scripts.evaluation_metrics import calculate_perplexity, evaluate_task_metrics

perplexity = calculate_perplexity(model, eval_dataset)
metrics = evaluate_task_metrics(predictions, ground_truth)
```

## Integration with SKILL.md

- SKILL.md contains conceptual information and best practices
- Code examples are in `examples/` for clarity
- Utilities are in `scripts/` for reusability
- This keeps token costs low while maintaining full functionality

## Models Supported

- Llama 3.2 (1B, 3B, 8B)
- Gemma 3 (2B, 7B)
- Mistral 7B
- Any HuggingFace compatible model

## Requirements

```
torch>=2.0
transformers>=4.36
peft>=0.7
datasets>=2.14
scikit-learn>=1.3
```

## Next Steps

1. Prepare your dataset using `scripts/data_preparation.py`
2. Choose approach: Full, LoRA, or QLoRA
3. Run corresponding example script
4. Evaluate using `scripts/evaluation_metrics.py`
5. See SKILL.md for detailed explanations and best practices
