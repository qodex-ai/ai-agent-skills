"""
Full Fine-Tuning Example

Update all model parameters during training for maximum performance.
See scripts/ for data preparation utilities.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


def full_fine_tune(model_id: str = "meta-llama/Llama-2-7b", output_dir: str = "./fine-tuned-llama"):
    """Full fine-tuning example with all parameters updated."""

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=50,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,  # Load your dataset here
        eval_dataset=None,   # Load your eval dataset here
    )

    trainer.train()
    return trainer.model


if __name__ == "__main__":
    model = full_fine_tune()
    print("✓ Full fine-tuning completed")
