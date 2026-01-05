"""
LoRA (Low-Rank Adaptation) Fine-Tuning Example

Parameter-efficient fine-tuning with 99% fewer trainable parameters.
"""

from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer


def lora_fine_tune(model_id: str = "meta-llama/Llama-2-7b", output_dir: str = "./llama-lora-adapter"):
    """LoRA fine-tuning with low-rank matrices."""

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Configure LoRA
    lora_config = LoraConfig(
        r=8,  # Rank of low-rank matrices
        lora_alpha=16,  # Scaling factor
        target_modules=["q_proj", "v_proj"],  # Which layers to adapt
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Wrap model with LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        logging_steps=10,
        save_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,  # Load your dataset here
    )

    trainer.train()

    # Save only LoRA weights
    model.save_pretrained(output_dir)
    return model


if __name__ == "__main__":
    model = lora_fine_tune()
    print("✓ LoRA fine-tuning completed")
