"""
QLoRA (Quantized LoRA) Fine-Tuning Example

Combines LoRA with quantization for extreme efficiency on single GPU.
"""

from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer


def qlora_fine_tune(model_id: str = "meta-llama/Llama-2-7b", output_dir: str = "./qlora-output"):
    """QLoRA fine-tuning for efficient single-GPU training."""

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        num_train_epochs=3,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=None,  # Load your dataset here
    )

    trainer.train()
    return model


if __name__ == "__main__":
    model = qlora_fine_tune()
    print("✓ QLoRA fine-tuning completed")
