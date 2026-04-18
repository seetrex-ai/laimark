"""
LoRA SFT baseline (reported as a failed approach in the paper).

Fine-tunes Qwen3-8B with LoRA on supervised self-generated solutions.
Saves one checkpoint per epoch. Kept for reproducibility of the
negative-result row in the main-result table.

Usage:
    python train_lora.py --data self_training_data.jsonl --output ./lora_self
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-8B"


def load_dataset_from_jsonl(path: str) -> Dataset:
    """Load chat-format JSONL into a HuggingFace Dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_chat(example, tokenizer):
    """Apply the chat template to convert messages into a single string."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="self_training_data.jsonl")
    parser.add_argument("--output", default="./lora_self")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA config targeting attention projections
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print(f"Loading dataset: {args.data}")
    dataset = load_dataset_from_jsonl(args.data)
    dataset = dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=["messages"],
    )
    print(f"Dataset size: {len(dataset)} examples")
    print(f"Sample (truncated): {dataset[0]['text'][:200]}...")

    training_args = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    # Save final adapter
    final_path = os.path.join(args.output, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
