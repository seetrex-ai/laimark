"""
DPO fine-tuning baseline (reported as a failed approach in the paper).

Two-phase training:
  Phase 1: SFT on correct solutions.
  Phase 2: DPO on preference pairs (chosen = pass, rejected = fail).

Kept for reproducibility of the negative-result row in the main-result
table. Not used for the headline numbers.

Usage:
    python train_dpo.py \
        --sft_data self_training_data_augmented.jsonl \
        --dpo_data dpo_training_data.jsonl \
        --output ./lora_dpo
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-8B"


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_chat(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_data", default="self_training_data_augmented.jsonl")
    parser.add_argument("--dpo_data", default="dpo_training_data.jsonl")
    parser.add_argument("--output", default="./lora_dpo")
    parser.add_argument("--model", default=MODEL_ID)
    # SFT params (same as v2)
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--sft_lr", type=float, default=2e-4)
    # DPO params
    parser.add_argument("--dpo_epochs", type=int, default=1)
    parser.add_argument("--dpo_lr", type=float, default=5e-5)
    parser.add_argument("--dpo_beta", type=float, default=0.1)
    # Shared
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # === Phase 1: SFT ===
    print("\n=== Phase 1: SFT ===")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )

    sft_dataset = Dataset.from_list(load_jsonl(args.sft_data))
    sft_dataset = sft_dataset.map(
        lambda ex: format_chat(ex, tokenizer),
        remove_columns=["messages"],
    )
    print(f"SFT dataset: {len(sft_dataset)} examples")

    sft_args = SFTConfig(
        output_dir=os.path.join(args.output, "sft"),
        num_train_epochs=args.sft_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=args.sft_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_length=args.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=sft_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting SFT...")
    sft_trainer.train()
    sft_model = sft_trainer.model
    print(f"SFT done. Loss: {sft_trainer.state.log_history[-1].get('train_loss', '?')}")

    # Save SFT checkpoint
    sft_path = os.path.join(args.output, "sft_checkpoint")
    sft_model.save_pretrained(sft_path)
    tokenizer.save_pretrained(sft_path)
    print(f"SFT checkpoint saved to: {sft_path}")

    # === Phase 2: DPO ===
    print("\n=== Phase 2: DPO ===")
    dpo_examples = load_jsonl(args.dpo_data)
    print(f"DPO dataset: {len(dpo_examples)} pairs")

    dpo_dataset = Dataset.from_list(dpo_examples)

    dpo_args = DPOConfig(
        output_dir=os.path.join(args.output, "dpo"),
        num_train_epochs=args.dpo_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=args.dpo_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        beta=args.dpo_beta,
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        max_length=args.max_seq_length,
        max_prompt_length=args.max_seq_length // 2,
        report_to="none",
    )

    dpo_trainer = DPOTrainer(
        model=sft_model,
        args=dpo_args,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
    )

    print("Starting DPO...")
    dpo_trainer.train()
    print("DPO done.")

    # Save final model
    final_path = os.path.join(args.output, "final")
    dpo_trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
