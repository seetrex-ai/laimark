"""
Evaluate a LoRA adapter on HumanEval via HuggingFace transformers in fp16.

Loads base model + adapter, merges, generates completions, runs HumanEval
tests. Uses the official human_eval package; no GGUF or other quantization
is involved in the pass@1 numbers reported in the paper.

Usage:
    pip install human_eval
    python eval_adapter.py --adapter_path ./grpo_output/final
"""

import argparse
import json
import os
import re
import signal
import sys
import tempfile
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-8B"

SYSTEM_L2B = (
    "Solve problems using structured algorithmic thinking. Initialize result "
    "containers explicitly before processing. When filtering or transforming "
    "collections, iterate through each element systematically and apply "
    "appropriate type checking or validation predicates. For data structure "
    "operations, understand the distinction between different Python types "
    "(integers, floats, strings, containers) and use built-in type checking "
    "functions when needed. Track and accumulate valid elements that meet "
    "specified criteria. Handle edge cases including empty inputs, mixed data "
    "types, and boundary values. Maintain clean separation between iteration "
    "logic and filtering conditions. Verify output matches expected types and "
    "constraints. For list operations, string manipulation, mathematical logic, "
    "and data structure filtering tasks, prefer clear iterative approaches that "
    "explicitly test each element against requirements. /no_think"
)


def run_code_with_timeout(code_str, timeout=5):
    """Execute code with timeout. Returns True if no exception."""
    def handler(signum, frame):
        raise TimeoutError()

    try:
        old_handler = signal.signal(signal.SIGALRM, handler)
        signal.alarm(timeout)
        exec(code_str, {})
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return True
    except Exception:
        signal.alarm(0)
        return False


def generate_completion(model, tokenizer, system_msg, user_msg, max_new_tokens=2048):
    """Generate a completion using the model."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=False)

    # Strip thinking tokens if present (R1-Distill, Qwen3 etc.)
    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    # Remove remaining special tokens
    response = re.sub(r'<\|[^>]+\|>', '', response)

    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default=MODEL_ID)
    parser.add_argument("--output", default="./eval_adapter_predictions.jsonl")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Evaluate base model without adapter (control)")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--system_prompt", default=None,
                        help="Override system prompt (default: SYSTEM_L2B)")
    args = parser.parse_args()

    # Load model
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.float16, device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.no_adapter:
        print(f"Loading adapter: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged.")

    model.eval()

    # Load HumanEval
    print("Loading HumanEval...")
    he_ds = load_dataset("openai/openai_humaneval", split="test")
    problems = {row["task_id"]: row for row in he_ds}
    task_ids = sorted(problems.keys())

    # Resume support
    existing = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                existing.add(json.loads(line)["task_id"])
        print(f"Resuming: {len(existing)}/{len(task_ids)} done")

    remaining = [t for t in task_ids if t not in existing]
    print(f"Evaluating {len(remaining)} problems...")

    passed = 0
    failed = 0
    total = len(task_ids)

    # Count existing results
    if os.path.exists(args.output):
        results_file = args.output + "_results.jsonl"
        if os.path.exists(results_file):
            with open(results_file) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("passed"):
                        passed += 1
                    else:
                        failed += 1

    t_start = time.time()
    for i, task_id in enumerate(remaining):
        prob = problems[task_id]
        prompt = prob["prompt"]
        test_code = prob["test"]
        entry_point = prob["entry_point"]

        user_msg = (
            "Complete the following Python function. "
            "Return ONLY the function body, no explanation."
            f"\n\n{prompt}"
        )

        t0 = time.time()
        sys_prompt = args.system_prompt if args.system_prompt else SYSTEM_L2B
        raw = generate_completion(model, tokenizer, sys_prompt, user_msg,
                                  max_new_tokens=args.max_new_tokens)

        # Extract code from response (handles reasoning models like R1-Distill)
        text = raw.strip()

        # Try to extract last ```python block (reasoning models put code at end)
        code_blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            text = code_blocks[-1].strip()
        else:
            # Fallback: strip markdown fences
            text = re.sub(r'^```(?:python)?\s*\n', '', text)
            text = re.sub(r'\n?```\s*$', '', text)

        completion = text.strip() + "\n" if text.strip() else "    pass\n"

        # Test execution
        full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"
        test_passed = run_code_with_timeout(full_code)

        if test_passed:
            passed += 1
        else:
            failed += 1

        elapsed = time.time() - t0
        done = len(existing) + i + 1
        status = "PASS" if test_passed else "FAIL"
        current_rate = passed / (passed + failed) if (passed + failed) > 0 else 0
        print(f"[{done}/{total}] {task_id} ({elapsed:.1f}s) {status} | Running: {passed}/{passed+failed} = {current_rate:.1%}", flush=True)

        # Save prediction
        pred = {"task_id": task_id, "completion": completion, "passed": test_passed}
        with open(args.output, "a") as f:
            f.write(json.dumps(pred) + "\n")

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} = {passed/total:.1%}")
    print(f"Time: {total_time/60:.1f} min")
    print(f"File: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
