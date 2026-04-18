"""
GRPO training with verifiable reward (RLVR).

The model generates code, a Python subprocess executes it against
test cases, and pass/fail becomes the reward signal for GRPO.
Uses TRL's GRPOTrainer with LoRA on Qwen3-8B.

Modes:
  Default:        HumanEval + MBPP + selfgen mixed dataset.
  --selfgen_only: only self-generated calibrated problems, no external
                  benchmarks (headline result of the paper).

Usage:
    python train_grpo.py [--epochs 2] [--num_generations 4]
    python train_grpo.py --selfgen_only --selfgen_file calibrated_selfgen.jsonl
"""

import argparse
import json
import multiprocessing
import os
import signal
import sys
import tempfile
import traceback
from functools import partial

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer


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

SYSTEM_GENERIC = "You are an expert Python programmer. Write clean, correct code. /no_think"


# === Reward Function: Code Execution ===

def _run_code_in_process(code_str, timeout=5):
    """Execute code in a subprocess with timeout. Returns True if no exception."""
    try:
        result = {"passed": False}

        def target(code, res):
            try:
                exec(code, {})
                res["passed"] = True
            except Exception:
                res["passed"] = False

        manager = multiprocessing.Manager()
        res = manager.dict({"passed": False})
        p = multiprocessing.Process(target=target, args=(code_str, res))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join(1)
            return False
        return res.get("passed", False)
    except Exception:
        return False


def _extract_code(text):
    """Extract Python code from response (handles markdown blocks, reasoning)."""
    import re
    # Try ```python blocks first
    blocks = re.findall(r'```python\s*\n(.*?)```', text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    # Strip markdown fences
    text = re.sub(r'^```(?:python)?\s*\n', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def _check_humaneval(completion, prompt, test_code, entry_point):
    """Check a HumanEval completion (body-only: prompt + completion)."""
    full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"
    return _run_code_in_process(full_code)


def _check_humaneval_fullfunction(completion, test_code, entry_point):
    """Check a HumanEval completion (full-function: completion IS the function)."""
    code = _extract_code(completion)
    full_code = code + "\n" + test_code + f"\ncheck({entry_point})\n"
    return _run_code_in_process(full_code)


def _check_mbpp(completion, text, test_list, test_setup_code=""):
    """Check an MBPP completion."""
    setup = test_setup_code + "\n" if test_setup_code else ""
    test_code = "\n".join(test_list)
    full_code = setup + completion + "\n" + test_code
    return _run_code_in_process(full_code)


def _check_deduction(prediction, code_str, func_name, input_expr):
    """Check a deduction prediction: does func(input) match the prediction?"""
    prediction = prediction.strip().strip("`").strip()
    verify = (
        f"{code_str}\n"
        f"_result = {func_name}({input_expr})\n"
        f"assert repr(_result) == repr({prediction}), "
        f"f'expected {{repr({prediction})}}, got {{repr(_result)}}'"
    )
    return _run_code_in_process(verify)


def _check_abduction(guess_input, code_str, func_name, expected_output):
    """Check an abduction guess: does func(guess) produce expected output?"""
    guess_input = guess_input.strip().strip("`").strip()
    verify = (
        f"{code_str}\n"
        f"_result = {func_name}({guess_input})\n"
        f"assert _result == {expected_output}, "
        f"f'expected {expected_output}, got {{repr(_result)}}'"
    )
    return _run_code_in_process(verify)


def code_reward_fn(completions, problem_type, **kwargs):
    """Reward function for GRPO. Returns list of floats (0.0 or 1.0).

    Expects kwargs to contain problem metadata passed via dataset columns.
    """
    rewards = []
    for i, completion in enumerate(completions):
        # Extract completion text
        comp_text = completion[0]["content"] if isinstance(completion, list) else completion

        # Strip thinking tokens if present
        if "</think>" in comp_text:
            comp_text = comp_text.split("</think>", 1)[1]
        comp_text = comp_text.strip()

        try:
            ptype = problem_type[i] if isinstance(problem_type, list) else problem_type

            if ptype == "humaneval":
                prompt = kwargs["he_prompt"][i]
                test = kwargs["he_test"][i]
                entry = kwargs["he_entry_point"][i]
                passed = _check_humaneval(comp_text, prompt, test, entry)
            elif ptype == "humaneval_full":
                test = kwargs["he_test"][i]
                entry = kwargs["he_entry_point"][i]
                passed = _check_humaneval_fullfunction(comp_text, test, entry)
            elif ptype == "mbpp":
                text = kwargs["mbpp_text"][i]
                test_list = json.loads(kwargs["mbpp_test_list"][i])
                setup = kwargs.get("mbpp_test_setup", [""] * len(completions))[i]
                passed = _check_mbpp(comp_text, text, test_list, setup)
            elif ptype == "selfgen":
                tests = kwargs["sg_tests"][i]
                full_code = comp_text + "\n" + tests
                passed = _run_code_in_process(full_code)
            elif ptype == "deduction":
                code = kwargs["dd_code"][i]
                func_name = kwargs["dd_func_name"][i]
                inp = kwargs["dd_input"][i]
                passed = _check_deduction(comp_text, code, func_name, inp)
            elif ptype == "abduction":
                code = kwargs["ab_code"][i]
                func_name = kwargs["ab_func_name"][i]
                expected = kwargs["ab_expected"][i]
                passed = _check_abduction(comp_text, code, func_name, expected)
            else:
                passed = False
        except Exception as e:
            passed = False

        rewards.append(1.0 if passed else 0.0)

    return rewards


# === Dataset Preparation ===

def _empty_columns():
    """Return empty values for all problem-type-specific columns."""
    return {
        "he_prompt": "", "he_test": "", "he_entry_point": "",
        "mbpp_text": "", "mbpp_test_list": "[]", "mbpp_test_setup": "",
        "sg_tests": "",
        "dd_code": "", "dd_func_name": "", "dd_input": "", "dd_expected": "",
        "ab_code": "", "ab_func_name": "", "ab_expected": "",
    }


def _load_selfgen(sg_file, weighted=False, full_function=False, system_prompt=None):
    """Load self-generated problems from JSONL. Supports induction, deduction, abduction."""
    sys_msg = system_prompt or SYSTEM_L2B
    examples = []
    for line in open(sg_file):
        row = json.loads(line)
        ptype = row.get("problem_type", "selfgen")
        cols = _empty_columns()

        if ptype in ("selfgen", "induction"):
            ptype = "selfgen"
            if full_function:
                prompt_text = (
                    "Write the complete Python function including the def line. "
                    "Here is the signature and docstring:\n\n"
                    f"{row['prompt']}\n\n"
                    "Return ONLY the complete function code, no explanation."
                )
            else:
                prompt_text = (
                    "Complete the following Python function. "
                    "Return ONLY the function body, no explanation."
                    f"\n\n{row['prompt']}"
                )
            cols["sg_tests"] = row["tests"]

        elif ptype == "deduction":
            prompt_text = row["prompt"]
            cols["dd_code"] = row["dd_code"]
            cols["dd_func_name"] = row["dd_func_name"]
            cols["dd_input"] = row["dd_input"]
            cols["dd_expected"] = row["dd_expected"]

        elif ptype == "abduction":
            prompt_text = row["prompt"]
            cols["ab_code"] = row["ab_code"]
            cols["ab_func_name"] = row["ab_func_name"]
            cols["ab_expected"] = row["ab_expected"]

        else:
            continue

        example = {
            "prompt": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt_text},
            ],
            "problem_type": ptype,
            **cols,
        }

        # Weighted repetition: problems near 50% pass rate appear more often
        if weighted:
            pass_rate = row.get("pass_rate", row.get("source_pass_rate", 0.5))
            weight = 1.0 - 2 * abs(pass_rate - 0.5)
            copies = max(1, round(weight * 3))
        else:
            copies = 1

        for _ in range(copies):
            examples.append(example)

    return examples


def build_dataset(selfgen_only=False, selfgen_file=None, weighted=False,
                   full_function=False, system_prompt=None):
    """Build dataset for GRPO.

    If selfgen_only=True, uses ONLY self-generated calibrated problems.
    Otherwise, combines HumanEval + MBPP + any available selfgen.
    """
    sys_msg = system_prompt or SYSTEM_L2B
    examples = []

    if selfgen_only:
        if not selfgen_file or not os.path.exists(selfgen_file):
            raise FileNotFoundError(f"selfgen_file required for --selfgen_only: {selfgen_file}")
        examples = _load_selfgen(selfgen_file, weighted=weighted,
                                 full_function=full_function, system_prompt=system_prompt)
        by_type = {}
        for e in examples:
            by_type[e["problem_type"]] = by_type.get(e["problem_type"], 0) + 1
        print(f"Self-generated ONLY: {len(examples)} problems (0 external benchmarks)")
        for t, c in sorted(by_type.items()):
            print(f"  {t}: {c}")
        return Dataset.from_list(examples)

    # HumanEval
    try:
        from human_eval.data import read_problems
        he_problems = read_problems()
    except ImportError:
        print("human_eval not installed, downloading from HuggingFace...")
        he_ds = load_dataset("openai/openai_humaneval", split="test")
        he_problems = {row["task_id"]: row for row in he_ds}

    for task_id, prob in sorted(he_problems.items()):
        if full_function:
            prompt_text = (
                "Write the complete Python function including the def line. "
                "Here is the signature and docstring:\n\n"
                f"{prob['prompt']}\n\n"
                "Return ONLY the complete function code, no explanation."
            )
            ptype = "humaneval_full"
        else:
            prompt_text = (
                "Complete the following Python function. "
                "Return ONLY the function body, no explanation."
                f"\n\n{prob['prompt']}"
            )
            ptype = "humaneval"
        cols = _empty_columns()
        cols["he_prompt"] = prob["prompt"]
        cols["he_test"] = prob["test"]
        cols["he_entry_point"] = prob["entry_point"]
        examples.append({
            "prompt": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt_text},
            ],
            "problem_type": ptype,
            **cols,
        })

    print(f"HumanEval: {len([e for e in examples if e['problem_type'] == 'humaneval'])} problems")

    # MBPP
    mbpp = load_dataset("mbpp", split="test")
    for row in mbpp:
        prompt_text = (
            "Write a Python function to solve the following problem. "
            "Return ONLY the function code, no explanation."
            f"\n\n{row['text']}"
        )
        cols = _empty_columns()
        cols["mbpp_text"] = row["text"]
        cols["mbpp_test_list"] = json.dumps(row["test_list"])
        cols["mbpp_test_setup"] = row.get("test_setup_code", "")
        examples.append({
            "prompt": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt_text},
            ],
            "problem_type": "mbpp",
            **cols,
        })

    print(f"MBPP: {len([e for e in examples if e['problem_type'] == 'mbpp'])} problems")

    # Calibrated self-generated problems
    sg_file = selfgen_file or os.path.join(os.path.dirname(__file__) or ".", "calibrated_all.jsonl")
    if os.path.exists(sg_file):
        sg_examples = _load_selfgen(sg_file, weighted=weighted,
                                     full_function=full_function, system_prompt=system_prompt)
        examples.extend(sg_examples)
        print(f"Self-generated: {len(sg_examples)} problems")
    else:
        print("No self-generated problems found")

    print(f"Total: {len(examples)} problems")

    return Dataset.from_list(examples)


# === Main ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./grpo_output")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of completions per problem per step")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--no_vllm", action="store_true")
    parser.add_argument("--selfgen_only", action="store_true",
                        help="Train with ONLY self-generated calibrated problems, no external benchmarks")
    parser.add_argument("--selfgen_file", default=None,
                        help="Path to calibrated selfgen JSONL file")
    parser.add_argument("--base_adapter", default=None,
                        help="Merge this adapter into base before training (train FROM improved model)")
    parser.add_argument("--weighted", action="store_true",
                        help="Weight problem repetition by learnability (50%% pass rate = more copies)")
    parser.add_argument("--full_function", action="store_true",
                        help="Ask model for complete function (not body-only). For larger models.")
    parser.add_argument("--system_prompt", default=None,
                        help="Override system prompt (default: SYSTEM_L2B)")
    args = parser.parse_args()

    sys_prompt = args.system_prompt or SYSTEM_L2B
    print(f"System prompt: {sys_prompt[:60]}...")
    print(f"Full function mode: {args.full_function}")

    print("Building dataset...")
    dataset = build_dataset(
        selfgen_only=args.selfgen_only,
        selfgen_file=args.selfgen_file,
        weighted=args.weighted,
        full_function=args.full_function,
        system_prompt=sys_prompt if args.system_prompt else None,
    )

    print(f"Loading tokenizer: {args.model}")
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

    grpo_config = GRPOConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=2,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        report_to="none",
        # vLLM for fast generation
        use_vllm=not args.no_vllm,
        vllm_gpu_memory_utilization=0.4,
    )

    # If base_adapter provided, merge it first (train FROM improved model)
    model_ref = args.model
    if args.base_adapter:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        print(f"Loading base model and merging adapter: {args.base_adapter}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, trust_remote_code=True
        )
        base_model = PeftModel.from_pretrained(base_model, args.base_adapter)
        base_model = base_model.merge_and_unload()
        # Save merged model temporarily
        merged_path = os.path.join(args.output, "_merged_base")
        base_model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        del base_model
        model_ref = merged_path
        print(f"Merged model saved to {merged_path}")

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model_ref,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=code_reward_fn,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    # Save final adapter
    final_path = os.path.join(args.output, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Final adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
