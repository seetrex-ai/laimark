"""
Calibrate deduction/abduction problems by learnability.

Samples K completions per problem, checks pass rate,
keeps only problems in [lo, hi] zone.

Usage:
    python calibrate_deduction_abduction.py \
        --input pool_deduction_abduction.jsonl \
        --output calibrated_da.jsonl \
        --adapter /path/to/adapter  # calibrate against this model
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def generate_batch(model, tokenizer, system_msg, user_msg, n, temperature=0.7, max_new_tokens=256):
    """Generate n completions in one batched forward pass."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    single = tokenizer(text, return_tensors="pt")
    input_ids = single["input_ids"].repeat(n, 1).to(model.device)
    attention_mask = single["attention_mask"].repeat(n, 1).to(model.device)
    prompt_len = single["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True, top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
        )

    results = []
    for i in range(n):
        generated = outputs[i][prompt_len:]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        if "</think>" in response:
            response = response.split("</think>", 1)[1]
        results.append(response.strip())
    return results


def execute_code(code_str, timeout=5):
    """Run code with timeout. Returns True if exit code 0."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code_str)
            tmpfile = f.name
        result = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True, timeout=timeout, text=True,
        )
        os.unlink(tmpfile)
        return result.returncode == 0
    except Exception:
        try:
            os.unlink(tmpfile)
        except Exception:
            pass
        return False


def check_deduction(prediction, code, func_name, input_expr):
    """Check deduction: does func(input) match prediction?"""
    prediction = prediction.strip().strip("`").strip()
    verify = f"{code}\n_r = {func_name}({input_expr})\nassert repr(_r) == repr({prediction})"
    return execute_code(verify)


def check_abduction(guess, code, func_name, expected):
    """Check abduction: does func(guess) == expected?"""
    guess = guess.strip().strip("`").strip()
    verify = f"{code}\n_r = {func_name}({guess})\nassert _r == {expected}"
    return execute_code(verify)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", default=None,
                        help="Adapter to merge (calibrate against improved model)")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--lo", type=float, default=0.2)
    parser.add_argument("--hi", type=float, default=0.8)
    args = parser.parse_args()

    print(f"=== Calibrate deduction/abduction ===")
    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter:
        from peft import PeftModel
        print(f"Merging adapter...")
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    model.eval()
    print("Model loaded.")

    # Load problems
    problems = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    print(f"Loaded {len(problems)} problems")

    calibrated = 0
    total = len(problems)

    with open(args.output, "w") as out:
        for idx, prob in enumerate(problems):
            ptype = prob.get("problem_type", "")
            prompt = prob.get("prompt", "")

            completions = generate_batch(
                model, tokenizer, SYSTEM_L2B, prompt,
                n=args.samples, temperature=0.7, max_new_tokens=256
            )

            passes = 0
            for comp in completions:
                if not comp:
                    continue
                try:
                    if ptype == "deduction":
                        ok = check_deduction(
                            comp, prob["dd_code"], prob["dd_func_name"], prob["dd_input"]
                        )
                    elif ptype == "abduction":
                        ok = check_abduction(
                            comp, prob["ab_code"], prob["ab_func_name"], prob["ab_expected"]
                        )
                    else:
                        ok = False
                    if ok:
                        passes += 1
                except Exception:
                    pass

            pass_rate = passes / args.samples
            in_zone = args.lo <= pass_rate <= args.hi

            if in_zone:
                calibrated += 1
                prob["pass_rate"] = pass_rate
                prob["samples"] = args.samples
                out.write(json.dumps(prob) + "\n")

            status = f"ACCEPT ({pass_rate:.0%})" if in_zone else f"reject ({pass_rate:.0%})"
            source = prob.get("source_entry_point", "?")
            print(f"[{idx+1}/{total}] {ptype} ({source}): {passes}/{args.samples} "
                  f"= {status} | calibrated={calibrated}", flush=True)

    print(f"\nDone: {calibrated}/{total} calibrated ({calibrated/total*100:.1f}%)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
