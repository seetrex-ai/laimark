"""
Generate and calibrate self-evaluation problems via HuggingFace transformers.

Runs the full generate -> verify -> calibrate -> dedup pipeline in-process,
with no dependency on Ollama. Use this variant on a GPU.

Pipeline:
    1. Generate problem (function + tests + reference solution)
    2. Verify: reference solution passes its own tests
    3. Calibrate: sample K solutions, keep if pass rate in [0.2, 0.8]
    4. Dedup by function name + docstring Jaccard

Usage:
    python generate_and_calibrate_hf.py --target 200 --model Qwen/Qwen3-8B
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_GEN = "You are an expert Python programmer who creates programming problems. /no_think"

SYSTEM_SOLVE = (
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

GENERATE_PROMPT = """Generate a challenging Python programming problem. You must provide ALL THREE parts:

1. A function with signature and docstring (including examples)
2. Test cases as assert statements
3. A correct reference solution

Format your response EXACTLY like this:

FUNCTION:
def function_name(params):
    \"\"\"Description of what the function does.
    >>> function_name(example_input)
    example_output
    \"\"\"

TESTS:
assert function_name(input1) == expected1
assert function_name(input2) == expected2
assert function_name(input3) == expected3

SOLUTION:
def function_name(params):
    # implementation
    return result

Requirements:
- The function must be self-contained (no imports needed, or include them)
- Include at least 5 test cases covering normal cases, edge cases, and tricky corner cases
- The solution must be correct and pass all tests
- Difficulty: {difficulty}
- Topic: {topic}
- The problem MUST require careful handling of edge cases that are easy to miss
- Include at least one test with empty input, boundary values, or unexpected type combinations
- Avoid trivially simple problems (no single-line solutions, no direct stdlib wrappers)"""

GENERATE_VARIANT_PROMPT = """Here is an existing programming problem:

{seed_prompt}

Generate a HARDER variant of this problem. The variant should:
- Test a similar concept but with additional complexity or constraints
- Require handling more edge cases
- NOT be solvable with the exact same approach

You must provide ALL THREE parts:

FUNCTION:
def function_name(params):
    \"\"\"Description of what the function does.
    >>> function_name(example_input)
    example_output
    \"\"\"

TESTS:
assert function_name(input1) == expected1
assert function_name(input2) == expected2
assert function_name(input3) == expected3
assert function_name(input4) == expected4
assert function_name(input5) == expected5

SOLUTION:
def function_name(params):
    # implementation
    return result"""

TOPICS = [
    "string manipulation", "list operations", "mathematical computation",
    "dictionary operations", "sorting and searching", "recursion",
    "data validation", "text processing", "number theory",
    "array transformation", "pattern matching", "combinatorics",
    "bit manipulation", "matrix operations", "stack/queue operations",
    "set operations", "encoding/decoding", "graph basics",
    "dynamic programming", "greedy algorithms",
]

DIFFICULTIES = ["hard", "hard", "very hard"]  # weighted toward hard/very hard


def generate_text(model, tokenizer, system_msg, user_msg, temperature=0.9, max_new_tokens=1500):
    """Generate text using HF model."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        if temperature > 0:
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        else:
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.0, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True)

    if "</think>" in response:
        response = response.split("</think>", 1)[1]

    return response.strip()


def generate_batch(model, tokenizer, system_msg, user_msg, n, temperature=0.7, max_new_tokens=1024):
    """Generate n completions for the same prompt in one batched forward pass."""
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    single = tokenizer(text, return_tensors="pt")
    # Repeat input n times for batched generation
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


def parse_problem(raw_text):
    """Parse model output into function, tests, solution."""
    text = raw_text.strip()

    func_match = re.search(r'FUNCTION:\s*\n(.*?)(?=\nTESTS:)', text, re.DOTALL)
    if not func_match:
        return None
    func_text = func_match.group(1).strip()

    tests_match = re.search(r'TESTS:\s*\n(.*?)(?=\nSOLUTION:)', text, re.DOTALL)
    if not tests_match:
        return None
    tests_text = tests_match.group(1).strip()

    sol_match = re.search(r'SOLUTION:\s*\n(.*?)$', text, re.DOTALL)
    if not sol_match:
        return None
    sol_text = sol_match.group(1).strip()

    name_match = re.search(r'def (\w+)\(', func_text)
    if not name_match:
        return None
    func_name = name_match.group(1)

    test_lines = [l.strip() for l in tests_text.split('\n') if l.strip().startswith('assert')]
    if len(test_lines) < 2:
        return None

    return {
        "function": func_text,
        "tests": test_lines,
        "solution": sol_text,
        "entry_point": func_name,
    }


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


def validate_problem(problem, timeout=5):
    """Verify solution passes tests."""
    code = problem["solution"] + "\n\n" + "\n".join(problem["tests"])
    return execute_code(code, timeout)


def check_solution_attempt(completion, prompt_text, tests_str):
    """Check if a completion attempt passes the tests (body-only)."""
    full_code = prompt_text + "\n" + completion + "\n\n" + tests_str
    return execute_code(full_code)


def check_fullfunction_attempt(completion, tests_str, entry_point=""):
    """Check if a full-function completion passes the tests."""
    # Extract code from markdown blocks if present
    code = completion.strip()
    code_blocks = re.findall(r'```python\s*\n(.*?)```', code, re.DOTALL)
    if code_blocks:
        code = code_blocks[-1].strip()
    else:
        code = re.sub(r'^```(?:python)?\s*\n', '', code)
        code = re.sub(r'\n?```\s*$', '', code).strip()
    full_code = code + "\n\n" + tests_str
    return execute_code(full_code)


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_trigrams(text):
    words = normalize_text(text).split()
    if len(words) < 3:
        return set(words)
    return {(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_docstring(prompt):
    m = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt


def log(msg, logfile="./gen_calibrate.log"):
    print(msg, flush=True)
    with open(logfile, "a") as f:
        f.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--target", type=int, default=200,
                        help="Target number of calibrated problems")
    parser.add_argument("--max_attempts", type=int, default=5000,
                        help="Max generation attempts")
    parser.add_argument("--samples", type=int, default=8,
                        help="Calibration samples per problem")
    parser.add_argument("--lo", type=float, default=0.2)
    parser.add_argument("--hi", type=float, default=0.8)
    parser.add_argument("--jaccard_threshold", type=float, default=0.6)
    parser.add_argument("--output", default="./calibrated_selfgen.jsonl")
    parser.add_argument("--existing", default=None,
                        help="Existing generated_problems.jsonl to calibrate first")
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter path to merge (for generating with improved model)")
    parser.add_argument("--proposer_only", action="store_true",
                        help="Generate problems only, no calibration (for adversarial curriculum setups where proposer and solver are separate)")
    parser.add_argument("--calibrate_only", action="store_true",
                        help="Calibrate existing problems only, no generation")
    parser.add_argument("--calibrate_adapter", default=None,
                        help="Adapter to merge for calibration (use with --calibrate_only)")
    parser.add_argument("--exclude_names", default=None,
                        help="File with function names to exclude (one per line)")
    parser.add_argument("--allow_name_dups", action="store_true",
                        help="Allow duplicate function names (only filter by Jaccard)")
    parser.add_argument("--full_function", action="store_true",
                        help="Ask for complete function (not body-only). For larger models.")
    parser.add_argument("--max_new_tokens", type=int, default=1024,
                        help="Max tokens for calibration generation")
    args = parser.parse_args()

    if args.proposer_only and args.calibrate_only:
        print("Error: cannot use --proposer_only and --calibrate_only together")
        sys.exit(1)

    mode = "calibrate_only" if args.calibrate_only else ("proposer_only" if args.proposer_only else "full")
    log(f"=== Generate & Calibrate (mode={mode}): target {args.target} problems ===")
    log(f"Model: {args.model}")
    if args.adapter:
        log(f"Adapter (generation): {args.adapter}")
    if args.calibrate_adapter:
        log(f"Adapter (calibration): {args.calibrate_adapter}")

    # Load excluded names
    excluded_names = set()
    if args.exclude_names and os.path.exists(args.exclude_names):
        with open(args.exclude_names) as f:
            excluded_names = {l.strip() for l in f if l.strip()}
        log(f"Excluding {len(excluded_names)} function names")

    # Load model
    log("Loading model...")
    adapter_to_merge = args.calibrate_adapter if args.calibrate_only else args.adapter
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Merge adapter if provided
    if adapter_to_merge:
        from peft import PeftModel
        log(f"Loading adapter from {adapter_to_merge}...")
        model = PeftModel.from_pretrained(model, adapter_to_merge)
        model = model.merge_and_unload()
        log("Adapter merged.")

    model.eval()
    log("Model loaded.")

    # State
    accepted_names = set()
    accepted_trigrams = []
    calibrated_count = 0

    # Resume support
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                entry = json.loads(line.strip())
                calibrated_count += 1
                accepted_names.add(entry["entry_point"])
                doc = extract_docstring(entry["prompt"])
                accepted_trigrams.append((entry["entry_point"], word_trigrams(doc)))
        log(f"Resumed: {calibrated_count} already calibrated")

    # Phase 1: Calibrate existing problems if provided
    if args.existing and os.path.exists(args.existing) and not args.proposer_only:
        log(f"\n--- Phase 1: Calibrating existing problems from {args.existing} ---")
        existing = []
        with open(args.existing) as f:
            for line in f:
                existing.append(json.loads(line.strip()))
        log(f"Loaded {len(existing)} existing problems")

        for idx, prob in enumerate(existing):
            if calibrated_count >= args.target:
                break

            ep = prob["entry_point"]
            if ep in excluded_names:
                continue
            if not args.allow_name_dups and ep in accepted_names:
                continue

            # Dedup by docstring Jaccard
            doc = extract_docstring(prob["prompt"])
            doc_tri = word_trigrams(doc)
            if any(jaccard(doc_tri, t) > args.jaccard_threshold for _, t in accepted_trigrams):
                continue

            # Calibrate: sample K solutions (batched)
            tests_str = prob["tests"]
            if args.full_function:
                user_msg = (
                    "Write the complete Python function including the def line. "
                    "Here is the signature and docstring:\n\n"
                    f"{prob['prompt']}\n\n"
                    "Return ONLY the complete function code, no explanation."
                )
            else:
                user_msg = (
                    "Complete the following Python function. "
                    "Return ONLY the function body, no explanation."
                    f"\n\n{prob['prompt']}"
                )
            completions = generate_batch(model, tokenizer, SYSTEM_SOLVE, user_msg,
                                         n=args.samples, temperature=0.7, max_new_tokens=args.max_new_tokens)
            if args.full_function:
                passes = sum(1 for c in completions
                             if c and check_fullfunction_attempt(c, tests_str, prob.get("entry_point", "")))
            else:
                passes = sum(1 for c in completions
                             if c and check_solution_attempt(c, prob["prompt"], tests_str))

            pass_rate = passes / args.samples
            in_zone = args.lo <= pass_rate <= args.hi

            if in_zone:
                calibrated_count += 1
                entry = {**prob, "pass_rate": pass_rate, "samples": args.samples}
                with open(args.output, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                accepted_names.add(ep)
                accepted_trigrams.append((ep, doc_tri))

            status = f"ACCEPT ({pass_rate:.0%})" if in_zone else f"reject ({pass_rate:.0%})"
            log(f"[existing {idx+1}/{len(existing)}] {ep}: {passes}/{args.samples} "
                f"= {status} | calibrated={calibrated_count}/{args.target}")

    # Load seed problems for variant generation
    seed_prompts = []
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                entry = json.loads(line.strip())
                seed_prompts.append(entry["prompt"])

    # Phase 2: Generate new problems (+ calibrate unless --proposer_only)
    if not args.calibrate_only and calibrated_count < args.target:
        if args.proposer_only:
            log("\n--- Phase 2: Generating problems (proposer_only, no calibration) ---")
            log(f"Target: {args.target} validated problems")
        else:
            log(f"\n--- Phase 2: Generating new problems (need {args.target - calibrated_count} more) ---")
        log(f"Strategy: 50% harder variants of {len(seed_prompts)} seeds, 50% new hard problems")

        attempts = 0
        generated = 0
        t_start = time.time()

        while calibrated_count < args.target and attempts < args.max_attempts:
            attempts += 1
            topic = random.choice(TOPICS)
            difficulty = random.choice(DIFFICULTIES)

            # Alternate: 50% variant of seed, 50% new problem
            use_variant = seed_prompts and random.random() < 0.5

            if use_variant:
                seed = random.choice(seed_prompts)
                prompt = GENERATE_VARIANT_PROMPT.format(seed_prompt=seed)
            else:
                prompt = GENERATE_PROMPT.format(topic=topic, difficulty=difficulty)

            raw = generate_text(model, tokenizer, SYSTEM_GEN, prompt,
                                temperature=0.9, max_new_tokens=1500)
            if not raw:
                continue

            problem = parse_problem(raw)
            if not problem:
                continue

            if not validate_problem(problem):
                continue

            generated += 1
            ep = problem["entry_point"]

            # Dedup
            if ep in accepted_names or ep in excluded_names:
                continue
            doc = extract_docstring(problem["function"])
            doc_tri = word_trigrams(doc)
            if any(jaccard(doc_tri, t) > args.jaccard_threshold for _, t in accepted_trigrams):
                continue

            tests_str = "\n".join(problem["tests"])

            if args.proposer_only:
                # Save without calibration — will be calibrated in a separate pass
                calibrated_count += 1
                entry = {
                    "prompt": problem["function"],
                    "tests": tests_str,
                    "solution": problem["solution"],
                    "entry_point": ep,
                    "topic": topic,
                    "difficulty": difficulty,
                    "source": "self-generated",
                    "pass_rate": -1,  # not calibrated yet
                    "samples": 0,
                }
                with open(args.output, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                accepted_names.add(ep)
                accepted_trigrams.append((ep, doc_tri))
                seed_prompts.append(problem["function"])

                elapsed = time.time() - t_start
                rate = generated / (elapsed / 60) if elapsed > 0 else 0
                log(f"[gen {generated}, att {attempts}] {ep} ({topic}) = SAVED (uncalibrated) | "
                    f"count={calibrated_count}/{args.target} [{rate:.1f} valid/min]")
                continue

            # Calibrate (batched)
            user_msg = (
                "Complete the following Python function. "
                "Return ONLY the function body, no explanation."
                f"\n\n{problem['function']}"
            )
            completions = generate_batch(model, tokenizer, SYSTEM_SOLVE, user_msg,
                                         n=args.samples, temperature=0.7, max_new_tokens=args.max_new_tokens)
            passes = sum(1 for c in completions
                         if c and check_solution_attempt(c, problem["function"], tests_str))

            pass_rate = passes / args.samples
            in_zone = args.lo <= pass_rate <= args.hi

            if in_zone:
                calibrated_count += 1
                entry = {
                    "prompt": problem["function"],
                    "tests": tests_str,
                    "solution": problem["solution"],
                    "entry_point": ep,
                    "topic": topic,
                    "difficulty": difficulty,
                    "source": "self-generated",
                    "pass_rate": pass_rate,
                    "samples": args.samples,
                }
                with open(args.output, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                accepted_names.add(ep)
                accepted_trigrams.append((ep, doc_tri))
                seed_prompts.append(problem["function"])

            elapsed = time.time() - t_start
            rate = generated / (elapsed / 60) if elapsed > 0 else 0
            status = f"ACCEPT ({pass_rate:.0%})" if in_zone else f"reject ({pass_rate:.0%})"
            log(f"[gen {generated}, att {attempts}] {ep} ({topic}) = {status} | "
                f"calibrated={calibrated_count}/{args.target} [{rate:.1f} valid/min]")

    log(f"\n{'='*60}")
    log(f"DONE: {calibrated_count} calibrated problems in {args.output}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
