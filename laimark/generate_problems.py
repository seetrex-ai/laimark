"""
Self-generated problem bank.

The model generates coding problems as triples:
  1. Function signature + docstring
  2. Test cases (assert statements)
  3. Reference solution

A Python subprocess validates that the reference solution passes the tests
with a 5-second timeout. Valid problems enter the candidate pool for the
calibration stage.

Usage:
    python generate_problems.py [--count 1000] [--model qwen3-8b]
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "generated_problems.jsonl")
LOG_FILE = os.path.join(SCRIPT_DIR, "gen_problems_progress.log")
OLLAMA_API = "http://localhost:11434/api/chat"

GENERATE_PROMPT = """Generate a Python programming problem. You must provide ALL THREE parts:

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
- Include at least 3 test cases covering normal and edge cases
- The solution must be correct and pass all tests
- Difficulty: {difficulty}
- Topic: {topic}"""

TOPICS = [
    "string manipulation", "list operations", "mathematical computation",
    "dictionary operations", "sorting and searching", "recursion",
    "data validation", "text processing", "number theory",
    "array transformation", "pattern matching", "graph/tree basics",
    "combinatorics", "bit manipulation", "matrix operations",
    "stack/queue operations", "set operations", "file path manipulation",
    "date/time calculations", "encoding/decoding",
]

DIFFICULTIES = ["hard", "very hard"]


def call_ollama(model, user_msg, temperature=0.9, max_tokens=1500, seed=None):
    options = {"temperature": temperature, "num_predict": max_tokens}
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert Python programmer who creates programming problems. /no_think"},
            {"role": "user", "content": user_msg},
        ],
        "think": False,
        "stream": False,
        "options": options,
    }
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except Exception:
        return ""


def parse_problem(raw_text):
    """Parse the model's output into function, tests, and solution."""
    text = raw_text.strip()

    # Extract FUNCTION section
    func_match = re.search(r'FUNCTION:\s*\n(.*?)(?=\nTESTS:)', text, re.DOTALL)
    if not func_match:
        return None
    func_text = func_match.group(1).strip()

    # Extract TESTS section
    tests_match = re.search(r'TESTS:\s*\n(.*?)(?=\nSOLUTION:)', text, re.DOTALL)
    if not tests_match:
        return None
    tests_text = tests_match.group(1).strip()

    # Extract SOLUTION section
    sol_match = re.search(r'SOLUTION:\s*\n(.*?)$', text, re.DOTALL)
    if not sol_match:
        return None
    sol_text = sol_match.group(1).strip()

    # Extract function name
    name_match = re.search(r'def (\w+)\(', func_text)
    if not name_match:
        return None
    func_name = name_match.group(1)

    # Extract test assertions
    test_lines = [l.strip() for l in tests_text.split('\n') if l.strip().startswith('assert')]
    if len(test_lines) < 2:
        return None

    return {
        "function": func_text,
        "tests": test_lines,
        "solution": sol_text,
        "entry_point": func_name,
    }


def validate_problem(problem, timeout=5):
    """Run solution + tests to verify the problem is valid."""
    code = problem["solution"] + "\n\n" + "\n".join(problem["tests"])
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            tmpfile = f.name
        result = subprocess.run(
            [sys.executable, tmpfile],
            capture_output=True, timeout=timeout, text=True
        )
        os.unlink(tmpfile)
        return result.returncode == 0
    except Exception:
        try:
            os.unlink(tmpfile)
        except Exception:
            pass
        return False


def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--model", default="qwen3-8b-laimark")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output file for generated problems")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for topic/difficulty choices and Ollama sampling")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    OUTPUT_FILE = args.output

    # Resume
    existing = 0
    if os.path.exists(OUTPUT_FILE):
        existing = sum(1 for _ in open(OUTPUT_FILE))
    log(f"=== Generating {args.count} problems with {args.model} ===")
    log(f"Existing: {existing}")

    # Verify Ollama
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        log("Ollama OK")
    except Exception:
        log("ERROR: Ollama not running")
        sys.exit(1)

    attempts = 0
    valid = existing
    t_start = time.time()

    while valid < args.count:
        attempts += 1
        topic = random.choice(TOPICS)
        difficulty = random.choice(DIFFICULTIES)

        prompt = GENERATE_PROMPT.format(topic=topic, difficulty=difficulty)
        raw = call_ollama(args.model, prompt, temperature=0.9, seed=args.seed + attempts)

        if not raw:
            continue

        problem = parse_problem(raw)
        if not problem:
            if attempts % 10 == 0:
                log(f"  [{attempts} attempts, {valid} valid] parse failed")
            continue

        if validate_problem(problem):
            valid += 1
            # Build training format
            entry = {
                "prompt": problem["function"],
                "tests": "\n".join(problem["tests"]),
                "solution": problem["solution"],
                "entry_point": problem["entry_point"],
                "topic": topic,
                "difficulty": difficulty,
                "source": "self-generated",
            }
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")

            elapsed = time.time() - t_start
            rate = (valid - existing) / (elapsed / 60) if elapsed > 0 else 0
            log(f"[{valid}/{args.count}] {problem['entry_point']} ({topic}, {difficulty}) "
                f"[{rate:.1f}/min, attempts: {attempts}]")
        else:
            if attempts % 10 == 0:
                log(f"  [{attempts} attempts, {valid} valid] validation failed")

    elapsed = time.time() - t_start
    log(f"\nDone. {valid} valid problems from {attempts} attempts "
        f"({valid/attempts*100:.1f}% success) in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
