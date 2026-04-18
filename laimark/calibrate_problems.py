"""
Calibrate self-generated problems by learnability.

For each candidate, sample K solutions from the model and compute pass rate.
Keep only problems in the learnability window [lo, hi] where the model is
neither trivially solving nor completely failing; this is where GRPO
produces non-zero group advantage.

Deduplicates by function name and docstring Jaccard similarity
(word trigrams, threshold 0.6).

Input:  generated_problems.jsonl (from generate_problems.py)
Output: calibrated_problems.jsonl (filtered, ready for train_grpo.py)

Usage:
    python calibrate_problems.py [--samples 8] [--lo 0.2] [--hi 0.8]
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
INPUT_FILE = os.path.join(SCRIPT_DIR, "generated_problems.jsonl")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "calibrated_problems.jsonl")
LOG_FILE = os.path.join(SCRIPT_DIR, "calibrate_progress.log")
OLLAMA_API = "http://localhost:11434/api/chat"

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


# === Ollama interaction ===

def call_ollama(model, prompt_text, temperature=0.7, max_tokens=1024, seed=None):
    """Call Ollama native API to generate a solution attempt."""
    user_msg = (
        "Complete the following Python function. "
        "Return ONLY the function body, no explanation."
        f"\n\n{prompt_text}"
    )
    options = {"temperature": temperature, "num_predict": max_tokens}
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_L2B},
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


# === Code execution ===

def execute_code(code_str, timeout=5):
    """Run code in a subprocess. Returns True if exit code 0."""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
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


def check_solution(completion, prompt_text, tests):
    """Check if a completion passes the problem's tests.

    The completion is the function body. We reconstruct the full function
    using the prompt (which contains the signature + docstring) and append
    the test assertions.
    """
    # Build full code: signature+docstring from prompt, body from completion
    full_code = prompt_text + "\n" + completion + "\n\n" + tests
    return execute_code(full_code)


# === Deduplication ===

def normalize_text(text):
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_trigrams(text):
    """Extract set of word trigrams from normalized text."""
    words = normalize_text(text).split()
    if len(words) < 3:
        return set(words)  # fallback: use individual words
    return {(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)}


def jaccard_similarity(set_a, set_b):
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def extract_docstring(prompt_text):
    """Extract the docstring content from a function prompt."""
    match = re.search(r'"""(.*?)"""', prompt_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"'''(.*?)'''", prompt_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return prompt_text


# === Logging ===

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# === Main calibration loop ===

def main():
    parser = argparse.ArgumentParser(description="Calibrate problems by learnability")
    parser.add_argument("--input", default=INPUT_FILE, help="Input problems JSONL")
    parser.add_argument("--output", default=OUTPUT_FILE, help="Output calibrated JSONL")
    parser.add_argument("--model", default="qwen3-8b-laimark", help="Ollama model name")
    parser.add_argument("--samples", type=int, default=8, help="Solutions to sample per problem")
    parser.add_argument("--lo", type=float, default=0.2, help="Min pass rate (inclusive)")
    parser.add_argument("--hi", type=float, default=0.8, help="Max pass rate (inclusive)")
    parser.add_argument("--jaccard_threshold", type=float, default=0.6,
                        help="Jaccard similarity threshold for dedup")
    parser.add_argument("--resume", action="store_true", help="Skip already calibrated problems")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for Ollama sampling (each sample uses seed + k)")
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    # Load candidates
    if not os.path.exists(args.input):
        log(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    candidates = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    log(f"Loaded {len(candidates)} candidate problems from {args.input}")

    # Resume support: load already calibrated problem IDs
    calibrated_ids = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    calibrated_ids.add(entry["entry_point"])
        log(f"Resume: {len(calibrated_ids)} problems already calibrated")

    # Verify Ollama
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
        log("Ollama OK")
    except Exception:
        log("ERROR: Ollama not running at localhost:11434")
        sys.exit(1)

    # State for deduplication
    accepted_names = set()
    accepted_trigrams = []  # list of (entry_point, trigram_set)

    # If resuming, populate dedup state from existing output
    if calibrated_ids and os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    accepted_names.add(entry["entry_point"])
                    doc = extract_docstring(entry["prompt"])
                    accepted_trigrams.append((entry["entry_point"], word_trigrams(doc)))

    # Calibration loop
    stats = {"total": 0, "sampled": 0, "in_zone": 0, "deduped": 0, "accepted": 0}
    t_start = time.time()

    for idx, problem in enumerate(candidates):
        stats["total"] += 1
        entry_point = problem["entry_point"]

        # Skip if already done (resume)
        if entry_point in calibrated_ids:
            continue

        # === Dedup by function name ===
        if entry_point in accepted_names:
            stats["deduped"] += 1
            continue

        # === Dedup by docstring similarity ===
        doc = extract_docstring(problem["prompt"])
        doc_trigrams = word_trigrams(doc)
        is_duplicate = False
        for _, existing_trigrams in accepted_trigrams:
            if jaccard_similarity(doc_trigrams, existing_trigrams) > args.jaccard_threshold:
                is_duplicate = True
                break
        if is_duplicate:
            stats["deduped"] += 1
            continue

        # === Sample K solutions and compute pass rate ===
        passes = 0
        for k in range(args.samples):
            sample_seed = args.seed * 1000 + stats["sampled"] * 100 + k
            completion = call_ollama(args.model, problem["prompt"],
                                     temperature=0.7, seed=sample_seed)
            if not completion:
                continue
            if check_solution(completion, problem["prompt"], problem["tests"]):
                passes += 1
        stats["sampled"] += 1

        pass_rate = passes / args.samples
        in_zone = args.lo <= pass_rate <= args.hi

        if in_zone:
            stats["in_zone"] += 1
            stats["accepted"] += 1

            # Add calibration metadata
            calibrated_entry = {
                **problem,
                "pass_rate": pass_rate,
                "samples": args.samples,
                "learnability_zone": [args.lo, args.hi],
            }

            with open(args.output, "a", encoding="utf-8") as f:
                f.write(json.dumps(calibrated_entry) + "\n")

            # Update dedup state
            accepted_names.add(entry_point)
            accepted_trigrams.append((entry_point, doc_trigrams))

        # Progress logging
        elapsed = time.time() - t_start
        rate = stats["sampled"] / (elapsed / 60) if elapsed > 0 else 0
        zone_label = f"ACCEPT ({pass_rate:.0%})" if in_zone else f"reject ({pass_rate:.0%})"
        log(
            f"[{idx + 1}/{len(candidates)}] {entry_point}: {passes}/{args.samples} "
            f"= {zone_label} | accepted={stats['accepted']} dedup={stats['deduped']} "
            f"[{rate:.1f} prob/min]"
        )

    # Final summary
    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"Calibration complete in {elapsed / 60:.1f} min")
    log(f"  Candidates:  {stats['total']}")
    log(f"  Sampled:     {stats['sampled']}")
    log(f"  In zone:     {stats['in_zone']} ({stats['in_zone'] / max(stats['sampled'], 1) * 100:.1f}%)")
    log(f"  Deduped:     {stats['deduped']}")
    log(f"  Accepted:    {stats['accepted']}")
    log(f"  Output:      {args.output}")

    # Distribution analysis
    if os.path.exists(args.output):
        pass_rates = []
        topics = {}
        with open(args.output, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    pass_rates.append(entry.get("pass_rate", 0))
                    t = entry.get("topic", "unknown")
                    topics[t] = topics.get(t, 0) + 1

        if pass_rates:
            avg_pr = sum(pass_rates) / len(pass_rates)
            log("\nPass rate distribution (accepted):")
            log(f"  Mean: {avg_pr:.2f}")
            log(f"  Min:  {min(pass_rates):.2f}")
            log(f"  Max:  {max(pass_rates):.2f}")
            log("\nTopic distribution:")
            for t, count in sorted(topics.items(), key=lambda x: -x[1]):
                log(f"  {t}: {count}")


if __name__ == "__main__":
    main()
