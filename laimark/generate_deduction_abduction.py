"""
Transform induction problems into deduction and abduction problems.

Reads calibrated induction problems (function + tests) and emits two variants:
- Deduction: "given this code + input, what output?" — verify by execution.
- Abduction: "given this code + output, what input?" — verify by
  output-equivalence.

Deterministic transformation; no LLM call required. Parses assert statements
to extract (func_name, input, expected_output) triples.

Usage:
    python generate_deduction_abduction.py \
        --input calibrated_selfgen.jsonl \
        --output deduction_abduction.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile


def parse_assert(line):
    """Parse 'assert func(args) == expected' into components.

    Returns (func_name, input_expr, expected_expr) or None.
    """
    line = line.strip()
    if not line.startswith("assert"):
        return None

    # Remove 'assert ' prefix
    expr = line[len("assert"):].strip()

    # Match: func_name(args) == expected
    # Handle nested parens in args by counting depth
    m = re.match(r'(\w+)\(', expr)
    if not m:
        return None

    func_name = m.group(1)
    start = m.end()

    # Find matching closing paren
    depth = 1
    i = start
    while i < len(expr) and depth > 0:
        if expr[i] == '(':
            depth += 1
        elif expr[i] == ')':
            depth -= 1
        i += 1

    if depth != 0:
        return None

    input_expr = expr[start:i - 1]
    remainder = expr[i:].strip()

    # Match == expected
    if not remainder.startswith("=="):
        return None

    expected_expr = remainder[2:].strip()
    # Remove trailing comment if present
    if "#" in expected_expr:
        expected_expr = expected_expr[:expected_expr.index("#")].strip()

    if not input_expr or not expected_expr:
        return None

    return func_name, input_expr, expected_expr


def validate_triple(solution, func_name, input_expr, expected_expr, timeout=5):
    """Verify that func(input) == expected using the reference solution."""
    code = f"{solution}\n_r = {func_name}({input_expr})\nassert _r == {expected_expr}, f'got {{repr(_r)}}'"
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
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


def make_deduction(solution, func_name, input_expr, expected_expr):
    """Create a deduction problem: predict the output."""
    prompt = (
        f"Given the following Python function:\n\n"
        f"{solution}\n\n"
        f"What is the output of `{func_name}({input_expr})`?\n"
        f"Return ONLY the output value as a Python expression, no explanation."
    )
    return {
        "prompt": prompt,
        "problem_type": "deduction",
        "dd_code": solution,
        "dd_func_name": func_name,
        "dd_input": input_expr,
        "dd_expected": expected_expr,
    }


def make_abduction(solution, func_name, input_expr, expected_expr):
    """Create an abduction problem: find an input that produces the output."""
    prompt = (
        f"Given the following Python function:\n\n"
        f"{solution}\n\n"
        f"Find an input `x` such that `{func_name}(x) == {expected_expr}`.\n"
        f"Return ONLY the input value as a Python expression, no explanation."
    )
    return {
        "prompt": prompt,
        "problem_type": "abduction",
        "ab_code": solution,
        "ab_func_name": func_name,
        "ab_expected": expected_expr,
    }


def process_problem(problem):
    """Extract deduction and abduction problems from an induction problem."""
    solution = problem.get("solution", "")
    tests_str = problem.get("tests", "")

    if not solution or not tests_str:
        return [], []

    test_lines = [l.strip() for l in tests_str.split("\n") if l.strip().startswith("assert")]

    deductions = []
    abductions = []

    for line in test_lines:
        parsed = parse_assert(line)
        if not parsed:
            continue

        func_name, input_expr, expected_expr = parsed

        # Validate the triple against the reference solution
        if not validate_triple(solution, func_name, input_expr, expected_expr):
            continue

        deductions.append(make_deduction(solution, func_name, input_expr, expected_expr))
        abductions.append(make_abduction(solution, func_name, input_expr, expected_expr))

    return deductions, abductions


def main():
    parser = argparse.ArgumentParser(
        description="Transform induction problems into deduction and abduction"
    )
    parser.add_argument("--input", required=True,
                        help="Input JSONL with calibrated induction problems")
    parser.add_argument("--output", required=True,
                        help="Output JSONL with deduction and abduction problems")
    parser.add_argument("--max_per_problem", type=int, default=3,
                        help="Max deduction/abduction problems per induction problem")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    problems = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))

    print(f"Loaded {len(problems)} induction problems")

    total_deduction = 0
    total_abduction = 0
    total_failed = 0

    with open(args.output, "w") as out:
        for idx, prob in enumerate(problems):
            deductions, abductions = process_problem(prob)

            # Limit per problem to avoid imbalance
            deductions = deductions[:args.max_per_problem]
            abductions = abductions[:args.max_per_problem]

            for d in deductions:
                d["source_entry_point"] = prob.get("entry_point", "unknown")
                d["source_pass_rate"] = prob.get("pass_rate", 0.5)
                out.write(json.dumps(d) + "\n")
                total_deduction += 1

            for a in abductions:
                a["source_entry_point"] = prob.get("entry_point", "unknown")
                a["source_pass_rate"] = prob.get("pass_rate", 0.5)
                out.write(json.dumps(a) + "\n")
                total_abduction += 1

            n_tests = len([l for l in prob.get("tests", "").split("\n")
                          if l.strip().startswith("assert")])
            n_valid = len(deductions)
            if n_tests > 0 and n_valid == 0:
                total_failed += 1

            print(f"[{idx + 1}/{len(problems)}] {prob.get('entry_point', '?')}: "
                  f"{n_valid} deduction + {len(abductions)} abduction "
                  f"(from {n_tests} asserts)")

    print(f"\nDone: {total_deduction} deduction + {total_abduction} abduction "
          f"= {total_deduction + total_abduction} total")
    print(f"Problems with 0 valid triples: {total_failed}/{len(problems)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
