"""
Validate the self-generated problem dataset (data/calibrated_selfgen.jsonl).

The reward function in laimark/train_grpo.py passes the `tests` field of
every problem through exec() in a subprocess during GRPO training. Any
malicious code in this field would be executed on every user who trains
the model. This validator parses each problem's `tests` and `solution`
fields and rejects imports or calls outside a conservative allowlist.

Run as:
    python scripts/validate_dataset.py [path]

Default path: data/calibrated_selfgen.jsonl

Exit codes:
    0  all problems pass
    1  one or more problems contain disallowed constructs

Policy:
    - `import X` / `from X import ...`: X's top-level module must be in
      ALLOWED_MODULES.
    - `__import__(name)`: name must be a string literal whose top-level
      module is in ALLOWED_MODULES. Dynamic arguments are rejected.
    - Calls to exec, eval, compile, open, globals, locals, vars,
      breakpoint, input, help are rejected.
    - Syntax errors are rejected.

The allowlist is deliberately narrow. Expanding it requires a deliberate
decision (and a review of which upstream problems need the extra module).
"""

import ast
import json
import sys
from pathlib import Path

ALLOWED_MODULES = {
    "math", "re", "itertools", "collections", "functools",
    "string", "operator", "bisect", "heapq", "typing",
    "datetime", "statistics", "random", "fractions",
    "decimal", "numbers", "copy", "unicodedata",
    "enum", "dataclasses", "abc",
}

BANNED_CALLS = {
    "exec", "eval", "compile",
    "open", "input", "breakpoint",
    "globals", "locals", "vars", "help",
}


def check_tree(tree: ast.AST, ctx: str) -> list[str]:
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    errors.append(
                        f"{ctx}: disallowed import `{alias.name}` "
                        f"(top-level `{root}` not in allowlist)"
                    )
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root not in ALLOWED_MODULES:
                errors.append(
                    f"{ctx}: disallowed from-import `from {mod} import ...` "
                    f"(top-level `{root}` not in allowlist)"
                )
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                if fn.id == "__import__":
                    if not node.args:
                        errors.append(f"{ctx}: __import__() with no argument")
                        continue
                    arg = node.args[0]
                    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                        errors.append(
                            f"{ctx}: __import__() with non-literal argument "
                            f"(dynamic imports are not allowed)"
                        )
                        continue
                    root = arg.value.split(".")[0]
                    if root not in ALLOWED_MODULES:
                        errors.append(
                            f"{ctx}: __import__('{arg.value}') — "
                            f"top-level `{root}` not in allowlist"
                        )
                elif fn.id in BANNED_CALLS:
                    errors.append(f"{ctx}: disallowed call `{fn.id}(...)`")
    return errors


def validate(path: Path) -> int:
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    n_problems = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n_problems += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"line {i}: JSON decode error: {e}")
                continue
            for key in ("tests", "solution"):
                code = data.get(key, "")
                if not code:
                    continue
                try:
                    tree = ast.parse(code)
                except SyntaxError as e:
                    all_errors.append(
                        f"line {i} {key} (entry_point={data.get('entry_point')}): "
                        f"syntax error: {e.msg} at line {e.lineno}"
                    )
                    continue
                ctx = f"line {i} {key} (entry_point={data.get('entry_point')})"
                all_errors.extend(check_tree(tree, ctx))

    if all_errors:
        print(f"Dataset validation FAILED on {path}:", file=sys.stderr)
        for err in all_errors:
            print(f"  {err}", file=sys.stderr)
        print(f"\n{len(all_errors)} error(s) across {n_problems} problem(s)", file=sys.stderr)
        return 1

    print(f"Dataset validation passed: {n_problems} problems in {path}")
    return 0


def main() -> int:
    default = Path(__file__).parent.parent / "data" / "calibrated_selfgen.jsonl"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    return validate(target)


if __name__ == "__main__":
    sys.exit(main())
