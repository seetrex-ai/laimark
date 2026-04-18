# Contributing

Contributions are welcome, especially replications on other base models or benchmarks.

## Setup

```bash
git clone https://github.com/seetrex-ai/laimark.git
cd laimark
python -m venv venv
source venv/bin/activate         # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

## Before submitting

1. Scripts should lint cleanly: `ruff check laimark/`
2. New pipeline stages should extend an existing script where possible rather than introducing parallel copies
3. Reproducibility: any new training run should document the random seed, base model, and the full command used

## Pull requests

- One logical change per PR
- Descriptive commit message (conventional commits preferred: `feat:`, `fix:`, `docs:`, `refactor:`)
- Reference any relevant issue

## Reporting experimental results

If you reproduce or refute a result from the paper on a different model or benchmark, please open an issue with:

- Base model + precision
- Exact command used
- Pass@1 (or equivalent) with seed
- Evaluation harness (the paper uses HuggingFace fp16 HumanEval)

Negative replications are as valuable as positive ones.
