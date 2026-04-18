# LAIMARK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://github.com/seetrex-ai/laimark)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![CI](https://github.com/seetrex-ai/laimark/actions/workflows/ci.yml/badge.svg)](https://github.com/seetrex-ai/laimark/actions)

LAIMARK (Local AI Metacognitive Agent with Recursive Knowledge) studies whether a language model can generate its own training curriculum and improve via reinforcement learning from verifiable reward. Four things run on a single base model: a prompt-evolution loop, a GRPO weight update, prompt re-optimization on the updated weights, and a problem-generation step that feeds the next GRPO round. Nothing outside the model participates, other than the Python interpreter used to check that generated code passes its own tests.

> **Paper:** [LAIMARK: Gains and Structural Limits of Self-Generated Curricula in Reinforcement Learning from Verifiable Reward](paper/Laimark.pdf) (April 2026)

## What this is

[DeepSeek-R1](https://arxiv.org/abs/2501.12948) and related RLVR systems improve base models using curated external problem sets paired with automatic evaluators. We ask what happens when the problem set comes from the model itself.

On HumanEval with Qwen3-8B (HuggingFace fp16 harness):

| Configuration | External problems | pass@1 |
|---|---|---|
| Base model | — | 63.4% |
| **GRPO, self-generated (G=4)** | **0** | **76.8%** |
| GRPO, curated (HumanEval + MBPP) | hundreds | 84.1% |

Self-generation with calibration captures about 65% of the curated-benchmark gain on two orders of magnitude less data.

Three limits cap this approach. First, iteration does not accumulate: a second GRPO round trained on problems calibrated against the first-round checkpoint converges back to it. Second, a curriculum dominated by a single task type (for example, 84% abduction-style problems) drops pass@1 to 61.0% — below the pre-training baseline — by shifting the output-format prior in a direction that misfits HumanEval. Third, at 32B parameters the learnability window closes entirely: the base model already solves nearly every problem it can formulate with a verified reference solution, so the selection criterion has nothing to accept.

## Repository structure

```
laimark/
  generate_problems.py              stage 1 — candidate problem generation
  calibrate_problems.py             stage 2 — calibration against current policy
  train_grpo.py                     stage 3 — GRPO training (main entry point)
  eval_adapter.py                   stage 4 — HumanEval evaluation (HF fp16)
  generate_and_calibrate_hf.py      HF-only pipeline (no Ollama required)
  generate_deduction_abduction.py   §5.2 — task-type diversity generation
  calibrate_deduction_abduction.py  §5.2 — calibration with type mix
  train_dpo.py, train_lora.py       §4 — baselines for failed approaches
  export_gguf.sh                    GGUF export for deployment

data/calibrated_selfgen.jsonl       22 self-generated calibrated problems
                                    (the curriculum loaded by --selfgen_only)
docs/experimental-results.md        consolidated results tables
paper/laimark.tex                   paper source (+ references.bib)
paper/Laimark.pdf                   compiled paper
```

## Installation

```bash
git clone https://github.com/seetrex-ai/laimark.git
cd laimark
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

Training and evaluation require a GPU with enough VRAM for Qwen3-8B in fp16 (we used an NVIDIA A100 80GB).

## Quick start

Train directly on the shipped curriculum (`data/calibrated_selfgen.jsonl`, 22 problems) and evaluate:

```bash
python laimark/train_grpo.py --selfgen_only --num_generations 4 --epochs 2 --seed 42
python laimark/eval_adapter.py --adapter_path ./grpo_output/final --seed 42
```

To regenerate the curriculum from scratch, the full pipeline is:

```bash
# 1. Generate ~1000 candidate problems (Ollama local backend)
python laimark/generate_problems.py --count 1000 --seed 42

# 2. Calibrate against the base model — keep problems in the learnability window [0.2, 0.8]
#    Roughly one candidate in ten survives verification and calibration.
python laimark/calibrate_problems.py --samples 8 --lo 0.2 --hi 0.8 --seed 42

# 3. GRPO on the calibrated self-generated curriculum, no external benchmarks
python laimark/train_grpo.py --selfgen_only --num_generations 4 --epochs 2 --seed 42

# 4. Evaluate the trained adapter on HumanEval
python laimark/eval_adapter.py --adapter_path ./grpo_output/final --seed 42
```

Every stage accepts `--seed` (default 42) and passes it to the underlying sampler (Ollama's `seed` option, `torch.manual_seed`, `GRPOConfig.seed`). Regeneration from step 1 is deterministic up to Ollama kernel non-determinism on multi-GPU setups; the shipped `data/calibrated_selfgen.jsonl` is a snapshot of one specific run.

`--num_generations` drives the result at small data volumes (paper §4). On the same 22-problem curriculum, moving from 2 to 4 gains 6.7 pass@1 points; adding more problems at `G=2` does less.

## Reproducing the paper's main numbers

Each of the four configurations in the paper corresponds to one pipeline run:

| Paper section | Command | Pass@1 |
|---|---|---|
| §4 — selfgen v2 | `train_grpo.py --selfgen_only --num_generations 4` | 76.8% |
| §4 — curated | `train_grpo.py --num_generations 4` (default: HE+MBPP mix) | 84.1% |
| §5.1 — iteration | `train_grpo.py --selfgen_only --base_adapter grpo_output/final` | 76.8% (no gain) |
| §5.2 — task-type R1 | `calibrate_deduction_abduction.py` then GRPO | 61.0% |
| §5.3 — 32B base | `eval_adapter.py --model Qwen3-32B --full_function` | 89.0% |

Trained LoRA adapters are not committed. Re-running the pipeline with `--seed 42` reproduces the curriculum and training trajectory; exact pass@1 parity depends on the hardware and on GPU-nondeterministic operations in PyTorch.

## Safety

> [!WARNING]
> The pipeline runs model-generated Python code in a subprocess with a 5-second timeout. The sandbox does not block filesystem, network, or environment access from the executed code. Run inside a container or disposable VM; see [SECURITY.md](SECURITY.md).

## Citation

```bibtex
@article{tabares2026laimark,
  title = {LAIMARK: Gains and Structural Limits of Self-Generated Curricula
           in Reinforcement Learning from Verifiable Reward},
  author = {Tabares Montilla, Jes{\'u}s},
  year = {2026},
  url = {https://github.com/seetrex-ai/laimark}
}
```

## Acknowledgements

The prompt-evolution component of LAIMARK builds on the open-ended evolution
framework of [Darwin Gödel Machine (Zhang et al., 2025)](https://arxiv.org/abs/2505.22954);
the weight-update and self-generation components are new to this work.

```bibtex
@article{zhang2025darwin,
  title={Darwin G{\"o}del Machine: Open-Ended Evolution of Self-Improving Agents},
  author={Zhang, Jenny and Hu, Shengran and Lu, Cong and Lange, Robert and Clune, Jeff},
  journal={arXiv preprint arXiv:2505.22954},
  year={2025}
}
```

## License

[Apache 2.0](LICENSE). Upstream attribution in [NOTICE](NOTICE).
