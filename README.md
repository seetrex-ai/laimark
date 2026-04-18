# LAIMARK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)](https://github.com/seetrex-ai/laimark)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![CI](https://github.com/seetrex-ai/laimark/actions/workflows/ci.yml/badge.svg)](https://github.com/seetrex-ai/laimark/actions)

**Gains and structural limits of self-generated curricula in reinforcement learning from verifiable reward.**

LAIMARK (Local AI Metacognitive Agent with Recursive Knowledge) is a closed-loop self-evolution system that composes four components on a single base model: prompt evolution, weight update via GRPO, self-refinement, and self-generated curricula. No external agent, judge, or human-in-the-loop at any stage.

> **Paper:** [LAIMARK: Gains and Structural Limits of Self-Generated Curricula in Reinforcement Learning from Verifiable Reward](paper/laimark.tex) (April 2026)

## What this is

RLVR systems like [DeepSeek-R1](https://arxiv.org/abs/2501.12948) improve base models on reasoning benchmarks using curated external problems with automatic evaluators. LAIMARK asks whether the same gain can be obtained with the model generating its own problems.

Results on HumanEval with Qwen3-8B (official HuggingFace fp16 harness):

| Configuration | External problems | pass@1 |
|---|---|---|
| Base model | — | 63.4% |
| **GRPO, self-generated (G=4)** | **0** | **76.8%** |
| GRPO, curated (HumanEval + MBPP) | hundreds | 84.1% |

Self-generation with calibration recovers roughly two thirds of the curated-benchmark improvement using training data two orders of magnitude smaller.

The paper also documents three structural limits that cap this approach:

1. **Iteration does not accumulate.** A second GRPO round on problems calibrated against the first-round checkpoint fails to improve over it.
2. **Task-type imbalance destroys transfer.** A curriculum dominated by abduction-style problems drops pass@1 to 61.0%, below the pre-training baseline.
3. **Inapplicability at scale.** With Qwen3-32B (89.0% base pass@1 without training), less than 1% of self-generated problems pass the learnability-window filter — a generator capable enough to produce well-formed verified problems is also capable enough to solve them.

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

docs/experimental-results.md        consolidated results tables
paper/laimark.tex                   paper source (+ references.bib)
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

One full round of the closed loop, end to end:

```bash
# 1. Generate candidate problems
python laimark/generate_problems.py --count 1000

# 2. Calibrate against the base model — keep problems in the learnability window
python laimark/calibrate_problems.py --samples 8 --lo 0.2 --hi 0.8

# 3. GRPO on the calibrated self-generated curriculum, no external benchmarks
python laimark/train_grpo.py --selfgen_only --num_generations 4 --epochs 2

# 4. Evaluate the trained adapter on HumanEval (official HF fp16 harness)
python laimark/eval_adapter.py --adapter grpo_output/final
```

The `--num_generations 4` flag is the single most consequential hyperparameter at small curriculum sizes (see paper §4); halving it to 2 drops the result by roughly 7 pass@1 points on the same data.

## Reproducing the paper's main numbers

Each of the four configurations in the paper corresponds to one pipeline run:

| Paper section | Command | Pass@1 |
|---|---|---|
| §4 — selfgen v2 | `train_grpo.py --selfgen_only --num_generations 4` | 76.8% |
| §4 — curated | `train_grpo.py --num_generations 4` (default: HE+MBPP mix) | 84.1% |
| §5.1 — iteration | `train_grpo.py --selfgen_only --base_adapter grpo_output/final` | 76.8% (no gain) |
| §5.2 — task-type R1 | `calibrate_deduction_abduction.py` then GRPO | 61.0% |
| §5.3 — 32B base | `eval_adapter.py --model Qwen3-32B --full_function` | 89.0% |

Trained LoRA adapters and raw output logs are not committed — the pipeline regenerates them deterministically with fixed random seeds.

## Safety

> [!WARNING]
> This repository executes untrusted, model-generated Python code in a subprocess-based sandbox with a 5-second timeout.
> The sandbox is sufficient for HumanEval-style benchmarks but is not a hardened isolation boundary against adversarial payloads.
> Run the pipeline in a Docker container, a disposable VM, or a sandboxed user account on a machine without sensitive data or privileged network access.

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
