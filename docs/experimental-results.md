# LAIMARK — Experimental Results

All pass@1 numbers use the official HuggingFace fp16 HumanEval evaluator.
Training on an NVIDIA A100 80GB; greedy decoding at evaluation.

See the [paper](../paper/laimark.tex) for full methodology and discussion.

## 1. Main Result

GRPO on 22 self-generated calibrated problems reaches 76.8% pass@1 — about
65% of the 20.7-point gain that the same GRPO run produces on 664 curated
problems from HumanEval and MBPP.

| Configuration | External problems | $G$ | Pass@1 | Gain captured |
|---|---|---|---|---|
| Base (Qwen3-8B) | — | — | 63.4% | — |
| Self-generated, $G{=}2$ | 0 | 2 | 70.1% | 32% |
| Self-generated, $G{=}4$ | 0 | 4 | **76.8%** | **65%** |
| Curated (HumanEval + MBPP) | hundreds | 4 | 84.1% | 100% |

## 2. Within-Batch Contrast Dominates Data Volume

At small curriculum sizes, GRPO's group size $G$ matters more than adding
problems. The two rows above with identical data and different $G$ differ
by 6.7 pass@1 points; the mechanism is the fraction of training steps with
non-degenerate reward variance.

Expected non-degenerate fraction at mean calibrated pass rate
$\hat{p} \approx 0.53$:
- $G = 2$: $2\hat{p}(1-\hat{p}) \approx 0.50$
- $G = 4$: $1 - \hat{p}^4 - (1-\hat{p})^4 \approx 0.88$

Measured: approximately 30% for $G{=}2$, approximately 80% for $G{=}4$.

## 3. Where Self-Generated Curricula Fail

### 3.1 Iteration does not accumulate

| Round | Base for training | Pass@1 |
|---|---|---|
| 1 (selfgen v2) | Qwen3-8B base | 76.8% |
| 2a (from base, problems calibrated against v2) | Qwen3-8B base | 65.2% |
| 2b (from v2 merged, problems calibrated against v2) | v2 | 76.8% |

Round 2b lands on 76.8%, identical to its starting checkpoint. Training on
problems chosen for sitting in v2's learnability window raises v2's pass
rate on those problems to 1, and leaves the rest untouched. A second
generation pass with v2 as proposer produces 1.7% survival against the
same criterion — the window is shrinking faster than the curriculum can
refill it.

### 3.2 Task-type imbalance destroys transfer

Two rounds of adversarial curriculum co-evolution with different
calibration regimes:

| Round | Induction | Deduction | Abduction | Calibrated | Useful steps | Pass@1 |
|---|---|---|---|---|---|---|
| R0 | 28 | 84 | 84 | 14% | ~11% | **80.5%** |
| R1 | 7 | 16 | 120 | 100% | ~60% | 61.0% |

R1 has 6× more useful training signal but drops pass@1 by 19.5 points.
Mechanism: 84% of the calibrated curriculum is abduction (short scalar
outputs), destroying the indentation prior that induction-style HumanEval
requires. Of 33 regressions R1→R0, 23 are Python indentation errors.

### 3.3 Inapplicability at scale

| Model | Prompt | Format | Pass@1 |
|---|---|---|---|
| Qwen3-8B | L2b | body-only | 63.4% |
| Qwen3-32B | generic | body-only | 6.1% |
| Qwen3-32B | generic | full-function | **89.0%** |
| Qwen3-32B | L2b | full-function | 87.8% |

Self-generation pipeline acceptance rates at 32B:

| Pool | Generator | Candidates | Accepted |
|---|---|---|---|
| Medium (verified) | Qwen3-8B | 236 | 2 (0.85%) |
| Hard (verified) | Qwen3-32B | 10 | 0 (0%) |

The learnability window is empty at 32B. A model that can formulate a
problem with a working reference solution is already, by that capability
alone, strong enough to solve it — so the pass rate on self-generated
candidates sits at 1.

## 4. Formatting vs. Capability

Per-problem analysis of 29 problems calibrated against the v2 checkpoint
(8 samples per problem, base vs. v2):

| Outcome | Problems | Fraction |
|---|---|---|
| v2 pass rate > base | 10 | 34% |
| v2 ≈ base | 18 | 62% |
| v2 pass rate < base | 1 | 3% |

Decomposition of the 10 v2-wins by mechanism:

| Mechanism | Problems |
|---|---|
| Formatting (correct logic, broken indentation in base) | 8 |
| Genuine logic improvement | 1 |
| Inconclusive on re-sampling | 1 |

Eight of the ten v2 wins are cases where the base model writes correct
code with broken indentation — the interpreter rejects it before checking
whether the logic works. v2 emits the same code with the leading
whitespace the function expects. One case is genuinely new behaviour:
on `is_valid_date`, the base model calls `re.fullmatch(format_str, ...)`
and the `%` characters in `"%Y-%m-%d"` make the match fail; v2 skips the
regex step and passes the format string to `strptime` directly.

## 5. Reproduction Commands

| Result | Command |
|---|---|
| Base Qwen3-8B | `python laimark/eval_adapter.py` (no adapter) |
| Self-generated, $G{=}4$ | `python laimark/train_grpo.py --selfgen_only --num_generations 4` |
| Curated GRPO | `python laimark/train_grpo.py --num_generations 4` |
| Iteration round 2b | `python laimark/train_grpo.py --selfgen_only --base_adapter grpo_output/final` |
| Task-type R1 | `python laimark/generate_deduction_abduction.py` → `calibrate_deduction_abduction.py` → `train_grpo.py` |
| Qwen3-32B base | `python laimark/eval_adapter.py --model Qwen3-32B --full_function` |

LoRA adapters and raw prediction logs are regenerated by the pipeline with
fixed random seeds; nothing is committed to the repository.
