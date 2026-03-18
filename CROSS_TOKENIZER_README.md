# Cross-Tokenizer Off-Policy Distillation for NeMo RL

This document describes the cross-tokenizer off-policy distillation feature built
on top of NeMo RL. It enables knowledge distillation between teacher and student
models that use **different tokenizers and vocabularies** (e.g., Qwen 8B teacher
distilling into a Llama 1B student).

---

## Table of Contents

1. [Base Commit](#base-commit)
2. [Overview](#overview)
3. [Architecture](#architecture)
4. [Commit History](#commit-history)
5. [New Files](#new-files)
6. [Modified Existing Files](#modified-existing-files)
7. [How to Run](#how-to-run)
8. [Configuration Reference](#configuration-reference)
9. [Design Decisions](#design-decisions)

---

## Base Commit

All changes are built on top of the **NeMo RL `v0.5.0`** release.

| Field          | Value                                                                      |
|----------------|----------------------------------------------------------------------------|
| Repository     | [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) (the `origin` remote) |
| Tag            | `v0.5.0`                                                                   |
| Commit hash    | `6c7089300fded94abfa49bb9cbf9eb357d862461`                                 |
| Commit message | `cp: Bump protobuf to 6.33.5 and python-multipart to 0.0.22 into r0.5.0 (#1851)` |
| Branch         | `xtoken/off-policy-distillation`                                           |

To verify locally:

```bash
git log --oneline 6c708930 -1
# Expected: 6c708930 cp: Bump protobuf to 6.33.5 and python-multipart to 0.0.22 into `r0.5.0` (#1851)

git tag --contains 6c708930
# Expected: v0.5.0
```

---

## Overview

Standard NeMo RL distillation assumes the teacher and student share the same
tokenizer. This fork removes that constraint by adding two major capabilities:

1. **Off-policy distillation** -- A training loop that uses a fixed dataset of
   text (Arrow files) instead of generating responses on-policy. The teacher
   produces logits for the fixed responses and the student aligns to them via KL
   divergence. This is simpler and cheaper than on-policy distillation because
   there is no student generation step or environment needed.

2. **Cross-tokenizer support (TokenAligner)** -- When the teacher and student
   use different tokenizers (e.g., Qwen's 151K-token vocabulary vs. Llama's
   128K-token vocabulary), a precomputed *projection matrix* maps student
   probabilities into the teacher's vocabulary space. A dynamic-programming
   *token alignment* algorithm aligns the two tokenizations of each text at
   the sequence level so the KL loss is computed at comparable positions.

Together, these enable distillation from any teacher to any student regardless
of their tokenizer, which was not previously possible in NeMo RL.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Training Step Overview                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. Load batch of text from Arrow dataset                               │
│     ├── Tokenize with student tokenizer → student_input_ids             │
│     └── Tokenize with teacher tokenizer → teacher_input_ids             │
│                                                                         │
│  2. Token alignment (TokenAligner.align)                                │
│     └── DP alignment of student & teacher token sequences               │
│         → aligned_pairs: list of (s_start, s_end, t_start, t_end)       │
│                                                                         │
│  3. Teacher forward pass (via IPC)                                      │
│     ├── Teacher model produces full-vocab logits                        │
│     ├── Log-softmax computed distributedly (TP-aware)                   │
│     └── Stored in GPU IPC buffers (no Ray data transfer)                │
│                                                                         │
│  4. Student forward pass + loss                                         │
│     ├── Student model produces logits                                   │
│     ├── Reads teacher logits from IPC buffers                           │
│     ├── CrossTokenizerDistillationLossFn computes:                      │
│     │   ├── Project student probs → teacher vocab via projection matrix │
│     │   ├── Chunk-average over aligned spans                            │
│     │   └── KL divergence per chunk, masked_mean reduction              │
│     └── Backprop through student only                                   │
│                                                                         │
│  5. (Optional) Periodic MATH/MMLU evaluation via colocated vLLM         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key components

- **TokenAligner** (`nemo_rl/algorithms/x_token/tokenalign.py`): Core module
  that handles vocabulary projection and sequence-level token alignment.
  - *Projection matrix*: A precomputed mapping from student vocabulary to
    teacher vocabulary. Each student token maps to a weighted set of teacher
    tokens (e.g., top-32 most likely correspondences).
  - *DP alignment*: Dynamic-programming algorithm that aligns student and
    teacher token sequences allowing 1:1, 1:N, N:1, and N:M mappings.
    Uses anchor-based optimization (unique n-gram matches) to avoid
    quadratic cost on long sequences.
  - *Token canonicalization*: Normalizes tokenizer-specific representations
    (SentencePiece byte tokens, space prefixes, Unicode artifacts) so the
    DP alignment can match across tokenizer families.

- **Projection matrix generators** (`nemo_rl/algorithms/x_token/`):
  Standalone scripts to precompute the projection matrix offline.
  - `minimal_projection_via_multitoken.py` -- Multi-token analysis: tokenize
    each student token string with the teacher tokenizer and distribute
    probability mass across the resulting sub-tokens.
  - `minimal_projection_generator.py` -- Embedding similarity: use LLM
    embedding layers to compute cosine similarity between vocabularies.
  - `reapply_exact_map.py` -- Post-processing: force exact 1:1 mappings for
    tokens that are identical across both tokenizers.

- **IPC teacher logits**: Teacher logits are passed between the teacher and
  student forward passes using CUDA IPC handles instead of serializing through
  Ray. This avoids expensive CPU-GPU-CPU round-trips for large logit tensors.

---

## Commit History

Eight commits on top of `v0.5.0`, listed oldest to newest:

| Hash       | Message                                                             |
|------------|---------------------------------------------------------------------|
| `0658b8d2` | Add off-policy distillation with MATH/MMLU eval and IPC optimization |
| `668c37ed` | Commit before refactoring                                           |
| `13066d63` | Simplify off-policy distillation IPC path and config                |
| `f733e57c` | Working IPC TP=1                                                    |
| `3204ac78` | Per-microbatch IPC teacher logits with TP=4 support                 |
| `d4de1d8f` | Clean up unused scripts and old distillation module                 |
| `f9fe64a5` | Add IPC/non-IPC toggle for off-policy distillation                  |
| `58a1bd71` | Integrate cross-tokenizer distillation (TokenAligner) into NeMo RL  |

---

## New Files

### Core algorithm

| File | Lines | Purpose |
|------|-------|---------|
| `nemo_rl/algorithms/off_policy_distillation.py` | ~1,100 | Off-policy distillation training loop. Contains `off_policy_distillation_train()` which iterates over a fixed dataset, runs teacher inference, and trains the student with KL loss. Handles checkpointing, validation, eval hooks, and cross-tokenizer data preparation. Created because the existing `distillation.py` is on-policy (generates student responses via rollout), which is unnecessary and expensive when training on a fixed text corpus. |

### Cross-tokenizer module (`nemo_rl/algorithms/x_token/`)

| File | Lines | Purpose |
|------|-------|---------|
| `tokenalign.py` | ~4,300 | Core `TokenAligner` class (`nn.Module`). Handles projection matrix loading/management, DP-based sequence alignment, token canonicalization, and multiple KL loss computation strategies (standard, optimized with vocab top-k, gold loss with common/uncommon vocab split). This is the central piece that makes cross-tokenizer distillation possible. |
| `minimal_projection_via_multitoken.py` | ~930 | Generates projection matrices via multi-token analysis. For each student token, tokenizes its string with the teacher tokenizer and distributes weight across the resulting sub-tokens with exponential decay. Preferred method for generating projection matrices. |
| `minimal_projection_generator.py` | ~570 | Generates projection matrices via embedding cosine similarity. Uses LLM first-layer embeddings or sentence transformers. Alternative to the multi-token method. |
| `reapply_exact_map.py` | ~230 | Post-processes a projection matrix to enforce perfect 1:1 mappings for tokens that are identical across both tokenizers (e.g., punctuation, digits). |
| `sort_and_cut_projection_matrix.py` | ~440 | Utility to sort projection matrix rows by weight and apply a top-k cutoff. Includes optional Sinkhorn renormalization. |
| `__init__.py` | 3 | Exports `TokenAligner`. |

### Training entry points and configs

| File | Purpose |
|------|---------|
| `examples/run_off_policy_distillation_arrow_with_eval.py` | Main training script. Extends `off_policy_distillation_train()` with periodic generation-based evaluation on MATH and MMLU using colocated vLLM. Handles cross-tokenizer setup when `token_aligner.enabled: true`. |
| `examples/configs/cross_tokenizer_off_policy_arrow.yaml` | Reference YAML config for Llama-3.2-1B (student) with Qwen3-8B-Base (teacher). Includes all token_aligner, loss_fn, eval, and cluster settings. |
| `submit_cross_tokenizer.sh` | SLURM submission script for the cross-tokenizer experiment. Supports chained job dependencies (`-n N` for sequential restarts). |

### Dataset support

| File | Purpose |
|------|---------|
| `nemo_rl/data/datasets/response_datasets/arrow_text_dataset.py` | `ArrowTextDataset` class for loading Arrow files with a `text` column. Wraps each text as an assistant message for SFT-style training. Created because the existing dataset classes expected prompt-response pairs or specific dataset formats, not raw text Arrow files. |

### Other new files

| File | Purpose |
|------|---------|
| `examples/run_off_policy_distillation_arrow.py` | Simpler off-policy script without evaluation (same-tokenizer only). |
| `examples/run_sft_arrow_with_eval.py` | SFT training on Arrow data with MATH/MMLU eval. Used as reference for the eval integration pattern. |
| `examples/configs/llama_off_policy_arrow.yaml` | Config for same-tokenizer off-policy distillation (Llama teacher + Llama student). |

---

## Modified Existing Files

### `nemo_rl/models/policy/lm_policy.py`

**What changed:** Added `teacher_forward()` method, extended `train()` with `is_teacher`, `teacher_logits`, and `topk_logits` parameters, and replaced hard `config["key"]` accesses with safer `config.get("key", {}).get(...)` patterns.

**Why:** The existing `Policy` class had no concept of a teacher-only forward pass. For IPC-based distillation, the teacher needs to run a forward pass that stores logits in GPU IPC buffers without returning data through Ray. The `train()` method was extended so the same worker infrastructure can handle both teacher inference and student training in a single call. The safer config accesses were needed because the teacher policy config omits optional keys like `dynamic_batching` and `sequence_packing` that the original code assumed were always present.

Key additions:
- `teacher_forward()` dispatches a teacher-only forward to workers, storing results in IPC buffers.
- `train()` gains `is_teacher=True` mode: skips optimizer, returns IPC handles instead of loss.
- `train()` gains `teacher_logits` parameter: when provided, each worker reads teacher logits from IPC handles for its microbatch.
- Config accesses like `config["dynamic_batching"]["enabled"]` changed to `config.get("dynamic_batching", {}).get("enabled", False)` to avoid `KeyError` when these sections are absent.

### `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py`

**What changed:** Extended `train()` to handle teacher inference mode (IPC buffer allocation, top-k logit extraction, distributed log-softmax) and student training mode (reading teacher logits from IPC handles). Disabled temperature scaling during training. Added debug logging for NaN detection.

**Why:** The DTensor worker is where the actual model forward pass happens. To support IPC-based distillation:
- In teacher mode (`is_teacher=True`), the worker runs the model, computes distributed log-softmax across TP ranks, optionally extracts top-k logits, and stores results in pre-allocated CUDA IPC buffers. This avoids serializing large logit tensors through Ray.
- In student mode, the worker reads teacher logits from IPC handles (using `rebuild_cuda_tensor_from_ipc`) and passes them to the loss function alongside the student's own logits.
- Temperature scaling was being applied during training, which distorts the KL divergence computation. It is now skipped (`skip=True`) during training and only applied during generation/inference.

### `nemo_rl/algorithms/loss_functions.py`

**What changed:** Extended `DistillationLossFn.__call__()` with three code paths (IPC top-k, IPC full-logprob, standard data-dict), and added the entirely new `CrossTokenizerDistillationLossFn` class (~560 lines).

**Why:**
- The original `DistillationLossFn` only supported a single path where teacher top-k logits were pre-computed and passed in the data dict. The IPC paths were added to avoid materializing teacher logits on CPU (they stay on GPU in IPC buffers).
- `CrossTokenizerDistillationLossFn` is new and handles the case where teacher and student have different vocabularies. It uses the TokenAligner's projection matrix to map student probabilities into the teacher's vocabulary space, then computes chunk-averaged KL divergence over aligned token spans. It also supports a *gold loss* variant that splits the vocabulary into common tokens (direct KL) and uncommon tokens (sorted L1 / Universal Likelihood Distillation).

### `nemo_rl/data/datasets/response_datasets/__init__.py`

**What changed:** Registered `ArrowTextDataset` in the dataset factory so `dataset_name: "arrow_text"` works in config files.

**Why:** The existing factory had no support for raw text Arrow files. Adding the registration allows the off-policy training scripts to load Arrow datasets through the standard NeMo RL data pipeline.

### `nemo_rl/algorithms/distillation.py`

**What changed:** Minor compatibility adjustments.

**Why:** Small fixes to ensure the existing on-policy distillation module works alongside the new off-policy code without import conflicts.

---

## How to Run

### Prerequisites

- NeMo RL environment (container or `uv` virtual env) based on `v0.5.0`
- Access to teacher and student model weights on HuggingFace (e.g., `Qwen/Qwen3-8B-Base`, `meta-llama/Llama-3.2-1B`)
- Training data as Arrow files with a `text` column
- SLURM cluster with GPU nodes

### Step 1: Generate the projection matrix

The projection matrix maps student vocabulary to teacher vocabulary. Generate it
once offline:

```bash
# Multi-token method (recommended)
python nemo_rl/algorithms/x_token/minimal_projection_via_multitoken.py \
    --student-model meta-llama/Llama-3.2-1B \
    --teacher-model Qwen/Qwen3-8B-Base

# Optionally enforce exact matches for identical tokens
python nemo_rl/algorithms/x_token/reapply_exact_map.py \
    --student-model meta-llama/Llama-3.2-1B \
    --teacher-model Qwen/Qwen3-8B-Base \
    --initial-projection-path cross_tokenizer_data/transformation_counts_via_multitoken.pt

# Optionally sort and cut to top-k per row
python nemo_rl/algorithms/x_token/sort_and_cut_projection_matrix.py \
    --input cross_tokenizer_data/transformation_counts_via_multitoken.pt \
    --top-k 32
```

The output is a `.pt` file containing `{indices, likelihoods}` tensors.

### Step 2: Configure the YAML

Edit `examples/configs/cross_tokenizer_off_policy_arrow.yaml` or create a new
config. The key sections are:

```yaml
token_aligner:
  enabled: true
  projection_matrix_path: "path/to/projection_map.pt"
  use_sparse_format: false
  loss_type: "KL"
  vocab_topk: 8192          # Reduce teacher vocab to top-8192 for speed
  max_comb_len: 4            # Max tokens in a single DP alignment chunk

policy:
  model_name: "meta-llama/Llama-3.2-1B"   # Student
  # ... optimizer, scheduler, dtensor config ...

teacher:
  model_name: "Qwen/Qwen3-8B-Base"        # Teacher
  # ... dtensor config (no optimizer needed) ...

loss_fn:
  loss_type: "KL"
  gold_loss: true            # Common-vocab KL + uncommon-vocab L1
  xtoken_loss: true          # Relaxed exact-map threshold (>=0.6)
  ce_loss_scale: 0.1         # Optional next-token CE loss
  dynamic_loss_scaling: true

data:
  dataset_name: "arrow_text"
  arrow_files: "/path/to/data/*.arrow"
  max_input_seq_length: 4096

distillation:
  use_ipc: true              # Required for cross-tokenizer
  topk_logits_k: 8192
  num_prompts_per_step: 768
  max_num_steps: 80000
```

### Step 3: Submit the job

```bash
# Single run
bash submit_cross_tokenizer.sh

# Chain 5 sequential jobs (each picks up from the last checkpoint)
bash submit_cross_tokenizer.sh -n 5
```

The script submits a SLURM job that:
1. Starts a Ray cluster across all allocated nodes
2. Runs `examples/run_off_policy_distillation_arrow_with_eval.py`
3. Periodically evaluates on MATH and MMLU via colocated vLLM
4. Logs to Weights & Biases

### Same-tokenizer mode

If the teacher and student share the same tokenizer, set `token_aligner.enabled: false`
(or omit the `token_aligner` section). The training loop falls back to the
standard `DistillationLossFn` with top-k teacher logits.

---

## Configuration Reference

### `token_aligner` section

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `false` | Master switch for cross-tokenizer mode |
| `projection_matrix_path` | str | required | Path to `.pt` projection matrix file |
| `use_sparse_format` | bool | `true` | Load projection as sparse COO (faster for large vocabs) |
| `loss_type` | str | `"KL"` | `"KL"`, `"cross_entropy"`, or `"chunked_ce"` |
| `exact_token_match_only` | bool | `false` | Only use 1:1 aligned positions for loss |
| `temperature` | float | `1.0` | Softmax temperature for KL computation |
| `vocab_topk` | int | `8192` | Reduce teacher vocab to top-k (0 = use all) |
| `reverse_kl` | bool | `false` | Use reverse KL direction |
| `projection_matrix_multiplier` | float | `1.0` | Scaling factor for projection matrix |
| `max_comb_len` | int | `4` | Max combination length for DP alignment |
| `learnable` | bool | `false` | Make projection matrix trainable |
| `project_teacher_to_student` | bool | `false` | Project teacher to student vocab instead |

### `loss_fn` section (cross-tokenizer)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `loss_type` | str | `"KL"` | Loss type |
| `temperature` | float | `1.0` | Softmax temperature |
| `vocab_topk` | int | `8192` | Teacher vocab top-k filtering |
| `exact_token_match_only` | bool | `false` | Restrict to 1:1 aligned positions |
| `reverse_kl` | bool | `false` | Reverse KL direction |
| `gold_loss` | bool | `false` | Common-vocab KL + uncommon-vocab sorted L1 |
| `xtoken_loss` | bool | `false` | Relaxed exact-map threshold (>=0.6 instead of ==1.0) |
| `ce_loss_scale` | float | `0.0` | Weight for auxiliary next-token CE loss (0 = disabled) |
| `dynamic_loss_scaling` | bool | `false` | Scale KL loss to match CE loss magnitude |

---

## Design Decisions

### Why off-policy distillation?

On-policy distillation (the existing `distillation.py`) generates student
responses, scores them with an environment, and uses the teacher to compute
logits on those responses. This requires a generation engine, an environment,
and produces different data each step. Off-policy distillation uses a *fixed*
text corpus -- the same data the teacher was trained on. This is:
- **Simpler**: No generation step, no environment, no rollout turns.
- **Cheaper**: No vLLM inference for student generation during training.
- **Deterministic**: Same data every epoch, easier to debug and reproduce.
- **Sufficient for distillation**: When the goal is to transfer the teacher's
  language modeling ability (not RL-specific behavior), a fixed corpus works well.

### Why IPC for teacher logits?

Teacher logits for a batch of shape `[B, S, V]` (e.g., `[768, 4096, 151936]`
for Qwen 8B) are hundreds of gigabytes. Passing them through Ray would require
serializing to CPU, transferring, and deserializing back to GPU. CUDA IPC handles
allow the student worker to read the teacher's GPU memory directly without any
data movement. This is why `distillation.use_ipc: true` is required for
cross-tokenizer mode.

### Why chunk-averaged KL?

When teacher and student tokenize the same text differently, there is no 1:1
correspondence between all token positions. For example, the word "unhappiness"
might be `["un", "happiness"]` in one tokenizer and `["un", "happ", "iness"]`
in another. The DP alignment finds these correspondences and groups them into
*chunks*. Within each chunk, the teacher and student distributions are averaged
over their respective token spans, renormalized, and compared via KL divergence.
This handles 1:1, 1:N, N:1, and N:M alignments uniformly.

### Why gold loss?

The standard projection-based path projects student probabilities into the
teacher's vocabulary space using a precomputed matrix. This introduces
approximation error for tokens that don't have clean 1:1 mappings. The *gold
loss* variant avoids projection entirely for tokens that have exact matches
between vocabularies (e.g., digits, punctuation, common words). For these
"common" tokens, KL is computed directly on the native log-probabilities. For
"uncommon" tokens (no exact mapping), it falls back to sorted L1 on probability
vectors (Universal Likelihood Distillation). This typically gives better gradient
signal for the majority of tokens.

### Why per-microbatch IPC?

Large batches are split into microbatches for gradient accumulation. Rather than
storing the entire batch of teacher logits in GPU memory (which may not fit),
each microbatch's teacher logits are stored in a separate IPC buffer. The
student reads only the current microbatch's buffer during its forward pass.
This keeps peak GPU memory proportional to the microbatch size, not the global
batch size. The implementation also supports tensor parallelism (TP > 1) where
each rank stores and reads only its local vocabulary shard.
