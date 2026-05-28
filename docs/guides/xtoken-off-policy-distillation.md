# Cross-Tokenizer (X-Token) Off-Policy Distillation

NeMo RL supports off-policy distillation between a student and a teacher that
**do not share a tokenizer** — for example, distilling a Qwen3-4B teacher into
a Llama-3.2-1B student. Cross-tokenizer ("x-token") distillation handles the
vocabulary mismatch by routing student logits through a precomputed
**projection matrix** that maps each student token to the teacher tokens it
most plausibly corresponds to, projecting the student into the teacher's
vocab space so the two distributions can be compared.

This guide explains how to:

1. Produce the projection matrix from a (student, teacher) tokenizer pair
2. Launch a distillation run that consumes it

## How it works

A full run has two phases. The three prep steps are *offline data prep* —
small CLI tools you run once per (student, teacher) pair — and the result is a
single `.pt` file. The final step is the actual distillation training loop.

```
                        ┌──────────────────────────────────────────────┐
                        │  Offline projection-matrix preparation       │
                        │                                              │
                        │  ┌────────────────────────────────────┐      │
  (student, teacher)    │  │ 1. minimal_projection_via_         │      │
  tokenizers       ────▶│  │    multitoken.py                   │      │
                        │  │    — multi-token mappings          │      │
                        │  └─────────────────┬──────────────────┘      │
                        │                    │                         │
                        │  ┌─────────────────▼──────────────────┐      │
                        │  │ 2. (optional) reapply_exact_map.py │      │
                        │  │    — pin exact 1-to-1 matches      │      │
                        │  └─────────────────┬──────────────────┘      │
                        │                    │                         │
                        │  ┌─────────────────▼──────────────────┐      │
                        │  │ 3. sort_and_cut_projection_matrix  │      │
                        │  │    .py — trim to runtime top_k     │      │
                        │  └─────────────────┬──────────────────┘      │
                        └────────────────────│─────────────────────────┘
                                             │
                                             ▼  projection_matrix.pt
                        ┌──────────────────────────────────────────────┐
                        │  4. examples/                                │
                        │     run_xtoken_off_policy_distillation.py    │
                        │     — student forward + teacher forward      │
                        │       (via CUDA-IPC), x-token KD loss        │
                        └──────────────────────────────────────────────┘
```

The projection matrix is a sparse `[V_student, top_k]` tensor that the
training-time loss multiplies against the student logits to project them into
the teacher's vocab space.

### Which prep steps are essential?

Of the three prep steps, **Step 1 (multi-token mappings)** and
**Step 3 (sort and trim)** are required — Step 1 builds the cross-vocab
mapping itself, and Step 3 produces the runtime-format `.pt` the training
loss expects. **Step 2 (reapply exact map) is optional** and pins exact
1-to-1 token mappings on top of Step 1, but we found the best results
on this branch by running **Steps 1 → 2 → 3**.

## Quickstart — single command

For the typical case, `tools/x_token/build_projection_matrix.sh` chains
the prep steps with auto-derived intermediate paths:

```bash
./tools/x_token/build_projection_matrix.sh \
    --student-model meta-llama/Llama-3.2-1B \
    --teacher-model Qwen/Qwen3-4B \
    --runtime-top-k 4
```

The wrapper writes the final matrix to
`cross_tokenizer_data/projection_matrix_<student>_<teacher>_top<N>.pt`
(override with `--final-output`). Pass `--skip-exact-map` to skip the
optional Step 2, or `--no-{scale-trick,reverse-pass,special-token-mapping}`
to tweak Step 1 defaults. Run `./tools/x_token/build_projection_matrix.sh
--help` for the full flag list.

The per-step recipes below are for advanced customization (non-default
weight thresholds, hand-picked intermediate filenames, etc.).

## Backend and scope

- **DTensor V2 only.** Set `policy.dtensor_cfg.enabled=true` and
  `policy.dtensor_cfg._v2=true`. The Megatron policy worker is not wired
  for cross-tokenizer distillation.
- **Teacher logits travel via CUDA IPC**, so student and teacher policies must
  be colocated on the same node. No remote-Ray transport for x-token logits.

## Step 1 — Build multi-token mappings

Many student tokens (e.g., `"12"`) tokenize into multiple teacher tokens
(e.g., `"1"`, `"2"`). `minimal_projection_via_multitoken.py` walks the
student vocab, re-tokenizes each token with the teacher tokenizer, and adds
weighted entries to the projection. With `--enable-reverse-pass` it also
does the symmetric teacher → student walk.

```bash
uv run python -m tools.x_token.minimal_projection_via_multitoken \
    --student-model "meta-llama/Llama-3.2-1B" \
    --teacher-model "Qwen/Qwen3-4B" \
    --top-k 32 \
    --enable-scale-trick \
    --enable-reverse-pass \
    --enable-special-token-mapping
```

Output: `cross_tokenizer_data/projection_map_Llama-3.2_to_Qwen3_multitoken_top_32_double_special.pt`.

Pass `--num-examples 50` to print a sample of student→teacher mappings after
the matrix is built — useful for spot-checking that special tokens, numerals,
and punctuation map to sensible teacher tokens.

When `--enable-scale-trick` is set, the script records `enable_scale_trick=True`
in the saved `.pt` so Step 3 can auto-enable `--preserve_last`.

## Step 2 (optional) — Reapply exact-token map

Some token pairs are *literally identical* (e.g., common punctuation, single
ASCII characters). `reapply_exact_map.py` pins those to 1-to-1 mappings with
weight 1.0, overwriting whatever Step 1 produced for them.

```bash
uv run python -m tools.x_token.reapply_exact_map \
    --student-model "meta-llama/Llama-3.2-1B" \
    --teacher-model "Qwen/Qwen3-4B" \
    --initial-projection-path cross_tokenizer_data/projection_map_Llama-3.2_to_Qwen3_multitoken_top_32_double_special.pt
```

Output is written next to the input as `<basename>_exact_map_remapped.pt`.

## Step 3 — Sort and trim to runtime `top_k`

The training loss only needs a small `top_k` per row (typical: 4–8). This
step sorts each row by weight and trims to the chosen runtime cap.

```bash
uv run python -m tools.x_token.sort_and_cut_projection_matrix \
    --initial-projection-path cross_tokenizer_data/projection_map_Llama-3.2_to_Qwen3_multitoken_top_32_double_special_exact_map_remapped.pt \
    --top_k 4 \
    --output_path cross_tokenizer_data/projection_matrix_llama_qwen_top4.pt
```

`--preserve_last` is `argparse.BooleanOptionalAction` with default `None`. When
unspecified, the script reads `enable_scale_trick` from the input matrix's
metadata (set in Step 1) and auto-enables preservation of the last column
slot. Pass `--preserve_last` or `--no-preserve_last` to override.

## Step 4 — Launch x-token distillation

The training entrypoint is `examples/run_xtoken_off_policy_distillation.py` with the
exemplar config at `examples/configs/xtoken_off_policy_distillation.yaml`. The exemplar
defaults to Llama-3.2-1B (student) ← Qwen3-4B (teacher), an arrow-text
corpus, and the P-KL loss mode. Override paths via Hydra CLI:

```bash
uv run python examples/run_xtoken_off_policy_distillation.py \
    --config examples/configs/xtoken_off_policy_distillation.yaml \
    loss_fn.projection_matrix_path=cross_tokenizer_data/projection_matrix_llama_qwen_top4.pt \
    data.train.data_files=/path/to/corpus/*.arrow \
    cluster.gpus_per_node=8 \
    cluster.num_nodes=1
```

The exemplar config keeps `loss_fn.projection_matrix_path` and
`data.train.data_files` as `null` so they must be supplied at the CLI — this
makes the config reusable across (student, teacher) pairs.

### Loss-mode knobs

`loss_fn` has two flags that pick between three behaviors:

| `gold_loss` | `xtoken_loss` | Behavior |
|---|---|---|
| `false` | (inert) | **P-KL** — full-vocab teacher logits via CUDA IPC; the loss derives a microbatch-global top-k inside, projects the student into teacher vocab via the projection matrix, and chunk-averages KL on the top-k subset. CE term is added. |
| `true` | `false` | **Gold loss** — split the vocab into an *exact-token-mapped* common set (KL) and an *uncommon* tail (sorted L1). |
| `true` | `true`  | **Gold + x-token loss** — same as gold, but relax the exact-map threshold to `>= 0.6` and allow multi-token projections to count as exact maps via a collision-replacement rule. |

Other relevant fields:

- `loss_fn.temperature` — softmax temperature applied symmetrically to student and teacher logits before KL.
- `loss_fn.vocab_topk` — microbatch-global top-k size for the P-KL path (inert when `gold_loss=true`).
- `loss_fn.uncommon_topk` — cap on the L1 uncommon-tail sort in the gold path (defaults to 8192).
- `loss_fn.reverse_kl` — compute `KL(student || teacher)` instead of `KL(teacher || student)`.

## Where files live

| Stage | Tool | Default output |
|---|---|---|
| Build multi-token | `tools/x_token/minimal_projection_via_multitoken.py` | `<output_dir>/projection_map_<student>_to_<teacher>_multitoken_top_<N>_double[_special].pt` |
| Reapply exact map | `tools/x_token/reapply_exact_map.py` | `<input>_exact_map_remapped.pt` |
| Sort and trim | `tools/x_token/sort_and_cut_projection_matrix.py` | `<input_dir>/<basename>_top_<N>_sorted[_preservelast].pt` (or `--output_path`) |
| Train | `examples/run_xtoken_off_policy_distillation.py` | per the run's `logger.log_dir` and `checkpointing.checkpoint_dir` |

## Related

- Config exemplar: [`examples/configs/xtoken_off_policy_distillation.yaml`](../../examples/configs/xtoken_off_policy_distillation.yaml)
- Trainer module: `nemo_rl/algorithms/xtoken_off_policy_distillation.py`
- Loss implementation: `nemo_rl/algorithms/loss/loss_functions.py::CrossTokenizerDistillationLossFn`
- Token alignment: `nemo_rl/algorithms/x_token/token_aligner.py::TokenAligner`
- Cross-tokenizer collator: `nemo_rl/data/cross_tokenizer_collate.py::CrossTokenizerCollator`
- KD data processor: `nemo_rl/data/processors.py::kd_data_processor`
