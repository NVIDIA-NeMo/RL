# Cross-tokenizer distillation

NeMo RL offers two algorithms for off-policy distillation when a student and
teacher use different tokenizers. Both reuse the cross-tokenizer collator,
token alignment, colocated teacher execution, and student training loop.
Only the student receives gradient updates.

| Algorithm | Vocabulary preparation | Training entrypoint | Exemplar configuration |
|---|---|---|---|
| **X-Token** | Weighted student-to-teacher projection matrix | `examples/run_xtoken_off_policy_distillation.py` | `examples/configs/xtoken_off_policy_distillation.yaml` |
| **CSD** | Student-to-teacher and teacher-to-student sub-token tables | `examples/run_csd.py` | `examples/configs/csd.yaml` |

The top-level `algorithm` field selects `xtoken` or `csd`. Each entrypoint
uses its corresponding exemplar. The
[X-Token guide](xtoken-off-policy-distillation.md) retains the complete
projection-matrix preparation and training workflow, including P-KL, Gold,
and H-KL loss settings.

CSD compares probability on events shared by the aligned tokenizations.
Single-token common events and prefix-support alternatives come from the
sub-token tables; a REST category retains probability outside the selected
events. CSD supports dense teacher logits and sparse top-k teacher transport,
with exact vocabulary normalization and retained realized labels. Student
next-token cross-entropy can be combined with the distillation objective.

**CSD sub-token tables are not X-Token projection matrices.** A table stores
complete sequences of destination token IDs, without projection weights.
Prepare both directions before launching CSD. The following example uses
`meta-llama/Llama-3.2-1B` as the student and `Qwen/Qwen3-4B` as the teacher.

## Prerequisites

- Run commands from the repository root in the environment described in
  [installation](../about/installation.md) or the [container guide](../docker.md).
  Initialize the checkout's dependencies with
  `git submodule update --init --recursive` and use the checked-in `uv.lock`.
- Table preparation uses PyTorch, Transformers, and Tokenizers on CPU; it
  loads fast tokenizers without downloading model weights or requiring GPUs.
  These libraries are included in the project's Python dependencies.
- Obtain access to the selected models and accept any applicable model
  terms, including the gated Llama model. Make Hugging Face authentication
  and the model cache available to the preparation process and training workers.
- Training requires a compatible Linux/CUDA environment, the model backend's
  dependencies, and enough GPU and host memory for the selected models.
  The example below uses one node with eight GPUs and the Megatron backend.
  The teacher and student time-share a colocated GPU pool; CUDA IPC requires
  the producing and consuming ranks to be on the same node.

Prepare the example's environment with:

```bash
git submodule update --init --recursive
uv sync --locked --extra mcore
```

Use the same tokenizer assets during preparation and training. Model IDs may
also be local snapshots. For reproducible runs, use pinned snapshots for both
the model and its tokenizer, or supply immutable revisions consistently.
No rollout service or cached teacher-logit dataset is required.

## Stage A: build and validate the sub-token tables

Build the forward table. Its rows are student token IDs and its chains contain
teacher token IDs:

```bash
uv run python -m tools.csd.build_subtoks_table \
  --source-tokenizer meta-llama/Llama-3.2-1B \
  --destination-tokenizer Qwen/Qwen3-4B \
  --max-subtokens 8 \
  --output cross_tokenizer_data/csd/student_to_teacher.pt
```

Reverse the tokenizer order to build the reverse table:

```bash
uv run python -m tools.csd.build_subtoks_table \
  --source-tokenizer Qwen/Qwen3-4B \
  --destination-tokenizer meta-llama/Llama-3.2-1B \
  --max-subtokens 8 \
  --output cross_tokenizer_data/csd/teacher_to_student.pt
```

The builder creates the output directory. Use `--source-revision` and
`--destination-revision` to pin remote tokenizer revisions, and `--overwrite`
to explicitly replace an existing table. `--max-subtokens` must be positive;
complete chains exceeding that limit remain unmapped. Raising it retains more
long decompositions and increases table size. The builder never truncates a
chain into a different token event.

Each `.pt` file contains:

| Key | Meaning |
|---|---|
| `subtoks` | Integer tensor `[source_vocab_size, max_subtokens]` containing destination token IDs; unused entries are `-1` |
| `lengths` | Integer vector of complete chain lengths; `0` means the source token is unmapped |
| `V_src`, `V_dst`, `N_max` | Vocabulary sizes, including added tokens, and the chain-length limit |
| `src`, `dst` | Source and destination tokenizer identifiers |
| `source_fingerprint`, `destination_fingerprint` | Identity checks for the tokenizer assets and special-token roles |

Exact common tokens produce length-one chains. The builder maps unambiguous
special-token roles such as EOS to EOS even when their strings differ;
unmatched specials remain unmapped instead of becoming ordinary literal text.
For byte-level BPE pairs it preserves incomplete UTF-8 fragments as bytes.
The builder and validator support fast tokenizers with a BPE model and
ByteLevel decoder, including the example model pair. Metaspace/SentencePiece,
WordPiece, and other decoder families are unsupported because isolated decoding
can alter whitespace or word markers with context. Inspect the reported mapped/unmapped counts
when changing tokenizer families or the chain limit.

Validate both saved tables against the tokenizers before allocating GPUs:

```bash
uv run python -m tools.csd.validate_subtoks_tables \
  --student-tokenizer meta-llama/Llama-3.2-1B \
  --teacher-tokenizer Qwen/Qwen3-4B \
  --student-to-teacher cross_tokenizer_data/csd/student_to_teacher.pt \
  --teacher-to-student cross_tokenizer_data/csd/teacher_to_student.pt
```

When revisions were supplied during construction, also pass the corresponding
`--student-revision` and `--teacher-revision` here. Validation checks tokenizer
fingerprints, tensor shapes and dtypes, destination ID ranges, padding,
special-token roles, and every retained ordinary token event. It exits with
an error on an inconsistent table. Regenerate both directions when either
tokenizer changes. The files must remain available at the configured paths
on every training worker, including inside a container.

## Stage B: configure and launch CSD training

The complete configuration is
[`examples/configs/csd.yaml`](../../examples/configs/csd.yaml). Its student,
teacher, and raw-text data interface match the example below. The relevant table
settings are:

```yaml
algorithm: csd
loss_fn:
  common_indices_from_subtoks: true
teachers:
  - model_name: Qwen/Qwen3-4B
    tokenizer:
      name: Qwen/Qwen3-4B
    aligner:
      projection_matrix_path: null
      student_to_teacher_table_path: cross_tokenizer_data/csd/student_to_teacher.pt
      teacher_to_student_table_path: cross_tokenizer_data/csd/teacher_to_student.pt
```

This is an excerpt of the exemplar, not a replacement for its model and
optimizer settings. The forward table supplies the common token map when
`loss_fn.common_indices_from_subtoks=true`; both directions supply the
prefix-support alternatives. The tables are loaded by the CSD loss for each
cross-tokenizer teacher. Add a separately prepared pair for each additional
cross-tokenizer teacher. The example does not need a projection matrix.

Useful CSD controls in the `loss_fn` block include:

| Field | Purpose |
|---|---|
| `csd_loss_fn` | Common-event divergence: `kl`, `jsd`, or `bce` |
| `csd_last_pos_loss_fn` | Last-position mismatch divergence; `null` inherits `csd_loss_fn` |
| `csd_jsd_beta` | JSD mixture coefficient; `0.5` is symmetric |
| `csd_mismatch_loss_beta` | Multiplier on each 1-to-many, many-to-1, and many-to-many chunk loss relative to a 1-to-1 chunk loss |
| `csd_realized_only_mismatch` | Score 1-to-many and many-to-1 chunks by their realized token chains against REST instead of prefix-support alternatives |
| `teacher_topk_ipc_k` | Teacher logits sent per row: the top `k` (default `128`) with exact normalization, or `0` for dense full-vocabulary logits |
| `dynamic_loss_scaling` | Rescale KD using CE magnitude; when `false`, use the explicit `kl_loss_weight` and `ce_loss_scale` coefficients |

The exemplar uses sparse top-128 transport and dynamic loss scaling. These are
training settings: they do not change how the static tokenizer tables are
constructed.

The default dataset is the raw `text` column of
`nvidia/Nemotron-Pretraining-Specialized-v1.1`, subset
`Nemotron-Pretraining-Formal-Logic`, loaded by dataset identifier so it
resolves from the Hugging Face cache.
To use your own corpus, set `data.train.data_files` to a file, glob, or dataset
identifier and `data.train.text_key` to its text column; for a file or glob,
also set `data.train.subset=null`. Supported local file
formats include `.arrow`, `.parquet`, `.json`, `.jsonl`, and `.txt`. The
`arrow_text` dataset and `kd_data_processor` leave tokenization to the shared
collator. `collator.mode=text` tokenizes each text with both tokenizers;
use the existing [chat-data guidance](xtoken-off-policy-distillation.md#text-and-chat-batches)
for conversational examples. CSD chat mode supervises only the final assistant
turn; earlier messages are context. In ordinary CSD chat mode, CE covers the
whole final assistant turn, including its template markers, and KD uses its
content and EOT. Native thinking alignment reuses the shared region-based
supervision described in that guide.

A text row that does not fit every model's sequence limit after tokenization
is clipped to a prefix of its text that fits all of them, so the student and
every teacher still cover the same text; the collator never truncates one
side alone. Chat rows are not clipped: an overlength chat row is rejected.
Keep `data.train.characters_per_sample=null` and `collator.num_packed_rows=1`.

For a self-contained smoke run, first create 128 short arithmetic texts. This
small fixture exercises training and is not a model-quality benchmark:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

path = Path("cross_tokenizer_data/csd/smoke_train.jsonl")
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("w") as stream:
    for number in range(128):
        text = f"The sum of {number} and 1 is {number + 1}."
        stream.write(json.dumps({"text": text}) + "\n")
PY
```

Launch a two-step smoke run from the repository root on the allocated node:

```bash
uv run python examples/run_csd.py \
  --config examples/configs/csd.yaml \
  teachers.0.aligner.student_to_teacher_table_path=cross_tokenizer_data/csd/student_to_teacher.pt \
  teachers.0.aligner.teacher_to_student_table_path=cross_tokenizer_data/csd/teacher_to_student.pt \
  data.train.data_files=cross_tokenizer_data/csd/smoke_train.jsonl \
  data.train.subset=null \
  data.train.characters_per_sample=null \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=8 \
  distillation.max_num_steps=2 \
  checkpointing.enabled=true \
  checkpointing.save_period=1 \
  checkpointing.checkpoint_dir=checkpoints/csd-smoke \
  logger.log_dir=logs/csd-smoke \
  logger.wandb_enabled=false
```

The provided model and tokenizer names must stay paired with the tables. To
change the student, override both `policy.model_name` and
`policy.tokenizer.name`; to change a teacher, override both
`teachers.0.model_name` and `teachers.0.tokenizer.name`, and rebuild its tables.
For pinned remote tokenizers, set the corresponding
`tokenizer.tokenizer_kwargs.revision` fields to the revisions used above.
Keep model revisions or local snapshots consistent with those tokenizers.

Inspect the resolved configuration and training logs for the intended
`algorithm`, table paths, finite CE/KD losses, and completed optimizer steps.
The checkpoint and log directories are set explicitly in the command.
Checkpoints are saved in Megatron format; convert one to Hugging Face format
with the Megatron command in [Evaluation](../about/evaluation.md).
Increase `distillation.max_num_steps` and choose a new checkpoint directory
for the actual experiment. Table validation establishes preparation correctness;
it does not replace a GPU smoke run for the selected backend, model pair,
and parallelism configuration.

## Teacher transport and context parallelism

Each cross-tokenizer teacher sends its logits to the student through CUDA IPC,
in one of two forms:

- **Sparse** (`loss_fn.teacher_topk_ipc_k > 0`, default `128`): for each teacher
  row, the top-k logits, the exact log-normalizer, and the realized teacher
  labels the loss needs. The KD then compares the top-k tokens plus a REST
  bucket that holds the remaining probability mass.
- **Dense** (`teacher_topk_ipc_k: 0`): the full-vocabulary logits.

Same-tokenizer teachers always send dense logits.

A Megatron student evaluates cross-tokenizer KD **owner-local** when the run
allows it. Each context-parallel rank keeps its own share of the student
logits, evaluates only the alignment chunks whose final student predictor row
it holds, and reads only the teacher rows those chunks need. The student
prefix of a many-to-1 or many-to-many chunk may sit on other CP ranks; those
ranks add their share to one small vector that is summed across CP. Each rank
returns its share of the loss, so the shares add up to the full loss and
Megatron does not divide it by the CP size. Logged losses are complete sums.
Otherwise the loss is evaluated **replicated**: every CP rank rebuilds its
contiguous sequence window of the student and teacher logits.

| Student | Cross-tokenizer teacher | `teacher_topk_ipc_k` | Teacher transport | KD evaluation |
|---|---|---|---|---|
| Megatron, owner-local eligible | Megatron | `> 0` | Sparse rows of each teacher CP shard | Owner-local |
| Megatron, owner-local eligible | Megatron or Automodel | `0` | Dense | Owner-local |
| Megatron, not eligible | Megatron or Automodel | `0` | Dense | Replicated |
| Automodel | Automodel | `> 0` | Sparse | Replicated |
| Automodel | Megatron or Automodel | `0` | Dense | Replicated |

Setup rejects the remaining combinations: sparse transport with a Megatron
student that is not owner-local eligible or with an Automodel cross-tokenizer
teacher, and sparse transport with an Automodel student and a Megatron
cross-tokenizer teacher. Set `teacher_topk_ipc_k: 0` for those runs. Setup
prints which evaluation a run uses and why.

Owner-local evaluation requires:

- a Megatron student with `pipeline_model_parallel_size: 1`, no sequence
  packing, and no dynamic batching;
- `kd_loss_mode: sum` with static teacher weights (`sum_weights_metric: null`);
- `csd_pure_alm: false` and a `csd_loss_fn` other than `bce`;
  `csd_last_pos_loss_fn: bce` only with `csd_realized_only_mismatch: true`;
- with sparse transport, `teacher_topk_ipc_keep_realized: true` (or realized-only
  mismatch), Megatron cross-tokenizer teachers, and a teacher
  `make_sequence_length_divisible_by` that is a multiple of 2 x CP (and of TP
  when the teacher uses sequence parallelism).

CSD lockstep packing supports dense transport only, and therefore the
replicated evaluation.

Memory notes for owner-local evaluation:

- The student logits are never gathered across CP; only per-row scalars and
  the selected token values cross ranks. A same-tokenizer teacher is the
  exception: it keeps the replicated contiguous-window KL, which gathers the
  student logits.
- A sparse teacher payload is read row by row instead of being copied as a
  full-sequence `[B, T, k]` tensor on every GPU. A dense teacher payload is read
  as the full-vocabulary rows the rank's chunks need, never as a sequence
  window.
- `csd_selected_row_chunk_size` (student and teacher rows per tile) and
  `csd_support_chunk_batch_size` (chunks per batch) bound the temporary memory
  of this evaluation; they do not change the loss.
- The Megatron sparse sender keeps persistent IPC buffers on the teacher GPUs,
  sized once for `max_total_sequence_length`, until training ends. Their handles
  are reused across steps; the student reopens them every step, which drives
  PyTorch's IPC reference count for these buffers below zero. Values are
  unaffected, but the buffers may not be released after an error or at
  shutdown.

## Implementation and further configuration

- [CSD loss and configuration](../../nemo_rl/algorithms/loss/csd.py) define the
  CSD objective and optional common/mismatch loss controls.
- [Shared training controller](../../nemo_rl/algorithms/xtoken_off_policy_distillation.py)
  selects the algorithm and manages teacher execution, student updates,
  validation, and checkpoints.
- [Token alignment](../../nemo_rl/algorithms/x_token/token_aligner.py) and the
  [cross-tokenizer collator](../../nemo_rl/data/cross_tokenizer/cross_tokenizer_collate.py)
  are reused by both algorithms.
- [Lockstep packing design](../design-docs/csd-lockstep-sequence-packing.md)
  describes shared sample membership and backend-specific packing constraints.
  Use the exemplar's unpacked configuration for the initial smoke run.
- [X-Token preparation and training](xtoken-off-policy-distillation.md) uses
  the separate projection-matrix toolchain under `tools/x_token/`.

CSD native reasoning/tool alignment reuses the shared native templates.
