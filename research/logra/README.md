# LoGRA: low-rank gradient sketches for NeMo RL

LoGRA replaces persistent dense gradients of selected weights with low-rank
sketches. The selected model weights still change; there is no persistent adapter
branch. This research project provides three examples without changing NeMo RL or
AutoModel core code:

| Example | Training loop | Sketch optimizer | Rollout engine |
|---|---|---|---|
| `run_grpo.py` (math) | Native synchronous GRPO | RowAdam; native AdamW on other weights | Native vLLM |
| `run_grpo.py` (sliding puzzle) | Native multi-turn GRPO | RowAdam; native AdamW on other weights | Native vLLM |
| `run_cua.py` | Token REINFORCE on OSWorld desktops | SGD with finite-difference KL scaling | Native vLLM HTTP |

All examples use native **full-weight** policy synchronization. They do not
implement compressed trainer-to-rollout transfers.

## 1. Environment

Use the repository's [Docker setup](../../docs/docker.md) and
[cluster setup](../../docs/cluster.md) to prepare NeMo RL with AutoModel and vLLM.
This version targets NeMo RL main `26e2b802` and uses the native AutoModel worker
and `policy.automodel_cfg` configuration block. Initialize the submodules before
installing dependencies. From the repository root:

```bash
git submodule update --init --recursive
cd research/logra
uv sync --locked --inexact --group test
```

The workspace uses Python 3.13.14. CUA additionally requires the repository's
`nemo_gym` extra and a separately provisioned OSWorld desktop runtime; follow
[CUA environment and data setup](docs/cua-setup.md) before running it.
Models, data, credentials, checkpoints, and logs remain outside Git.

The measured runs below used a prepared Python 3.12.3 GPU container with
PyTorch 2.13.0+cu130, Transformers 5.12.1 and vLLM 0.29.0. This is distinct from
the workspace-locked Python environment; the private OSWorld runtime has not been
validated in the latter. `analysis/environment.py` records imported versions and
source paths so that differences are visible.

## 2. GRPO: Qwen2.5-Math-7B

Allocate eight GPUs and run these commands from `research/logra`:

```bash
# Two-update smoke test.
uv run --locked run_grpo.py --config configs/grpo_logra_smoke.yaml

# 100-update LoGRA experiment.
uv run --locked run_grpo.py --config configs/grpo_logra_7b.yaml \
  grpo.seed=42 logger.log_dir=/output/logra-s42 \
  checkpointing.checkpoint_dir=/output/logra-s42/checkpoints

# Native Dense AdamW control: only the worker's LoGRA toggle changes.
uv run --locked run_grpo.py --config configs/grpo_logra_7b.yaml \
  logra.enabled=false grpo.seed=42 logger.log_dir=/output/dense-s42 \
  checkpointing.checkpoint_dir=/output/dense-s42/checkpoints
```

GSM8K is loaded through NeMo RL's dataset loader: `train` for training and the
first 256 `test` questions for evaluation. Download the Qwen checkpoint and data
in advance if compute nodes lack network access; override `policy.model_name`
and `policy.tokenizer.name` with the local model path. Repeat the two experiments
with `grpo.seed=43` and fresh output directories to reproduce the paired comparison.

| Setting | Dense and LoGRA |
|---|---|
| Model / data | Qwen2.5-Math-7B / GSM8K |
| Batch / context | 16 prompts × 8 responses; microbatch 1; 2,048 tokens |
| Hardware | 8 A100 80GB: 4 training GPUs + 4 rollout GPUs |
| Precision / memory | FP32 parameter storage, BF16 compute, activation checkpointing, packing |
| Optimizer schedule | LR 5e-6; 50-update warmup from 10%, then constant; weight decay 0.01 |
| Evaluation | 256 fixed questions every 10 updates; temperature/top-p 1.0/1.0 |

LoGRA uses rank 256, refreshed Rademacher projections, projection seed 42 and
RowAdam (`beta2=0.95`, `epsilon=1e-8`). RowAdam keeps one second moment per output
row and no first moment. Other trainable parameters retain native AdamW. The
native reference-policy KL penalty is 0.01 in both modes; **predicted-KL step
control is not enabled in this GRPO recipe**.

![Paired GRPO memory and evaluation curves](assets/grpo-7b-comparison.png)

Two training seeds, 100 updates each; mean ± sample standard deviation:

| Metric | Dense AdamW | LoGRA |
|---|---:|---:|
| Mean update peak, GiB / training GPU | 36.13 ± 0.03 | 19.36 ± 0.03 |
| Maximum update peak, GiB / training GPU | 37.28 ± 0.19 | 20.15 ± 0.03 |
| Final GSM8K subset score, % | 85.35 ± 5.25 | 88.48 ± 1.93 |

Mean update peak falls by **46.40%**. Memory excludes rollout GPUs and is not a
whole-node or time-averaged measurement. Both methods start at 42.97%; two seeds
on a 256-question subset do not establish superiority or equivalence. The full
curves include Dense seed 43's drop at the final evaluation.

## 3. Multi-turn GRPO: sliding puzzle

This public example reuses NeMo RL's sliding-puzzle prompt generator, game
logic, reward and GRPO loop. The model reads a text board and emits an action;
the environment returns the updated board. Solving the puzzle earns reward 1.
No desktop sandbox, external reward service or private dataset is required.

The recipe uses Qwen2.5-7B-Instruct and boards of size 2×2 or 3×3, generated
by the upstream example (1 or 3 shuffle moves). Each update has 8 boards × 4
trajectories, at most 8 rollout turns, 96 generated tokens per turn and a
2,048-token total context limit. It uses two training GPUs and two rollout GPUs,
FP32 parameter storage, BF16 computation, activation checkpointing, no sequence
packing or FSDP CPU parameter offload. Native optimizer offload during refit
remains enabled in both. LoGRA uses rank-256 refreshed Rademacher sketches
(projection seed 42) and RowAdam; the Dense control uses native AdamW. Both use
the same LR schedule and reference-policy KL penalty. There is no predicted-KL
controller here.

From `research/logra`, with four allocated GPUs:

```bash
# Two-update integration smoke; pass logra.enabled=false for Dense.
bash tests/functional/sliding_puzzle_logra.sh

# Paired 20-update memory profiles, with fresh directories for each run.
for seed in 42 43; do
  uv run --locked run_grpo.py --config configs/grpo_logra_sliding_puzzle.yaml \
    logra.enabled=false grpo.seed=$seed logger.log_dir=/output/puzzle/dense-s$seed
  uv run --locked run_grpo.py --config configs/grpo_logra_sliding_puzzle.yaml \
    logra.enabled=true grpo.seed=$seed logger.log_dir=/output/puzzle/logra-s$seed
done

uv run --locked analysis/export_events.py /output/puzzle --output /output/puzzle/memory.csv
uv run --locked analysis/plot_puzzle_memory.py /output/puzzle/memory.csv \
  --steps 20 --output /output/puzzle/sliding-puzzle-memory
```

![Sliding-puzzle training-memory comparison](assets/sliding-puzzle-memory.png)

Measured on one node with four NVIDIA GB200 GPUs (two training, two rollout),
using training seeds 42 and 43 with 20 updates each. The measured order was
Dense → LoGRA for seed 42 and LoGRA → Dense for seed 43.

| Method | Mean per-GPU update peak, GiB | Across-seed SD, MiB |
|---|---:|---:|
| Dense AdamW | 71.09 | 0.28 |
| LoGRA | 29.73 | 0.03 |

LoGRA reduces mean update peak by **58.19%**.
The seed variation is small, so the plotted bands and error bars are nearly
invisible. [Vector figure](assets/sliding-puzzle-memory.pdf).

This is a short memory profile, not a task-performance or convergence benchmark.
The metric is peak PyTorch allocated memory during each policy-training update,
averaged across training GPUs; rollout memory is excluded. All 20 updates are
included, and error bars represent sample standard deviation across the two
seed means; curve bands show the across-seed standard deviation at each update.
Boards are generated procedurally; the two methods' response lengths can differ.
GPU type and training-GPU count must match within a comparison. This profile
uses two training GPUs; the math run above uses four, so their absolute per-GPU
memory figures are not directly comparable.

## 4. Computer use: ScaleCUA and OSWorld

CUA needs model weights, task manifests, setup assets, a writable cache, and an
authorized sandbox endpoint. See [setup instructions](docs/cua-setup.md) for their
layout and manifest format. Provision the same paths and runtime on every node,
and initialize Ray using the repository's cluster guide.

```bash
# From research/logra, on 4 nodes × 4 GB200 GPUs.
export OSWORLD_ROOT=/assets/osworld
export OSWORLD_CREDENTIAL_FILE=/private/sandbox.json
export CUA_OUTPUT_DIR=/output/cua-s43

# One-update, two-episode smoke with one held-out task; no checkpoints.
uv run --locked --project ../.. --extra automodel --extra nemo_gym python \
  run_cua.py --config configs/cua_smoke.yaml \
  osworld.setup_cache=/writable/osworld_setup_cache

# Use a NEW output directory for training.
export CUA_OUTPUT_DIR=/output/cua-training-s43
uv run --locked --project ../.. --extra automodel --extra nemo_gym python \
  run_cua.py --config configs/cua_prod4.yaml training.max_steps=100 \
  osworld.setup_cache=/writable/osworld_setup_cache

# Resume the same experiment, retaining the original KL-annealing clock.
uv run --locked --project ../.. --extra automodel --extra nemo_gym python \
  run_cua.py --config configs/cua_prod4.yaml training.max_steps=200 \
  training.resume="$CUA_OUTPUT_DIR/checkpoints/step_100" \
  training.eval_at_start=false osworld.setup_cache=/writable/osworld_setup_cache
```

`cua_prod4.yaml` is the validated v2/verifier recipe; its name is retained for
existing launchers. `cua_osworld.yaml` retains the earlier mixed-template recipe,
which is **a different experiment**. Setting only `training.max_steps` does not
shorten `cua.kl_budget_steps`: the validated annealing schedule spans 600 updates.
Inline evaluations default to every 50 updates plus the final update. For an
independent evaluator, use the [checkpoint-request interface](docs/external-evaluation.md).

| Setting | ScaleCUA recipe |
|---|---|
| Model | Nemotron Omni RFC0040 iter881 SFT checkpoint |
| Resources | 8 training GPUs with EP8 + 8 rollout GPUs with four TP2 engines |
| Data / desktop | 815 ScaleCUA rows, 813 task IDs; `osworld-v2` |
| Collection | 128 completed episodes/update; 256 in flight; generation overlaps training |
| Context / sampling | 32K total context; 8,192 tokens/turn; temperature/top-p 1.0/1.0 |
| Objective | Token REINFORCE, prompt EMA baseline, token importance weights clipped to [0.5, 2] |
| Compression | Rank-256 refreshed Rademacher; SGD; other weights frozen |
| Step scaling | Proposal relative norm 6e-5; FD-KL budget 5e-4 → 5e-5 over 600 updates; α cap 3 |
| Checkpoints | Every 100 updates and at the final update; model, optimizer, baseline and pending task tickets |

![First 100 updates of ScaleCUA training](assets/cua-first100-training.png)

![Matched original-stack evaluation of the initial and step-100 policy](assets/cua-first100-evaluation.png)

The first 100 updates have mean training rewards of **62.93%** in updates 1–50
and **64.25%** in updates 51–100; the median logged update time is **4.81 min**.
A separately run original-stack evaluation gives **48.93% → 52.11%** over 361
tasks. This is one trajectory and one evaluation per checkpoint, with partial
verifier credit; the paired gain's task-level standard error is 2.41 points.
It does not establish long-horizon stability or a statistically significant gain.
The external evaluation uses a 49,152-token context, temperature/top-p 0.5/0.9
and `osworld-v2`; the inline evaluator is not asserted to reproduce that protocol.

Training and evaluation share 297 source task UUIDs and one exact instruction.
Removing the identical instruction gives **49.06% → 51.98%** on 360 tasks;
this subset is still not source-disjoint. Failed evaluation episodes remain in
reported task means. The first-100-update figures are an explicitly selected
window, not a claim about the whole longer trajectory.

## 5. Code map and supported scope

```text
run_grpo.py       Native GRPO setup and loop; math or sliding-puzzle data
run_cua.py        Desktop rollout → training batch → update → refit/checkpoint
logra/
  actor_environments.py, policy.py   Register/select research workers
  worker.py                         Native worker extension, metrics and checkpointing
  compression.py                    Linear sketches and transient expert-gradient projection
  optimizer.py, row_adam.py          Sketch updates and optimizer state
  setup.py, config.py                LR schedule and validated LoGRA settings
  cua.py                            REINFORCE, baseline, route replay and FD-KL scaling
  token_capture.py                  Exact tokens, screenshots and compressed CPU media
  rollout_pool.py                    Asynchronous task tickets and resumable sampling
  osworld.py                         Desktop adapters, batch construction and scoring
  evaluation_queue.py                Requests for external checkpoint evaluation
```

GRPO supports FSDP2 dense HF linear layers with TP=CP=1, synchronous execution,
and no PEFT, draft co-training, quantization or CPU parameter offload. CUA additionally supports the
validated hybrid attention/Mamba/MoE model with EP. Linear layers collect sketches
directly; fused experts briefly materialize their native dense gradient before
projection. Fused Mamba `out_proj` hooks are bypassed and those weights remain
frozen. Neither recipe claims arbitrary optimizer/model/backend compatibility.

FD KL control rescales a proposal using probe contexts; it is not a hard bound
on realized KL. Resume requires matching parallelism and CUA/OSWorld settings.
Pending task tickets are restored, but unfinished desktops are rerun rather than
resumed bit for bit. Old `logra.osworld` checkpoint class names remain importable.

## 6. Tests and analysis

```bash
uv run --locked --group test pytest tests/unit
uv run --locked --group test python -m torch.distributed.run \
  --standalone --nproc-per-node=2 tests/functional/fsdp_equivalence.py
uv run --locked --group test python -m torch.distributed.run \
  --standalone --nproc-per-node=4 tests/functional/expert_equivalence.py
uv run --locked --group test python -m torch.distributed.run \
  --standalone --nproc-per-node=4 tests/functional/cua_equivalence.py
bash tests/functional/grpo_logra.sh
bash tests/functional/sliding_puzzle_logra.sh
# Opt-in: real desktop/model prerequisites and full allocation are required.
LOGRA_RUN_CUA_TEST=1 bash tests/functional/cua_logra.sh
```

The numerical tests cover distributed sketch averaging, clipping, expert updates,
route replay, FD rollback/scaling and REINFORCE gradients. The model-level smoke
can also check multimodal HTTP capture, refit and checkpoint reload:

```bash
uv run --locked --project ../.. --extra automodel --extra nemo_gym python \
  tests/functional/cua_model_smoke.py --config configs/cua_prod4.yaml --with-generation
```

That model smoke uses four GPUs (EP2 training + TP2 rollout), synthetic screenshots,
and no desktop unless `--with-desktop` is supplied. It is not a benchmark score.
The opt-in CUA shell test skips when its external prerequisites have not been
requested; a skip is not a successful end-to-end validation.

Regenerate the GRPO figure from native TensorBoard logs (NumPy/Matplotlib required):

```bash
uv run analysis/export_events.py /path/to/results --output /output/comparison.csv
uv run analysis/plot_comparison.py /output/comparison.csv --output /output/comparison
uv run analysis/summarize_comparison.py /output/comparison.csv --output /output/summary.json
```

Raw experiment artifacts remain outside the source tree. Analysis rejects missing
updates/evaluations rather than interpolating them; bands are across-seed sample
standard deviations.
