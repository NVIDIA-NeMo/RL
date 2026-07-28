# Nano SWE RL with the TransferQueue Data Plane

A reproducible 6-node smoke recipe that runs agentic SWE RL on
Nemotron-3-Nano-30B-A3B with rollouts flowing through the **TransferQueue (TQ)
data plane** instead of an in-process buffer.

It exists to answer one question end to end: *which NeMo-RL entrypoint actually
puts SWE rollouts through TransferQueue, and what does a config need for that to
work?* The answer is `examples/run_grpo_single_controller.py`, and this recipe is
the smallest configuration where you can watch it happen.

## Verified result

Run on 6 GB200 NVL72 nodes (Slurm job 5648757, 2026-07-28):

| | |
|---|---|
| Entrypoint | `examples/run_grpo_single_controller.py` |
| Config | `examples/configs/ultra/nano_swe_teacher_sc.yaml` |
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Shape | train 4 nodes (TP2·PP2·CP4, EP1) / gen 2 nodes (vLLM TP2 → 4 engines) |
| Overrides | `grpo.num_prompts_per_step=2 policy.train_global_batch_size=8 grpo.max_num_steps=5` |
| Outcome | train steps 1–5 completed, no traceback, no OOM |
| Launched via | `swe_nano_sc_interactive.sh` (attach + run the driver by hand) |

Reproduced from a clean **batch** submission — job 5667301, 2026-07-28, same
6-node shape via `swe_nano_sc.sh` with no overrides beyond `max_num_steps=5`, so
the full 64-rollout batch and `min_groups_for_streaming_train=16`:

```
train step 5/5  trainer_v=5  lag=1
step_metrics: loss=6.26e-05  grad_norm=0.0097  num_valid_samples=64  reward=0.0
ray.sub exiting (exit_code=0)          # 63 min wall, warmup included
```

Per-step timing was generation-dominated, as an async SWE run should be:
`total 302s = exposed_generation 193s (64%) + policy_training 69s (23%) +
logprobs 35s (12%) + weight_sync 4s (1%)`.

The TQ actors were live and serving the rollout→train hop throughout:
`TransferQueueController` plus two `SimpleStorageUnit` actors logging `PUT_DATA`,
`GET_DATA`, `KV_RETRIEVE_META`, `SET_CUSTOM_META` and `CLEAR_DATA`.

The final step reported `loss=7.7e-05`, `grad_norm=0.013`,
`global_valid_toks=112555`. **Rewards were 0.0 for every rollout** — the base
Nano model solves no SWE-bench instance in a 5-step smoke. This recipe validates
the *loop*, not model quality.

## Which paths honour the data plane

This is the part that is easy to get wrong: enabling `data_plane` in a config
does not mean it is used. Verified against the source:

| Entrypoint / mode | TQ honoured? | Why |
|---|---|---|
| `run_grpo_nemo_gym.py`, `async_grpo.enabled=true` | **No** | `async_grpo_train` builds the in-memory `ReplayBuffer` (`nemo_rl/algorithms/grpo.py:3898`); the TQ-backed `TQReplayBuffer` (`nemo_rl/algorithms/async_utils/replay_buffer.py:638`) is never constructed. The `data_plane` block is silently a no-op. |
| `run_grpo_nemo_gym.py`, `async_grpo.enabled=false` | Yes | `grpo_train_sync` reads/writes through the data plane. |
| `run_grpo_single_controller.py` | Yes, and **required** | The entrypoint raises `ValueError` unless `data_plane.enabled=true`, and `SingleControllerActor` commits each group via `TQReplayBuffer` → `dp_client.put_samples` (`nemo_rl/algorithms/single_controller.py:259`). |

Since the production SWE paradigm is async (sync leaves the 16 training GPUs
idle for the ~11 minutes a 64-rollout SWE batch takes), **SingleController is the
only path that gives async *and* a real data plane** — hence the recipe below.

## Quick start

All launchers share `swe_nano.env` and override the entrypoint and config. Run
them from a networked shell at the repo root (the Slurm controller is
unreachable from sandboxes).

Unattended reproduction, one command:

```bash
DRY_RUN=0 bash swe_nano_sc.sh grpo.num_prompts_per_step=2 policy.train_global_batch_size=8
```

Without `DRY_RUN=0` it prints the resolved driver command and exits, which is
worth doing once. Interactive mode — allocate, keep Ray up, run the driver by
hand, edit, re-run — is what the verified run used and is the better choice
whenever you expect to change anything:

```bash
# async GRPO via SingleController + honoured TransferQueue  (the verified recipe)
bash swe_nano_sc_interactive.sh

# sync GRPO + honoured TransferQueue  (simplest data-plane smoke)
bash swe_nano_tq_interactive.sh

# async GRPO, no data plane  (baseline)
bash swe_nano_interactive.sh
```

The batch and interactive paths build the identical driver command; batch just
lets `ray.sub` run it instead of handing it to you.

Each interactive launcher allocates 6 nodes, starts Ray, idles, and prints:

```bash
bash <jobid>-attach.sh        # shell on the head node, Ray already up
source <jobid>-run-cmd.sh     # run the driver; edit + re-source to iterate
```

Iterating inside one allocation is the point — a cold start pays for the ~60 GB
checkpoint download plus Megatron conversion and vLLM graph capture before the
first rollout, so you do not want to requeue for every config fix.

To reach a training step quickly instead of waiting for the default 64-rollout
batch:

```bash
bash swe_nano_sc_interactive.sh grpo.num_prompts_per_step=2 policy.train_global_batch_size=8
```

Cancel with `scancel <jobid>`.

## What to change before you run it

`swe_nano.env` splits its paths into two blocks:

**Shared read-only — reuse as is.** The training container, the nemo-skills
sandbox image, the SWE prompt set (`swe.jsonl`, 7816 instances) and the SWE-bench
`.sif` images are all world-readable on Lustre. They total hundreds of GB; do not
copy them.

**Per-user write — change all of these.** They are owned by `zhiyul` and you
cannot write to them:

| Variable | What lands there |
|---|---|
| `CODE_DIR` | your checkout of this branch — it is mounted into the container, so it must be the tree you are editing |
| `WORKSPACE_DIR` | results, Ray logs, checkpoints |
| `HF_HOME` | HuggingFace cache (~60 GB for the Nano BF16 checkpoint) |
| `PERSISTENT_CACHE` | vLLM / Triton / Inductor compile caches, reused across jobs |
| `SLURM_ACCOUNT` | an account you can charge |
| `EXP_NAME` | namespaces `results/`, `ray_logs/` and the W&B run |

Export `HF_TOKEN` too if you can — the verified run downloaded the checkpoint
unauthenticated and was rate limited.

## Config chain

```
nano_swe_teacher_sc.yaml            SingleController + TQ  (the recipe)
└── nano_swe_teacher_qwen3mesh.yaml TP2·PP2·CP4, EP1, alltoall, vLLM TP2, data_plane block
    └── nano_swe_teacher.yaml       49k context, gym venv reuse, sif_dir
        ├── swe_teacher.yaml        the Ultra SWE stage (rewards, data, optimizer)
        └── _nano_smoke_gb200.yaml.inc  Nano/GB200 mesh + smoke sizing
```

`nano_swe_teacher_sync_tq.yaml` is the sync variant, branching off
`nano_swe_teacher_qwen3mesh.yaml`.

### Why each SingleController-specific setting is there

Every one of these came from a crash on the way to the working run:

| Setting | Without it |
|---|---|
| `policy.draft.enabled: false` | `KeyError: 'draft'` — `run_grpo_single_controller.py` reads `config.policy["draft"]["enabled"]` unconditionally, and the SWE/Ultra chain has no draft block. |
| `loss_fn.reference_policy_kl_penalty: 0.01` | `AttributeError: 'MegatronPolicyWorker' object has no attribute 'reference_state_dict'` — the SC refit swaps in reference weights, but the reference model is only initialized when the KL penalty is > 0 (`single_controller_utils/setup.py:208`). The SWE base sets it to 0. |
| `grpo.val_period: 0`, `checkpointing.enabled: false` | SC supports neither validation nor checkpointing yet. |
| `async_rl` block | SC uses its own async config, not `grpo.async_grpo`. `min_groups_for_streaming_train` decides how many groups buffer before the first training step. |
| `moe_token_dispatcher_type: alltoall` (in the mesh config) | `AssertionError: hybrid-ep kernel ... at least 2 ranks, but got 1` — the inherited flex/hybridep dispatcher requires EP≥2, and this recipe runs EP1. |
| `data_plane.global_segment_size` / `local_buffer_size` | `DataPlaneConfig` requires them even though only the `mooncake_cpu` backend reads them; the `simple` backend ignores the values. |

Two more constraints that are not config fields:

- **Batch invariant.** `num_prompts_per_step × num_generations_per_prompt` must
  equal `train_global_batch_size` on the SC split path (one RL step = one
  optimizer step). Overriding one without the other raises `ValueError`.
- **Absolute entrypoint path.** `run_grpo_single_controller.py` must be spawned
  from `${CODE_DIR}` by absolute path. The launcher mounts only `nemo_rl/` and
  `examples/configs` over the container, so the container's baked `examples/` has
  no such file and `uv run ./examples/...` fails with "Failed to spawn".

## Verifying TQ is really engaged

Config echo alone is not evidence — the async `run_grpo_nemo_gym.py` path prints
`data_plane={'enabled': True, ...}` and then ignores it. Look for **runtime
actors** in the driver log:

```
(TransferQueueController pid=...)  Per-operation statistics: NOTIFY_DATA_UPDATE: ...
(SimpleStorageUnit pid=..., ip=...) Per-operation statistics: PUT_DATA: req_count=... GET_DATA: ...
(TransferQueueController pid=...)  ... KV_RETRIEVE_META: ... SET_CUSTOM_META: ...
```

`PUT_DATA` is a rollout committing to TQ; `KV_RETRIEVE_META` is the trainer
consuming it. Then confirm training progresses:

```
🚀 Launching SingleControllerActor
train step 5/5  trainer_v=5  lag=1
step_metrics={'loss': ..., 'grad_norm': ..., ...}
```

## Where the metrics are

**W&B is enabled automatically from a secrets file.** `swe_nano.env` sources
`NRL_SECRETS_FILE` (a file of `export VAR=...` lines) before anything else, so
every nano launcher picks up `WANDB_API_KEY` and `HF_TOKEN` without the caller
exporting anything. Point it at your own file:

```bash
NRL_SECRETS_FILE=/path/to/secrets.sh DRY_RUN=0 bash swe_nano_sc.sh
```

The launcher prints which secrets it found — `[SECRETS] sourced ... (wandb=set
hf=set)` — and the summary line confirms the result: `W&B: nano-swe-smoke /
<EXP_NAME> (enabled=True)`. With no readable secrets file you get a `[WARN]`, and
`ultra_launch.sh` falls back to `logger.wandb_enabled=False`: no dashboard link,
plus rate-limited unauthenticated HF downloads for the ~60 GB checkpoint.

Project and run name are wired from `WANDB_PROJ=nano-swe-smoke` and `EXP_NAME`,
so the run lands at `https://wandb.ai/<entity>/nano-swe-smoke/runs/<run>`.

**Without W&B, the per-step metrics are in the SingleController actor's own log**,
not in the driver log — this trips people up, because `ray-driver.log` shows the
rollout progress bars and TransferQueue actor stats but never prints `train step`:

```bash
ls workspace/ray_logs/<EXP_NAME>/<jobid>-logs/ray/session_*/logs/worker-*.out
grep -E "train step|step_metrics|_sync_weights" <that file>
```

That file carries `train step N/M  trainer_v=..  lag=..`, the per-phase timing
breakdown (`exposed_generation`, `policy_training`, `weight_sync`), and the full
`step_metrics` dict. `logger.log_dir` under the run directory holds the same
metrics on disk.

## Known limits

- **Rewards are 0.** The base model solves no SWE instance in a smoke run. Real
  signal needs a capable policy and many more steps.
- **EP1 is memory-hungry.** Expert parallelism of 1 replicates every expert on
  each MP rank. With the reference model loaded (required by SC, see above) peak
  usage was ~170 GB/rank of 189 GB. If you scale up context, batch, or model,
  expect the initial weight sync to OOM first; raise EP, enable
  `optimizer_cpu_offload`, or shorten the sequence.
- **Walltime and QOS.** `WALLTIME=3:59:00` (the `batch` partition maximum) with
  no QOS. Two hours is the practical floor — shorter allocations expired
  mid-warmup during development. The recipe deliberately avoids the `short` QOS:
  it trades a priority boost for a 2h ceiling and a per-user node cap, and a
  6-node job here was held with `Reason=QOSMaxNodePerUserLimit` while a larger
  job of the same user was running. For scale: with `num_prompts_per_step=2` a
  4-rollout batch took 6–8 minutes; the default 64-rollout batch took ~11.
- **Submodule pin.** This branch pins `3rdparty/Gym-workspace/Gym` to `v0.4.0` to
  match the container's prebuilt gym venvs (`skip_venv_if_present: true`).
  Bumping it without rebuilding the container forces a concurrent nemo-gym
  rebuild and a uv cache lock timeout.

## Related

- `nemo_rl/data_plane/factory.py` — data-plane client construction and backends
- `nemo_rl/data_plane/docs/data-plane-async-proposal.md` — design notes
- `docs/guides/async-grpo.md` — the async trainer this recipe replaces
- `examples/configs/grpo_math_1B_megatron_single_controller.yaml` — the math
  SingleController reference config the SC blocks were adapted from
