# Agentic-VBench GRPO on the unified-teacher branch

This branch (`arushig/agentic-vbench-unified-teacher`) builds on
`rohit/unified-teacher-supervl3p5` and carries the minimal changes needed to
train Nemotron Super Omni on AgenticVBench through the SingleController path.
The full history of the environment, its validation and the context-compaction
work lives on the `cc-pr4200` branch; this page records only what differs here.

## Context compaction merged (2026-10-08)

After the first validated smoke the branch also absorbed `cc-pr4200` (merge
commit 6d9a282a6), so it now carries NeMo-RL PR #4200's framework-owned
context capture and the full cc-pr4200 AgenticVBench history alongside the
unified-teacher changes. Consequences:

- `CONTEXT_COMPACTION=true` works again through `launch_sc.sh`; the compaction
  arm uses `...-agentic-vbench-sc-256k.v1.yaml` (unpacked, CP=1, dense mask
  skipped), while the V2 recipe stays identity-only (packing, CP=2) and declares
  `context_compaction: false`.
- The Gym submodule is `cc-pr3368-unified`: the cc-pr3368 Gym (framework-owned
  route, environment-server layer, AgenticVBench agent) plus the two
  unified-teacher Gym commits (prefill boundary routes in captured extras,
  image-tools capture path) plus the optional proxy step-cap reply
  (`local_step_cap_reply`, default false here because the framework-owned route
  roots that turn as a segment). Its venvs live in
  `data/agentic_vbench/gym_venvs_cc_unified`.
- Merge resolutions: `route_assembly.py` keeps both route backpatch paths (the
  unified-teacher `routed_experts_prefix_boundary` and the CC
  `predecessor_tail_route`), the token-capture hosting test keeps both branches'
  cases, and the launcher is the cc one with this branch's defaults and guard.
- Known: one `image_tools_agent` test fails on the merged Gym ("Unexpected call:
  policy_model /v1/responses"); it does not involve the AgenticVBench path and
  needs a look by the image-tools owner.

The sections below describe the state before that merge and remain accurate
for the identity path.

## What is on this branch and not on the base

| Area | Change |
|---|---|
| Gym submodule | `arushig/agentic-vbench-unified-teacher` on top of the base's Gym branch: the gym-native AgenticVBench agent (`responses_api_agents/agentic_vbench_agent/`: Harbor 0.6.6, stock OpenCode 1.14.39, pinned benchmark, remote Podman backend, Hub judge, capture proxy) |
| Recipe | `examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-agentic-vbench-sc.v2.yaml` in the V2 format (`async_rl`, `data_plane`, `token_capture`), derived from `configs/examples/recipes/vlm/vlm_grpo-supervl3p5-unified-teachers-ready-first-32n4g.yaml` and the cc-pr4200 SC recipe |
| Launchers | `scripts/agentic_vbench/` (Slurm launch, preflight, dataset builder, backend, image prebuild, venv prefetch, smoke) and `tools/ensure_agentic_vbench_deps.sh` |

Nothing in `nemo_rl/` changes. In particular the branch has no
`token_capture.context_compaction`: token capture is the ordinary
one-row-per-rollout kind, so OpenCode's compaction and pruning stay off
(`AGENTIC_VBENCH_COMPACTION_AUTO/PRUNE=false`) and the launcher rejects
`CONTEXT_COMPACTION=true`.

## Recipe choices versus the cc-pr4200 SC recipe

- `async_rl.sampler: ready_first` with `max_staleness_versions: 2`, three steps
  of prompt groups in flight (`max_inflight_prompts`/`max_buffered_rollouts`),
  `on_dropped_prompt: shrink` (ready_first stamps no target step, so `replace`
  is a no-op there), and the data-plane sizing from the base recipe. Watch
  `timing/train/exposed_generation` in W&B: the point of ready_first is to hide
  the 10-25 minute agent episodes behind training.
- Sequence packing on with CP=2. Without framework-owned compaction the CC
  constraints (no packing, CP=1, token-level loss, no advantage clipping) do not
  apply, and packing removes the two learner memory limits hit on cc-pr4200
  (the dense `[L, L]` causal mask on unpacked rows and the step-2 OOM at 70-80k
  tokens), so rows up to the 256k window are trainable.
- Everything else matches the validated cc-pr4200 settings: one detached MTP
  block with zero loss weight, frozen router with zero bias update rate,
  sampling pinned on Gym's model server, 64 inline images per prompt, FlashInfer
  autotune off, checkpoint every step with the data-plane snapshot.

## Ordinary capture and OpenCode's step-cap turn

When OpenCode's step budget is spent it appends an assistant message
("CRITICAL - MAXIMUM STEPS REACHED ...") and asks for a text-only reply. That
message is not a served completion, so ordinary parent resolution cannot attach
the turn to the chain and the ledger poisons the call, masking the rollout (62 of
64 rollouts in the 2026-10-07 off-mode run). The agent config therefore sets
`local_step_cap_reply: true`: the proxy answers that turn itself, records it as
an ignored side call, and the episode ends normally. Verifiers score the files
the agent produced, not that final text.

## Container

The base recipe was run in `nemo-rl.nightly-gym.20260930.sqsh`, an aarch64
image for a 4-GPU-per-node (GB200) cluster; it cannot start on the x86 B200
nodes here (`enroot-switchroot: /bin/bash: Exec format error`). The base
branch's `uv.lock` differs from cc-pr4200's in exactly one package (Ray 2.56.1
versus 2.58.0, and this Gym accepts `ray>=2.56.1`), so the x86 image
`images/nemo-rl-cc-pr4200.sqsh` is the faithful equivalent and is the launcher
default. `validate_single_controller_config` accepts the V2 recipe in it.

## Running

Initialize the submodules the launcher mounts over the image first (a fresh
worktree has them empty, which hides the installed `megatron.bridge` and
`megatron.core` and fails at worker creation):

```bash
git submodule update --init --depth 1 3rdparty/Gym-workspace/Gym
git submodule update --init --recursive --depth 1 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
```

The launcher's preflight now refuses to submit while any of them is empty.

```bash
AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-3 bash scripts/agentic_vbench/submit_backend.sh   # if no backend is up
python3 scripts/agentic_vbench/prebuild_task_images.py --benchmark-root data/agentic_vbench/agentic-vbench \
  --manifest data/agentic_vbench/train/assembly_first8_x4_manifest.json --backend-dir data/agentic_vbench/backend-cc-3 \
  --docker-cli-dir data/agentic_vbench/docker-client --log-dir results/prebuild
WANDB_API_KEY=... WANDB_MODE=online AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-3 \
  bash scripts/agentic_vbench/smoke_identity_32turns_256k.sh
```

`data/agentic_vbench/` is git-ignored and shared by path with the cc-pr4200
checkout (`link_shared_assets.sh`): the pinned benchmark checkout, Harbor venv,
Docker client, datasets and backend directories.

Known operational constraints carried over: the cluster service
`svc-hwinf-cs-sched` has cancelled every multi-node AgenticVBench job past
about 30-41 minutes (hence per-step checkpoints and chained cycles), and
Harbor installs OpenCode inside every task container at trial start (keep
`AGENTIC_VBENCH_CONCURRENCY` around 16 or the Node download fails under load).

## Validation status

First smoke on this branch (job 2176033, 2026-10-08; 8 nodes, 4 prompts x 4
generations, 12 turns, 256k window, W&B online
`agentic_vbench_ut_smoke_4x4_12t_20261008-160340`):

- Step 1 trained end to end with sequence packing and CP=2 and wrote its
  checkpoint at 22 minutes: total step time 206 s (training 23 s, logprobs 9 s,
  checkpoint 126 s), `timing/train/exposed_generation` 0.0008 s (ready_first
  kept generation fully hidden behind training), 16 valid samples, 0 poisoned
  rollouts, 0 masked, token-in continuation rate 0.91, token-mult-prob error
  1.04. Rewards were all 0.0 at 12 turns (loss 0), so the step validates the
  pipeline rather than learning; the 32-turn configuration is the one with
  reward signal.
- The proxy's local step-cap reply fired in every episode that hit the turn
  cap, and ordinary capture attached every served call (no "Unresolved parent"
  poisoning), confirming the fix for the failure seen on the cc branch's
  off-mode run.
- Earlier in the day the first launch failed at worker creation because the
  fresh worktree's Megatron-Bridge submodule was empty under the launcher's
  mount (now a preflight check), and the Gym venv prefetch rejected the
  environment-server stanza this Gym does not have (removed on this branch).
