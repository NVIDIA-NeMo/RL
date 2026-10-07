# Agentic-VBench GRPO with context compaction

This checkout trains the **gym-native Agentic-VBench environment** (the one the
eval-factory work publishes from `workspace/agenticvbench-eval-factory/Gym`,
branch `agentic-vbench`) through NeMo-RL's SingleController GRPO path with
NeMo-Gym **token capture**, and optionally with **framework-owned context
compaction**: OpenCode may summarize its history mid-episode while NeMo-RL
reconstructs one training row per model context.

The environment is unchanged from evaluation: the 100 pinned tasks of
`PhiloLabs/agentic-vbench@410a75fe`, upstream Harbor 0.6.6 running stock
OpenCode 1.14.39 inside the task container on a remote rootless Podman
backend, native verifiers, and the NVIDIA Inference Hub judge for Repurpose.
See `3rdparty/Gym-workspace/Gym/benchmarks/agentic_vbench/README.md` for the
benchmark protocol and [context-compaction.md](./context-compaction.md) for
the reconstruction mechanism. This guide covers what training adds.

## Paired revisions

| Component | Revision | Source |
|---|---|---|
| NeMo-RL | `72235d78c85c7d1e6f2da6daa37a63787e14972a` (branch `cc-pr4200`) | [PR #4200](https://github.com/NVIDIA-NeMo/RL/pull/4200) head, stacked on [PR #4124](https://github.com/NVIDIA-NeMo/RL/pull/4124) `c617a2c01`, base main `08b846956` |
| NeMo-Gym | `cf90856658d161ccc70d7f5cf4bee75191ab07d1` ([PR #3368](https://github.com/NVIDIA-NeMo/Gym/pull/3368) head) + the ten `agentic-vbench` environment commits (`39363a70..d44b7154`, cherry-picked) + the training commit below | submodule branch `cc-pr3368` |
| Agentic-VBench tasks | `410a75fe3dae37de4344fc9e5317da089505330a` | pinned by `responses_api_agents/agentic_vbench_agent/core.py` |
| Harbor / OpenCode | 0.6.6 / 1.14.39 | upstream, unchanged |

The Gym PR base (`1bb369249`) is older than the eval-factory branch's base
(`2ba9a556`, Gym main); the environment commits are additive, so they apply
cleanly, but Gym core here is the PR's revision. The PRs are open and not yet
CI-qualified. The internal validation MR (manufactured environment, seeding and
observation patches, validation guide) was not available to this checkout.

## How the environment cooperates with capture

Stock OpenCode streams Chat completions to an OpenAI-compatible endpoint from
inside the task container, and the environment returns Harbor's ATIF trajectory.
NeMo-RL token capture needs two things that does not provide, so the training
commit adds a per-episode **capture proxy** (`capture_proxy.py`) between
OpenCode and Gym's rollout-scoped policy route:

- Framework-owned context (compaction training) accepts one *non-streaming*
  Chat completion per call. The proxy forwards OpenCode's request without
  `stream`, then replays the complete completion as the SSE chunk stream
  OpenCode expects (Gym's own `synthesize_chat_completion_sse`).
- NeMo-RL matches the agent's returned output to captured calls by content and
  tool-call id. The proxy records every served completion verbatim under
  `<episode>/model_calls/NNNN.json`; `capture_output` converts them with the same
  Chat conversion the capture ledger applies, so fingerprints are identical by
  construction. `terminal_response_id` is the last served response id.
- OpenCode tags calls with `x-opencode-agent`. Compaction summaries
  (`compaction`) are returned for training by default
  (`train_compaction_summaries`); session-title calls are disabled
  (`opencode_disable_title`) and ignored if they occur.
- The policy URL comes from `base_url_for_run(..., body)` with the request model,
  so the explicit capture id NeMo-RL sends names the ledger (a dumped body loses
  it). Evaluation runs (no token capture) still talk to Gym directly and return
  the ATIF output exactly as before.
- OpenCode's compaction policy is pinned by configuration
  (`opencode_compaction_auto`, `opencode_compaction_prune`); stock defaults keep
  both on. Under ordinary capture both must be off (an append-only context);
  under context compaction `auto` is on and `prune` off by default.

Compaction trigger (stock OpenCode): once a completion's total tokens reach
`model_context_tokens - min(model_output_capability_tokens, OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX)`,
OpenCode summarizes older history and continues from the summary plus the
recent turns. The runner sets that cap to the output limit. Advertise the
policy's real context (49152 for Super Omni) so compaction happens before the
engine overflows; an overflow would surface as an empty `length` completion
through the capture route.

Environment variables read by `configs/agentic_vbench_train.yaml`:

| Variable | Default | Effect |
|---|---|---|
| `AGENTIC_VBENCH_ROOT` | fetched into the cache | Pinned benchmark checkout (must be clean at `410a75fe`) |
| `AGENTIC_VBENCH_HARBOR_PYTHON` | required | Python with `harbor==0.6.6` and `python-dotenv` |
| `AGENTIC_VBENCH_OUTPUT_ROOT` | set by the launcher | Episode artifacts (Harbor jobs, runner logs, recorded model calls) |
| `AGENTIC_VBENCH_RUNTIME_ROOT` | set by the launcher | Node-local scratch for Podman/Harbor state |
| `AGENTIC_VBENCH_DOCKER_CLI_DIR` | fetched into the cache | Static `docker` + `cli-plugins/docker-compose` |
| `AGENTIC_VBENCH_BACKEND_DIR` | required | Directory where the backend job publishes `client.env` |
| `AGENTIC_VBENCH_JUDGE_PROTOCOL` | `nvinference-hybrid` | Hub judge for Repurpose; empty disables judge staging |
| `AGENTIC_VBENCH_CONTEXT_TOKENS` / `_OUTPUT_TOKENS` | `49152` / `8192` | OpenCode's model limits; output is also per-call `max_tokens` |
| `AGENTIC_VBENCH_MAX_TURNS` | `32` | Cap on OpenCode's agentic steps |
| `AGENTIC_VBENCH_COMPACTION_AUTO` / `_PRUNE` | unset (OpenCode defaults) | Pin OpenCode compaction / pruning |
| `AGENTIC_VBENCH_TRAIN_COMPACTION_SUMMARIES` | `true` | Return summary generations for training |
| `AGENTIC_VBENCH_CONCURRENCY` | `12` | Harbor trials in flight per agent server |

## Container

No prebuilt image on this cluster matches RL main at `08b846956` (vLLM 0.29.0,
torch 2.13, Transformer Engine 2.18). Build one from the public CUDA base with
Enroot on a CPU node:

```bash
cd /lustre/fsw/portfolios/nemotron/users/arushig/nemo-rl-cc-pr4200
CODE_DIR=$PWD sbatch --export=ALL,CODE_DIR=$PWD --account=nemotron_n4_omni --partition=cpu \
  --cpus-per-task=64 --mem=700G --time=12:00:00 --job-name=nrlcc-container-build \
  --output=/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200-build/slurm-%j.log \
  scripts/agentic_vbench/container/build_container.sbatch
```

`build_inside.sh` replays `docker/Dockerfile`'s base, hermetic and release stages
(uv 0.11.28, Python 3.13.14, `uv lock --check`, `uv sync --frozen` for the
default, `vllm`, `mcore`, `automodel` and `modelopt` extras, one container-owned
venv per Ray actor under `/opt/ray_venvs`, editable installs, the container
fingerprint); SGLang and TRT-LLM are skipped. The checked-in `uv.lock` pins Ray
2.58.0 for the whole workspace so the head and every worker venv agree. The rootfs lives on the node's tmpfs (the `/raid` Enroot
path is quota-limited) with the home mount disabled (root remapping would
otherwise put uv's cache in the quota-limited home). Output:
`/lustre/fsw/portfolios/nemotron/users/arushig/images/nemo-rl-cc-pr4200.sqsh`
(`CONTAINER` overrides). Training mounts this checkout's `nemo_rl/`, `examples/`,
`tools/`, `scripts/` and the Gym/Megatron-Bridge submodules over `/opt/nemo-rl`.

Validate the image on a CPU node before spending GPU time; this imports every
worker venv and runs `validate_single_controller_config` on the recipe with
compaction off and on:

```bash
bash scripts/agentic_vbench/container/validate_container.sh
```

## Runtime inputs

Reuse the inputs the eval-factory runs already prepared (git-ignored symlinks
under `data/agentic_vbench/`):

```bash
bash scripts/agentic_vbench/link_shared_assets.sh \
  --benchmark /lustre/fsw/portfolios/nemotron/users/arushig/workspace/agenticvbench-eval-factory/agentic-vbench \
  --harbor-venv /lustre/fsw/portfolios/nemotron/users/arushig/pivot-rl-video/data/agentic_vbench/harbor-venv \
  --docker-client /lustre/fsw/portfolios/nemotron/users/arushig/workspace/agenticvbench-eval-factory/docker-client
```

The Harbor venv's interpreter must live on the shared filesystem so the venv
also runs inside the training container (the pivot-rl-video one does; the
eval-factory `.venv-harbor-native` resolves into `$HOME`). `setup_harbor.sh`
creates a fresh one with uv. The benchmark checkout is used read-only; the
agent refuses a checkout whose `tasks/` differ from the pinned revision.

Task containers run on a rootless Podman backend on a CPU node:

```bash
export AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-smoke   # new dir per backend
bash scripts/agentic_vbench/submit_backend.sh
until test -s "$AGENTIC_VBENCH_BACKEND_DIR/client.env"; do sleep 10; done
```

Cancel that allocation after training; its endpoint exists only while it runs.
Repurpose tasks judge through the Hub: `run_sc.sh` loads `NVINFERENCE_API_KEY`
from `~/.config/agentic-vbench/nvinference.env` into the job environment
(`AGENTIC_VBENCH_CREDENTIALS_FILE` overrides the path) and never logs it.

## Recipe and launchers

`examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-agentic-vbench-sc.v1.yaml`
derives from the Super Omni async-Gym recipe and switches to SingleController:
`grpo.async_grpo: null`, `data_plane` (TransferQueue, simple backend),
`token_capture.enabled: true`, `async_rl` with lag-1 in-order sampling, and the
constraints `validate_cc_objective` enforces (token-level loss, no advantage
clipping or message-level advantage overrides, `seq_logprob_error_threshold: null`,
no sequence packing or dynamic batching, `min_step_batch_fraction: 1`). It loads
`responses_api_agents/agentic_vbench_agent/configs/agentic_vbench_train.yaml`.

| Script | Purpose |
|---|---|
| `scripts/agentic_vbench/build_train_dataset.py` | Rows from the pinned inventory (`--tasks "repair assembly"`, `--limit`, `--repeats`) with the Gym routing stamp |
| `scripts/agentic_vbench/run_sc.sh` | Preflight (checkout revision, Harbor, Docker client, backend, judge key) then `launch_sc.sh`; Hydra overrides pass through |
| `scripts/agentic_vbench/launch_sc.sh` | Slurm submission (`ray.sub`) with container-owned venvs; `CONTEXT_COMPACTION=true|false`, `DRY_RUN=1` |
| `scripts/agentic_vbench/smoke_cc_step120.sh` | 2 steps, 8 nodes (4 train / 4 gen), 8 tasks × 4, 20k OpenCode context (compaction at 16k) inside 12-turn episodes, no judge |
| `scripts/agentic_vbench/compare_cc_20steps.sh` | Paired 20-step arms on identical prompts: `compaction`, `identity` (CC route, no rewrite), optionally `off` (ordinary capture) |

```bash
AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-smoke \
  DRY_RUN=1 bash scripts/agentic_vbench/smoke_cc_step120.sh
AGENTIC_VBENCH_BACKEND_DIR=$PWD/data/agentic_vbench/backend-cc-smoke \
  bash scripts/agentic_vbench/smoke_cc_step120.sh
CC_ARMS="compaction identity" AGENTIC_VBENCH_BACKEND_DIR=... \
  bash scripts/agentic_vbench/compare_cc_20steps.sh
```

Sizing used here for the comparison (the 2-/3-node figures in the internal guide
are for its stub model; Super Omni needs TP8/EP16): 8 nodes per arm (4 training,
4 generation, `SEGMENT_SIZE=4`), 8 prompts x 8 generations, all 100 tasks with
two repeats (`data/agentic_vbench/train/all_x2.jsonl`, 200 rows for 160 prompt
slots), `AGENTIC_VBENCH_CONCURRENCY=32`, `SAVE_PERIOD=2` so chained job cycles
(`JOB_CYCLES`) can resume after a cluster cancellation, and every task image
prebuilt on the backend beforehand (`prebuild_task_images.py` with the
`all_x2_manifest.json`). Non-native families (repurpose, sequencing,
understanding) need the Hub judge, so `AGENTIC_VBENCH_JUDGE_PROTOCOL` stays at
its default and the credentials file must exist.

Each run writes `results/<name>/{logs,checkpoints,token_capture,episodes,slurm}`
and an `avb_environment.env` snapshot. Compare per-step reward, response length,
`model_calls`/`compaction_summaries` from the Gym responses log, and training-row
counts between arms; audit exact tokens and masks from `token_capture/`, and
inspect `episodes/<task>-<key>/model_calls/` for the served completions.

## Validation status

Verified here:

- Environment tests in the Gym submodule venv (55 passed): the original
  eval-factory suites plus the capture proxy (streamed request → non-streaming
  upstream on the rollout-prefixed route, SSE replay with usage, passthrough,
  upstream errors forwarded and marked), an agent-level capture flow through the
  real proxy (explicit capture id honored, output fingerprints equal to Gym's
  `completion_items` for the served completions, compaction/title knobs reaching
  the runner), runner config knobs, and the training config resolving from the
  environment.
- Launcher dry run from the pinned checkout (dataset manifest, SingleController
  driver command).
- The container (`nemo-rl-cc-pr4200.sqsh`, 2026-10-06): every worker venv
  imports (ray 2.56.1, vLLM 0.29.0 / torch 2.13, Transformer Engine 2.18,
  Gym with the agent and proxy, reassembler), and
  `validate_single_controller_config` accepts the recipe with
  `token_capture.context_compaction` off and on (`validate_container.sh`).
  Three recipe settings came out of that check: `token_capture.capture_dir`
  must be declared for the launcher override, the stall watchdog must exceed
  the 7200 s rollout timeout, and the in-order sampler needs buffer capacity
  for two steps of prompt groups.
- The first GPU attempt (8 nodes) brought up Ray on B200 nodes, prewarmed the
  Super Omni processor and built the master config, then stopped in
  `setup_single_controller`: SingleController checkpointing with the built-in
  samplers requires `checkpointing.save_data_plane: true` (a setup-time rule the
  CPU validator does not run). The recipe now sets it.
- The second and third attempts got through data loading and topology selection
  and died at the first actor creation with "Failed to connect to GCS". Ray's
  synced `gcs_server.err` showed the GCS aborting on an unregistered pubsub
  channel. Cause: a Ray version skew inside the image. The driver venv that runs
  the Ray head had ray 2.56.1 (RL's lock), but every worker venv had ray 2.59.0,
  because the Gym PR head requires `ray>=2.58.0`, RL's `uv.lock` was out of date
  for it, and the venv prefetch re-syncs non-frozen. A 2.59 worker publishing
  its version-mismatch error crashed the 2.56 GCS. Fix: `uv lock` with Ray
  pinned to 2.58.0 (the version the PRs were qualified on), re-sync every venv in
  the image (`container/patch_container.sbatch`), and make the build fail on a
  stale lock (`uv lock --check` in `build_inside.sh`).
- The fourth attempt passed worker creation (32 vLLM + 32 Megatron actors) and
  failed in the Megatron worker constructor: NeMo-RL's Megatron context
  parallelism requires sequence packing, which context compaction forbids
  (fixed-batch execution padding). The CC recipe therefore uses CP=1 with TP8
  sequence parallelism and full activation recomputation; PP=2 is the fallback
  if 49k-token rows do not fit.
- The fifth attempt reached model construction and every Megatron worker
  rejected `mtp_num_layers: 0` ("hybrid_layer_pattern defines 1 MTP depths"):
  the current Megatron-Bridge maps the checkpoint's single MTP block into the
  layer pattern. The recipe now carries it like upstream's tested `-mtp`
  variant (one layer, not repeated, detached heads) with loss weight 0, so the
  head is refit but not trained and the objective stays plain token-level GRPO.

- The sixth attempt loaded the model (Megatron checkpoint and vLLM weights) and
  stalled in vLLM's kernel warm-up: FlashInfer MoE autotune at
  `max_num_batched_tokens` (49152) with a cold cache (flashinfer 0.6.18) had
  not finished after 20 minutes when the cluster scheduler service cancelled
  the job at 30 minutes, before any Gym server came up. The recipe now sets
  `vllm_kwargs.enable_flashinfer_autotune: false` (heuristic kernels; re-enable
  once a warm `flashinfer_autotune_cache` exists under `VLLM_CACHE_ROOT`).
- Running `examples/nemo_gym/prefetch_venvs.py` on a CPU node
  (`scripts/agentic_vbench/prefetch_gym_venvs.sh`) exercised the real Gym config
  parser and rejected the training config: the Gym PR head requires an
  environment server in front of every bound agent
  (`AgentWithoutEnvironmentServerError`). `agentic_vbench_train.yaml` now
  declares `agentic_vbench_environment_server` (legacy_agent) like the
  eval-factory benchmark config. Rollout routing is unchanged:
  rows still carry `agent_ref.name = agentic_vbench_agent`.
- Startup work moved off the GPU allocation: task images are prebuilt on the
  backend (`scripts/agentic_vbench/prebuild_task_images.py`) and the Gym server
  venvs are created ahead of time into `NEMO_GYM_VENV_DIR`, so a job's first
  rollouts begin as soon as the policy is loaded.

- The seventh attempt (job 2166337) reached rollouts for the first time: Ray,
  model load, vLLM warm-up without autotune, Gym servers, refit, 8 task
  containers on the backend and captured model calls within 15 minutes. Every
  model call then failed: OpenCode sends chat requests without sampling
  parameters and the NeMo-RL vLLM server rejects an unset `top_p` (vLLM would
  resolve it from `generation_config.json`, off-policy). Fix: the recipe pins
  `policy_model.responses_api_models.vllm_model.sampling_overrides` to the
  policy's `generation.temperature`/`top_p`, Gym's documented knob for callers
  outside the training loop. (After the first failed call the model server
  refuses further calls in that attempt: "Cannot admit a call after incomplete
  capture", so one 500 fails the whole episode by design.)

- The eighth attempt (job 2166480) collected a full step of rollouts with the
  sampling pin: 32 episodes in 15 minutes at the agent's default concurrency of
  12, every model call served (about 13 calls and 6k-14k generated tokens per
  12-turn episode), OpenCode compacting twice per episode, verifier rewards
  recorded (all 0.0 on these assembly tasks under the 12-turn cap), and capture
  lineage written per call. The first optimizer step then failed in
  `validate_cc_execution_padding`: CC execution padding requires dropless
  per-token routing without auxiliary losses or router-bias updates, and the
  base recipe enables the expert bias with a 1e-3 update rate
  (`freeze_moe_router` freezes only the router weight). The recipe now sets
  `moe_router_bias_update_rate: 0.0`. The smoke also runs all 32 episodes of a
  step concurrently (`AGENTIC_VBENCH_CONCURRENCY`).
- Stock OpenCode 1.14.39 sends no `x-opencode-agent` header, so the proxy's
  compaction accounting fell back to request shape (no tools, one of the two
  fixed compaction prompts); `compaction_summaries` was 0 before that fix.
- The capture ledger logged "extends candidate ... with roles [...]; rooting a
  new segment" for 5 calls across 15 episodes: OpenCode replayed a known prefix
  and appended an assistant message (the compaction summary) rather than
  observations, which the splice cannot represent, so those calls render fresh.
  This is the designed fallback, not a failure; a compaction rewrite itself roots
  silently.
- Offline check of that attempt's artifacts (39 rollouts, 468 calls): the
  request-shape classifier finds 92 compaction summaries (2-4 per 12-turn
  episode); excluding them from training would drop 20-40% of each episode's
  generated tokens, so `train_compaction_summaries: true` is the meaningful
  default. The capture lineage shows 243 context roots (`mode: text`, fresh
  render) and 225 exact-token continuations (`mode: token_in`); per rollout the
  chain looks like `RccRRcRRcccR`, i.e. one root at the start and two per
  compaction (the summarization request has its own system prompt; the rewritten
  context after it starts a new segment), with continuations in between. Runs of
  consecutive roots late in some rollouts are OpenCode compacting on nearly every
  turn once image-heavy tool output keeps the 20k smoke context above the
  threshold; the 49k comparison setting does not have that pressure.

Not yet verified (requires GPU allocations):

- A two-step compaction smoke and the 20-step paired comparison.
- Stock OpenCode's compaction under framework-owned capture end to end: the PR
  authors qualified a custom fork, not stock OpenCode. The proxy makes the
  requests non-streaming and sequential, which is what the route requires, but
  the first smoke decides.
- Images under compaction: `single-controller.md` still lists "Compaction" as
  unsupported for media capture; that sentence dates from the media-capture commit
  (`08b846956`) that PR #4124 builds on, and PR #4124's reassembler fetches and
  verifies media occurrences under `context_compaction`. OpenCode's `read` tool
  sends PNG frames inline, so this is the first real-harness exercise of that
  combination; a rejected rollout surfaces as `retained_media_changed` or
  `media_chain_incompatible` in the finalizer metrics.
- Row counts: one rollout should publish about `1 + 2 x compactions` rows.
  Counts close to `model_calls` would mean OpenCode's re-serialized history is
  treated as a rewrite on every call (still exact, no prefix reuse).

Known limits from the PRs: concurrent or branching model calls are rejected, CC
groups are not redispatched on transport failure, and CC checkpoint/resume comes
from PR #4200.
