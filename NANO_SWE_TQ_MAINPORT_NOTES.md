# Running Nano SWE + TransferQueue on `main`

This reproduces `docs/guides/nano-swe-transferqueue.md` (branch `zhiyul/swe_tq`) on
today's `main` branch, instead of on that branch. It works: job `6292902` ran all 5
GRPO steps, TransferQueue was actually active, and the model solved part of a real
SWE-bench task twice (`reward=0.25`).

## TL;DR — run it

```bash
cd /lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/RL

# dry run: prints the command it would run, submits nothing
bash swe_nano_sc.sh grpo.num_prompts_per_step=2 policy.train_global_batch_size=8 grpo.max_num_steps=5

# real run
DRY_RUN=0 bash swe_nano_sc.sh grpo.num_prompts_per_step=2 policy.train_global_batch_size=8 grpo.max_num_steps=5
```

Must run from the path above (with `fsw`, not `fs1`) — the launcher figures out
container mounts from your current directory.

The two numbers on the command line (`num_prompts_per_step=2`,
`train_global_batch_size=8`) must always multiply out correctly:
`num_prompts_per_step × num_generations_per_prompt == train_global_batch_size`.
Right now that's `2 × 4 = 8`. If you change one, change the others to match.

## What you need

**Container** — most recent nightly build that includes Gym:
```
/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/enroot-images/nightly_gym_20260810.sqsh
```
Checked with `file`: it's a real local squashfs image (~77GB), not something pulled
from a registry when the job starts. Check
`/lustre/fsw/portfolios/llmservice/users/zhiyul/enroot-images/` for anything newer
before reusing this — that's where new nightly-gym builds land, and this doc won't
know about them automatically.

**Sandbox** (separate container, runs the code-execution side):
```
/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_reasoning/users/zhiyul/containers/nemo-skills-sandbox.sqsh
```

## The scripts

| File | Job |
|---|---|
| `ultra_launch.sh` | Builds the actual training command, sets up mounts, and calls `sbatch`. Copied byte-for-byte from the branch — nothing needed to change here except a small speed fix (see below). |
| `swe_nano_sc.sh` | What you actually run. Loads settings from `swe_nano.env`, then calls `ultra_launch.sh`. |
| `swe_nano.env` | All the settings: paths, container, how many nodes, walltime, priority queue. Edit this to change node count or container. |
| `ray.sub.nano-swe` | A copy of the repo's `ray.sub` with one line changed (walltime). See "gotchas" below for why we don't edit the original. |

Config files actually added to the tracked repo, under `examples/configs/ultra/`:
```
nano_swe_teacher_sc.yaml
nano_swe_teacher_qwen3mesh.yaml
nano_swe_teacher.yaml
swe_teacher.yaml
_nano_smoke_gb200.yaml.inc
```
No training code was touched. `main` already had everything the recipe needs — the
entry point, the TransferQueue plumbing, all of it. Only the recipe's config files
were missing.

## Watching a run

Don't use `slurm_bridge.sh` for anything, ever. Use the Slurm broker tools
(`slurm_my_jobs`, `slurm_query`) for status, and read log files straight off disk for
everything else.

The interesting per-step training numbers (loss, reward, etc.) are **not** in the main
driver log — they're in a separate worker log:
```
workspace/ray_logs/nano-swe-sc-tq-mainport/<job-id>-logs/ray/session_*/logs/worker-<hash>-01000000-<pid>.out
```
Find the right `<pid>` by searching the driver log for `SingleControllerActor pid=`,
then find the matching `worker-*<pid>.out` file under that job's `ray/` folder.

## Runs so far

| Job | Context length | Nodes | Priority | Result | W&B |
|---|---|---|---|---|---|
| 6292902 | 49152 | 6 | short | **All 5 steps finished.** TransferQueue confirmed active. Reward 0.25 twice — real partial solves. This is the proof the recipe works on `main`. | https://wandb.ai/nvidia/nano-swe-smoke/runs/19bpdqos |
| 6299959 | 196608 | 6 | short | 3 of 5 steps finished (reward 0.25 again), then the job got cancelled — not a timeout, ended at 1h21m of a 2h limit, and we don't know exactly why. No crashes or memory errors in the 3 steps it did run, but each step took much longer than at the shorter context length. | https://wandb.ai/nvidia/nano-swe-smoke/runs/wj5j91mv |

Both runs log to the same W&B project (`nvidia/nano-swe-smoke`) under the same run
name, so use the links above rather than the project's run list to find a specific one.

The longer-context (196608) run hasn't finished end-to-end yet — worth trying again
with more walltime (see below) so it has room to actually finish instead of getting cut
off.

**196608 is now the default context length** in `nano_swe_teacher.yaml` (it started as
49152 for a quick smoke test, then got bumped up). We checked that this isn't just a
config number nobody uses — the actual average tokens per sequence roughly doubled
between the two runs above, so the longer context is genuinely being exercised. To go
back to the faster, shorter setting for a quick check, add this to the command line:
```
policy.max_total_sequence_length=49152 policy.generation.vllm_cfg.max_model_len=49152
```

## Gotchas — why these specific changes were needed

**Two small additions to `swe_teacher.yaml`:**

1. **Three new fields under `policy.generation`**: `val_temperature`, `val_top_p`,
   `val_top_k`. A `main` commit landed after the original branch was cut that made
   these fields required. Fix: set them to mirror the existing training values
   (`${.temperature}`, `${.top_p}`, `${.top_k}`) — that's what every other recipe on
   `main` already does.
2. **`vllm_kwargs.moe_backend: triton`**. Without this, vLLM's default backend crashes
   while loading this model's weights (`shard_dim=0 is not a valid data dimension`).
   This exact fix already exists elsewhere in this repo for the same underlying
   problem (see `swe_teacher_cp16_gbs128_detect.yaml`), so it's not a new discovery —
   just applying a known fix to a new recipe.

**Don't set `env.nemo_gym.uv_venv_dir`.** It's tempting to point this at a shared
folder so all nodes see the same Python environment for the SWE agent, but don't —
we tried it and it caused two separate failures. The right answer is to leave it
alone: Gym's own default already puts the environment inside the (already
shared/mounted) Gym folder, so every node sees it automatically. No override needed.
This matches what the original branch's launcher does too.

**`ray.sub`'s built-in time limit wins over the one you pass on the command line.**
This bit us before: a job died at exactly 60 minutes even though we'd asked for 4
hours. `ray.sub` is shared by other jobs, so instead of editing it directly, we keep a
separate copy (`ray.sub.nano-swe`) with just the time limit changed, and point the
launcher at that copy. **If you change the walltime in `swe_nano.env`, update this
copied file to match, or your new walltime will be silently ignored.**

**Some config overrides need a `+` in front.** Normally you override a value like
`some.key=value`. But for a couple of settings that aren't already declared in the
config schema, you need `+some.key=value` instead, or it'll error out. Not obvious
until you hit it.

**The provenance-logging step was very slow.** Every launch used to run `git status`
on the whole repo to record what changed, which took 1-2+ minutes because this repo
has a huge number of tracked files on a slow network filesystem. Now capped at 15
seconds — if it's still slow after that, it just notes "timed out" and moves on
instead of hanging.

**Never use `slurm_bridge.sh`.** Full stop, no exceptions. If something needs Slurm
access from an environment that can't reach it directly, use the Slurm broker tools
for whatever they support, or just run the command yourself in a normal terminal.

**Short priority queue (`short`) is fast but has a 2-hour limit**, and possibly a
limit on how many nodes you can use at once (we tried to confirm the exact number and
couldn't get a clean answer — so we just avoided testing past 6 nodes). It worked fine
for the 49152-context run. For the 196608-context run, generation is the slow part and
it may not fit in 2 hours — use the normal queue with more time
(`WALLTIME=3:59:00`, and don't set `SLURM_QOS`) for a full long-context run.
