# Regular Super RL: reviewed gold alignment

The reference is the `grpo_superv3_5_rlvr_v43_broad_falcon_r3-oci-hsg-20260905-r1`
script/YAML in pipeline commit `7e8d920115402f68d0f9047f23977246b4a88f46`.
These recipe changes require a new native smoke; they do not retroactively change
the source or configuration of a completed job. No training certification is
implied by static tests or scheduler acceptance.

## Context and policy serving

The earlier CMH adapter retained a 262144-token total/packing budget even after
reducing output to 102400. The regular recipe now uses **131072** for total
context and vLLM model length. Train/logprob packing budgets derive from that
single value and their respective microbatch sizes, avoiding stale copies.

The policy vLLM scheduler now allows **256 sequences** and **32768 batched
tokens**, matching the gold settings. This is an unbenchmarked throughput
candidate, not evidence of improved speed or memory fit. It does not change the
DeepSeek judge's separate scheduling or 8192-token response budget.

The policy and all agents retain **102400 cumulative assistant output tokens**,
reasoning enabled, and original prompts. At full output length only 28672 tokens
remain for input, template overhead and tool history. Validate real tokenized
trajectories before launch; do not silently truncate prompts, suppress reasoning,
or reduce the agreed output allowance to make them fit. Multi-turn history can
hit the context ceiling before exhausting the cumulative output allowance.

Implementation: `training_configs/super_rl/experiments/regular_s120_smoke.yaml`.
Regression coverage: `tests/unit/tools/test_regular_recipe_assets.py`.

## Output-format reward penalties

The earlier regular recipe disabled unwanted-token and malformed-think-tag
penalties. Both are now enabled, alongside the existing duplicated-reasoning
and empty-final penalties. The existing implementation and token IDs are
unchanged: unwanted `[2]`, think-open `12`, think-close `13`. Verify these IDs
against the actual tokenizer before using another model.

This deliberately changes the reward contract, not the source prompts or
reasoning mode. Check unwanted-token/malformed-think-tag rates and inspect
flagged multi-turn trajectories in the next smoke. If extraction or parsing
incorrectly flags valid output, fix that source rather than silently disabling
the requested penalty. Historical reward measurements remain unchanged and
are not measurements of the new contract.

## Explicit startup idle-reaper grace

Use the clean submission entrypoint with **`--bootstrap-grace-minutes 75`** for
this baseline. It puts the same structured `OccupiedIdleGPUsJobReaper` comment
as gold on the **outer sbatch request**, where it can be observed. Adding an
`#SBATCH` directive to a sourced `ray.sub` would not affect the outer job.

```bash
uv run --no-sync tools/super_rl/submit.py --bootstrap-grace-minutes 75 \
  -- --account=<account> --partition=<partition> --qos=<qos> \
  --nodes=64 --gpus-per-node=4 --segment=16 --time=02:00:00 <reviewed-launcher.sbatch>
```

This command is **scheduler test-only**; actual submission additionally requires
`--submit` before `--`. Pass reviewed workload environment names via `--env` as
needed. The grace is explicit, not applied to every site's jobs by default.
It supersedes a batch-file comment; an additional first-component CLI comment
is rejected rather than silently discarding it. Later heterogeneous service
components retain their own comments and do not acquire a 75-minute or
1440-minute exemption implicitly.

The grace does not fix stalled training, extend Slurm walltime, or override site
policy. Verify the stored outer job comment after a real submission. A scheduler
test accepting the JSON does not prove that the reaper service honored it.

## CP/EP and segment candidate

Gold's effective parallelism is TP4/CP4/EP16/PP1, with expert TP1. The previous
CMH adapter uses TP4/CP16/EP32/PP1. CP16 is not a GB300 hardware requirement.
Keep `regular_s120_smoke.yaml` as the CP16/EP32 topology control; use the small
inherited `regular_s120_cp4_ep16.yaml` for the gold-aligned **candidate**. It
changes only CP, EP and Ray segment size; it does not change the data, loss,
MTP, judge, reasoning mode, age 2 or GBS4096.

For our **64 learner GPUs**, not the full 256-GPU Ray allocation:

| Quantity | CP16/EP32 control | CP4/EP16 candidate |
| --- | ---: | ---: |
| Non-expert DP = 64 / (TP × CP × PP) | 1 | 4 |
| Expert DP = 64 / (expert TP × EP × PP) | 2 | 4 |
| CP communication group size | 16 | 4 |
| EP communication group size | 32 | 16 |
| Routed experts per rank (512 experts, expert TP1) | 16 | 32 |
| Context partition upper bound at 131072 total tokens | 8192 | 32768 |

These are two different rank-group constructions, not one product
`TP × CP × EP × DP`. The candidate has legal divisibility and smaller CP/EP
groups, but doubles experts per rank and increases the local context bound.
Compared with the **old 262144/CP16** configuration, its context partition
bound doubles from 16384 to 32768. Sequence parallelism, packing, activation
recomputation and Mamba behavior further affect real memory use. DP4 is not a
fourfold speedup guarantee. CP8 is a fallback to measure if CP4 lacks memory
headroom; it gives DP2 and a 16384-token context partition bound at 131072.

An additional CPU-only check evaluated the pure rank-generation definitions
from pinned Megatron-Core `f2f0f7bfd88fcb1243df55275988d6af52daea35`, without
initializing distributed workers. CP16/EP32, CP8/EP16 and CP4/EP16 all partitioned
the 64 ranks with the expected TP/CP/DP/EP group sizes and matching PP groups.
This checks rank arithmetic, not communication performance or native startup.

`cluster.segment_size` controls Ray's topology-aware learner selection. The
outer Slurm Ray allocation also needs matching `--segment`. Bigger segments
favor larger contiguous topology blocks but impose stricter scheduling
constraints. Our 16-node learner can fit one segment16 block, versus two
segment8 blocks; verify the actual rank-to-node/NVLink-domain map after startup.
Topology discovery can fall back, so setting a number is not placement proof.

On 2026-09-15, CMH accepted **both segment8 and segment16** in read-only
`sbatch --test-only` requests for 64 nodes, four GPUs/node, with the team's
account, partition and QoS, two hours, and the structured 75-minute comment. No GPU job was submitted. Acceptance and estimated start
times are neither reservations nor guarantees about runtime placement.
The earlier 20-node smoke shape is not divisible by 8 or 16; do not blindly
reuse it with these segment sizes. Judge allocation components remain separate.

Next native gate: fixed data and output budget, real worker/source identities,
rank placement, peak allocated/reserved memory, logprob/forward/backward time,
finite loss/gradient norms, router replay, update/refit and checkpoint/reload.
Compare all-route and long-trajectory behavior before claiming a faster or
stable replacement. Static tests do not exercise CUDA, NCCL or memory fit.

## What gold's environment differences actually mean

Gold invokes `uv run --verbose python ...` from the NeMo-RL project. `uv` selects
the project environment and may synchronize its dependencies as needed; it is
not a different training algorithm. This invocation does not use `--locked`
or `--frozen`, and UV network access is enabled. It may reuse an existing
environment; it does not necessarily download packages on every launch.
Our launcher selects the image's `/opt/nemo_rl_venv/bin/python` explicitly.
Actual driver and worker environments must be checked separately.

Gold mounts host code and whole Gym/Bridge workspaces at writable container
paths. That permits environment/build changes there and can shadow code baked
into the image. Our immutable source mounts and prebuilt helper overlay are
a deliberate alternative; do not make all source writable merely to imitate
gold. Writable run-owned cache/build paths remain necessary.

The inspected HSG checkout has uncommitted changes. Checking out its HEAD
alone therefore does not reproduce that working tree:

- `ray.sub`: the observed substantive additions are per-job and `latest-logs`
  symlinks. A shell flag reorder has the same flags. These are logging
  conveniences, not CP/EP or loss changes.
- `uv.lock`: the registry entry for `torch-memory-saver` changes from
  `0.0.9.post1` to `0.0.10b1`; the separate Git variant retains its pin.
  Both locks contain 551 package records, with 315 changed records, many
  involving dependency markers. This does **not** mean 315 package upgrades
  or prove which variant the ARM worker actually imports.
- The Bridge submodule is also dirty; identical base pins do not prove
  identical executed source.

Do not copy the dirty lock or logging edits wholesale. Capture actual package
versions/import paths under each worker interpreter, then commit only a
demonstrably necessary dependency or runtime change with a targeted test.

## DeepSeek versus gold's Qwen judge service

The service boundary is the same: Gym's reward resource calls a separate
OpenAI-compatible model service. The policy rollout engine is not the judge.
Our math/equivalence resources bind to `deepseek_v4_flash_judge_model` and the
explicit local `SUPER_RL_JUDGE_URL`, with reasoning enabled. Changing the judge
model still changes the reward evaluator; matching architecture does not make
Qwen and DeepSeek verdicts equivalent.

Gold uses independent replicas, load balancers and service readiness/lifecycle
management. Our reviewed mixed launcher uses one separate four-GPU DS node,
a direct endpoint, readiness gating and a health watchdog. These are similar
separation and lifecycle controls, **not equivalent high availability**. DP4
inside one engine/node is not four independent fault-tolerant replicas. A fatal
judge failure stops training rather than inventing a reward or silently using
a hosted provider. SciCode-only training did not validate the DS integration.

A multi-replica local DS pool with health-based routing is a separate reliability
change requiring a replica-loss test. It is not implemented by these recipe
commits. Do not copy gold's whole Qwen/GenRM/safety fleet or apply the policy's
32768 batched-token limit to the separately tuned DS service automatically.
