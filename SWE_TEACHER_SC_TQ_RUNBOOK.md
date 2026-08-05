# Ultra 550B SWE-Teacher: legacy async vs SingleController + TransferQueue

Repro for the A/B between the legacy in-memory async path and the SingleController +
TransferQueue (SC+TQ) data plane, on 48× GB200. Both launchers wrap `ultra_launch.sh` and
share `swe.env`, so shape, model and data are identical by construction — only the entrypoint
and config differ.

## Run it

```bash
cd <repo-root>          # the launchers source secrets.sh themselves

DRY_RUN=0 bash swe_legacy.sh                                # legacy async
DRY_RUN=0 bash swe_sc.sh                                    # SC+TQ, inflight=2
DRY_RUN=0 bash swe_sc.sh async_rl.max_inflight_prompts=4    # SC+TQ, inflight=4
```

Drop `DRY_RUN=0` to print the resolved driver command and exit.

**Register the GPU-idle reaper exemption while PENDING or just after RUNNING** — the 550B load
sits at SM≈0 for ~30 min and SWE rollout leaves training GPUs idle for long stretches. Without
it the job is cancelled with a clean `exit_code=0` and no traceback:

```bash
scontrol update jobid=$JOB Comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"120","reason":"benchmarking","description":"550B load ~30min SM~0; SWE rollouts idle GPUs"}}'
```

## Reference runs — W&B `nemorl-dataplane-zhiyul`

| Run | Path | inflight | Command |
|---|---|---|---|
| [`rb88wuo0`](https://wandb.ai/nvidia/nemorl-dataplane-zhiyul/runs/rb88wuo0) | legacy async | n/a | `bash swe_legacy.sh` |
| [`0pta4j34`](https://wandb.ai/nvidia/nemorl-dataplane-zhiyul/runs/0pta4j34) | SC+TQ | 2 | `bash swe_sc.sh` |
| [`0e4e72g8`](https://wandb.ai/nvidia/nemorl-dataplane-zhiyul/runs/0e4e72g8) | SC+TQ | 4 | `bash swe_sc.sh async_rl.max_inflight_prompts=4` |

All three: `num_prompts_per_step=2` × `num_generations_per_prompt=16` = `train_global_batch_size=32`,
`max_buffered_rollouts=64`, `min_groups_for_streaming_train=2`.

> Every config with an `async_rl` block derives `max_inflight_prompts` from
> `${grpo.num_prompts_per_step}`, so no config can express `inflight != num_prompts_per_step` —
> the inflight=4 arm must come from a CLI override. `ultra_launch.sh` forwards positional
> arguments to Hydra verbatim, so appending it to any launcher works.

## What differs

Shared (from `swe.env`, not variables in the comparison): 48 nodes (32 train + 16 gen,
`SEGMENT_SIZE=16`), account `nemotron_sw_post`, partition `batch`, QOS `short`, walltime
`1:59:00`, same 550B checkpoint, `swe.jsonl`, container.

| | `swe_legacy.sh` | `swe_sc.sh` |
|---|---|---|
| Entrypoint | `run_grpo_nemo_gym.py` | `run_grpo_single_controller.py` |
| Config | `tiny_swe_teacher.yaml` | `tiny_swe_teacher_sc.yaml` |
| Async | `async_grpo.enabled` + in-memory `ReplayBuffer` | `SingleControllerActor` + TransferQueue |
| Data plane | none — `data_plane` block is a silent no-op | required; entrypoint raises without it |

`swe_legacy.sh` pins `grpo.val_period=0` and `checkpointing.enabled=false` because SC supports
neither — without that the two runs would not be comparable.

Enabling `data_plane` does **not** mean it is used — only `run_grpo_single_controller.py` (always)
and `run_grpo_nemo_gym.py` with `async_grpo.enabled=false` honour it. Verify at runtime rather than
from the config echo:

```bash
grep -aoE "PUT_DATA|KV_RETRIEVE_META" "$LD/ray-driver.log" | sort | uniq -c   # rollout commit / trainer consume
```

`docs/guides/nano-swe-transferqueue.md` has the per-entrypoint table with source-line evidence.

## Compare

These runs logged throughput as:

```
timing/train/valid_tokens_per_sec_per_gpu    global_valid_toks / step_time / all_gpus
timing/train/total_tokens_per_sec_per_gpu    total_num_tokens  / step_time / all_gpus
```

Prefer `total_` for A/B — `valid_` shrinks whenever the `mask_sample` filter drops sequences,
moving the metric for reasons unrelated to performance. Both are **end-to-end** (full step time,
all GPUs incl. generation), so on this generation-bound workload they are dominated by rollout.

`wandb_compare.py <run-id>…` plots these (defaulting to the three runs above). It addresses runs
by **id**, not display name — both SC arms share one `EXP_NAME`, so a name lookup cannot separate
them — and tries both key eras, since a pending change moves all rates to `performance/…` and
leaves `timing/train/` for durations.

`wandb_backfill_config.py` rebuilds a run's config from `provenance.txt` for SC runs launched
before the `log_hyperparams` fix (both SC runs above were backfilled; `rb88wuo0` was not).

## SC-specific settings

Why each SC config setting exists — and the batch invariant
`num_prompts_per_step × num_generations_per_prompt == train_global_batch_size`, which is why the
launchers pass those three as a unit — is documented with file:line evidence in
`docs/guides/nano-swe-transferqueue.md` ("Why each SingleController-specific setting is there").
The same settings apply at Ultra scale.

## Monitor

```bash
JOB=<jobid>; LD=$WORKSPACE_DIR/ray_logs/<EXP_NAME>/$JOB-logs
grep -aoE "train step [0-9]+|Watchdog caught|EngineDeadError" "$LD/ray-driver.log" | sort | uniq -c
```

SC per-step metrics are in the **actor's** log, not `ray-driver.log`:

```bash
grep -E "train step|step_metrics" "$LD"/ray/session_*/logs/worker-*.out
```

## Known limits

- **Rewards ≈ 0** on short runs — these validate the loop and its throughput, not model quality.
- **Sandbox startup failure**: if the nemo-skills sandbox `srun` dies (`command not found`,
  `/dev/null: No such file or directory` inside the container), `ray.sub` exits 1 at ~50 s.
  Usually node-local — resubmit or exclude the node set.
