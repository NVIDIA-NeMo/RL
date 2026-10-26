# Sample Table

`async_rl.sample_table` logs one table per optimizer step on the
SingleController path, with one row per trained sample. Each row holds the
sample's ids, rewards and masks, and the means of its logprob and advantage
sequences over response tokens. The table goes to W&B, to NeMo Lens as OTel log
records, or both. Rows are keyed by `rollout_id`, so a table can be joined with
other per-rollout records, such as the Gym-results log proposed in
NVIDIA-NeMo/RL#4890.

## Configuration

```yaml
async_rl:
  sample_table:
    enabled: true
    name: samples          # logged as train/samples
logger:
  wandb_enabled: true      # destination: W&B
telemetry:                 # destination: NeMo Lens logs
  enabled: true
  logs_enabled: true
```

`SampleTableConfig` uses `extra="forbid"`, so a misspelled key fails at startup.
`validate_single_controller_config` also fails at startup in two cases:

- The run has neither destination enabled.
- W&B is on and a step's batch, `num_prompts_per_step * num_generations_per_prompt`,
  is above W&B's table limit of 200,000 rows (`WANDB_TABLE_MAX_ROWS`, which is
  `wandb.Table.MAX_ARTIFACT_ROWS`). Every row of a step is logged, so a table is
  never silently truncated. Lens has no row limit.

The table does not depend on `log_full_train_data` and does not read the
training-data dump.

## Data flow

```
advantage stage (controller or pool shard), per call
  summarize_chunk(tensors it already holds) -> AdvantageOutcome.sample_table
controller, per call      _sample_table_chunks.append(outcome.sample_table)
controller, step close    columns, rows = merge_table_chunks(chunks)
  if logger.wandb_enabled:    step_metrics[name] = Table(...)    built in a thread
  if Lens logs are on:        emit_table(columns, rows, step, "train/<name>")
```

- **W&B:** the table rides in `step_metrics`, the same path as the existing Gym
  full-result tables, so it is logged as `train/<name>` in the step's own W&B
  commit. `wandb.Table` validates every cell, so the controller builds it in
  `asyncio.to_thread`. Backends without a table type skip it.
- **NeMo Lens** (`nemo_rl/telemetry/tables.py`): `emit_table` queues the table
  and returns. A daemon thread turns each row into one OTel log record:
  - The `event_name` is the table name.
  - Its attributes are `rl.table`, `rl.step`, and `rl.row.<column>` for each
    non-null cell.
  - It carries the run's resource attributes (`nemo.run.id`).
  - It is emitted under the span that was active at the `emit_table` call.

  `shutdown_telemetry` drains the queue within half of its time budget, before
  the final flush. With no Lens `LoggerProvider`, emitting is a no-op.

## Table rows

There is one row per sample, in the order the advantage calls returned.

| Column | Meaning |
|---|---|
| `sample_id` | TransferQueue sample id |
| `group_id`, `rollout_id`, `weight_version` | The row's tags; `None` when absent |
| `input_lengths` | Tokens in the row |
| `response_tokens` | Positions with `token_mask == 1` within the row's length |
| `mean_<seq>` | Mean of each float sequence (`advantages`, `generation_logprobs`, and `prev_logprobs` / `teacher_logprobs` when computed) over those positions, skipping NaN and inf; `None` when nothing is left |
| `rewards`, `sample_loss_mask`, `pre_seq_error_sample_loss_mask` | One value per sample; NaN and inf become `None` |

With token capture on, the TransferQueue `sample_id` already equals Gym's
rollout id. With it off, joining needs a `rollout_id` tag on each row, which
#4890 proposes.

## Overhead

The numbers below were measured with `bench_sample_table.py` on the NeMo RL
2026-10-21 container, on a GB200 node, with 256-row chunks. The summary
reductions run vectorized on the tensors the advantage stage already holds.

| Step | Summaries (shards, all chunks) | Merge (controller) | W&B `Table(...)` (worker thread) | W&B `run.log` (step log) | Lens: step path / background |
|---|---|---|---|---|---|
| 2,048 rows x 8,192 tokens | 0.12 s | 0.001 s | 0.10 s | 0.75 s | ~0 / 0.03 s |
| 32,768 rows x 512 tokens | 1.77 s | 0.013 s | 1.54 s | 1.33 s | ~0 / 0.76 s |
| 200,000 rows x 64 tokens | 1.14 s | 0.04 s | 9.9 s | 4.3 s | ~0 / 4.6 s |

The summaries are spread across the advantage shards and run beside each
call's own work. Only the merge and W&B's `run.log` sit on the controller's step
path. `run.log` serializes the table as part of the step's normal W&B logging,
so W&B is the one destination whose cost grows with the batch: 4.3 s at 200,000
rows. Lens alone keeps the step path flat.

How the cost is kept low:

- **Summaries come from tensors the stage already holds.** Nothing re-reads the
  training-data dump. An earlier design re-parsed a 1 GB dump on the controller,
  which took about 11 s per step.
- **The reductions run on whole chunks.** A per-row loop had cost about 3 ms per
  row.
- **No files.** Summaries return in `AdvantageOutcome`. On Lustre, a per-chunk
  open/append/close cost about 15 ms.
- **Lens never touches the step path.**
- **W&B work is split.** The `Table` build runs in a worker thread. Only
  `run.log` runs at the step's log.
