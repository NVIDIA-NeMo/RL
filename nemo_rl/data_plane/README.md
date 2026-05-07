# nemo_rl.data_plane

Stable boundary between NeMo-RL and any data-plane implementation
(currently `transfer_queue`; future: `nv-dataplane`). All call sites in
`nemo_rl/algorithms`, `nemo_rl/experience` and `nemo_rl/models` go through
`DataPlaneClient` — never `import transfer_queue` directly.

The full design lives in
[`research/data_plane_integration_plan.md`](../../research/data_plane_integration_plan.md).
This README is a quickstart for Stage 1 consumers.

## Install

`tensordict` and `TransferQueue==0.1.6` are base dependencies of
nemo-rl — `uv sync` (or `pip install -e .`) is enough; there is no
`[data-plane]` extra to remember. Worker venvs (built per-backend by
`nemo_rl.utils.venvs.create_local_venv` via bare `uv sync`) pick them up
automatically too, so the TQ adapter works on every worker class
(FSDP2, DTensor, mcore, automodel) without per-extra plumbing.

## Usage

```python
from tensordict import TensorDict
import torch

from nemo_rl.data_plane import build_data_plane_client

client = build_data_plane_client({
    "enabled": True,
    "impl": "transfer_queue",
    "backend": "simple",          # or "mooncake_cpu"
    "storage_capacity": 1_000_000,
    "num_storage_units": 2,
})

client.register_partition(
    partition_id="train",
    fields=["input_ids", "advantages"],
    num_samples=1024,
    consumer_tasks=["prev_lp", "ref_lp", "train"],
)

# Producer (rollout, ref policy, …) — sync put. Use ``async_kv_batch_put``
# only when composing with an existing event loop (e.g. async rollout
# actor); see ``research/data_plane_integration_plan.md`` §1.2.
client.kv_batch_put(
    keys=["uid-0", "uid-1"],
    partition_id="train",
    fields=TensorDict({"input_ids": torch.zeros(2, 128, dtype=torch.long)},
                      batch_size=[2]),
)

# Consumer — task-mediated discovery + tensor fetch.
meta = client.get_meta(
    partition_id="train",
    task_name="train",
    required_fields=["input_ids", "advantages"],
    batch_size=64,
)
batch = client.get_data(meta)        # TensorDict
```

## When `enabled=False`

The factory raises — there is intentionally no NoOp prod fallback.
Use the legacy `nemo_rl.algorithms.grpo.grpo_train` trainer for that
case (it never engages the data plane). The TQ-mediated trainer lives
at `nemo_rl.algorithms.grpo_sync.grpo_train_sync` and assumes
`enabled=True`.

`NoOpDataPlaneClient` exists in `adapters/noop.py` purely as a test
fixture for the ABC contract tests — production callers must not import
it.

## Hard rules

These are checked at the adapter; violating them is a TypeError, not a
warning.

* **No Python leaves on the bus** (P3). `kv_batch_put(fields=...)` must
  be a `TensorDict` of tensors. Use `tags=` for primitives, the Ray
  object store for arbitrary Python objects.
* **`select_fields` is required on read** (P2). `get_data` raises if
  neither `select_fields` nor `meta.fields` is set — silently fetching
  the full sample record (verl's footgun) is not allowed.

## What lands in later stages

* **Stage 2** — `codec.py` (`BatchedDataDict ↔ TensorDict`, jagged
  bridge `materialize(layout="padded")`).
* **Stage 3** — GRPO call sites wired through `DataPlaneClient`.
* **Stage 4** — per-DP-rank fetch entrypoint
  (`policy.train_from_dp_meta`).
* **Stage 5** — Mooncake CPU backend swap.

## Operational assumptions (Phase 1)

* One Ray cluster per experiment. The TQ controller is a globally named
  Ray actor; running two trainers in the same cluster will collide.
* Storage capacity sizing rule of thumb:
  `storage_capacity ≥ 2 × num_prompts × n_gens × max_seq_len ×
  bytes_per_token × num_active_fields`.
