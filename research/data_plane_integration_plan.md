# NeMo-RL Data Plane Integration Plan

**Owner:** zhiyul
**Date:** 2026-05-01
**Status:** Stage 1 ready to start — designed for parallel team execution
**Reference integrations (both share the same idea, different worker plumbing):**

Both verl and rl-arena converge on the same data-plane shape — **driver balances from metadata only (no tensor fetch); workers fetch their own slice from TQ direct (1-hop)**. They differ only in how the worker-side TQ I/O is wired:

| Source | Driver-side balance | Worker-side TQ I/O | Files |
|---|---|---|---|
| **verl** | `_balance_batch` reads `seq_len` from tags, runs Karmarkar-Karp, `batch.reorder([...])` permutes meta keys in place | `@tqbridge` decorator wraps the existing trainer worker; on entry calls `kv_batch_get(meta)`, on exit calls `kv_batch_put(output)`. Trainer doesn't know TQ exists. | `../verl/verl/trainer/main_ppo_sync.py:998-1022`, `../verl/verl/utils/transferqueue_utils.py:296-354` |
| **rl-arena** | `client.shard_for_dp(meta, dp_world_size) -> list[KVBatchMeta]` returns explicit per-rank metas using sort-by-seqlen + stride (the algorithm NeMo-RL's `BatchedDataDict.shard_by_batch_size(dynamic_batching_args=...)` already uses) | Each `TrainActor` is its own Ray actor with `self._client = DataPlaneClient()`; calls `client.kv_batch_get(keys=shard.keys, partition_id=shard.partition_id, ...)` directly. Explicit method on the worker. | `../rl-arena/arena/dataplane_client.py:275-314` (shard_for_dp), `../rl-arena/arena/workers.py:381-406` |

**Which one we follow for NeMo-RL.** verl's `tqbridge` decorator is a better fit for NeMo-RL because `Policy.train` already dispatches per-DP-rank via `worker_group.run_all_workers_sharded_data` — a decorator on the existing trainer is the smallest change. But the rl-arena shape is equally valid in principle and lives in the codebase as a working 1-hop reference; if the decorator approach hits friction we can fall back to the explicit `shard_for_dp` + `kv_batch_get` pattern without changing the data-plane semantics.

**Backend baseline.** rl-arena also serves as the throughput baseline for backend swap (SimpleStorage / Mooncake CPU / Mooncake GPU) and jagged-tensor transport validation — that's an orthogonal use we keep regardless of which worker-plumbing shape NeMo-RL adopts.
**Storage backend (Phase 1):** SimpleStorage only — Mooncake CPU/GPU RDMA out of scope until backend swap is exercised in Phase 5

---

## 1. Goals & Hard Constraints

| # | Requirement | How it shapes the design |
|---|---|---|
| G1 | Backend within TransferQueue must be swappable (Simple → Mooncake CPU → Mooncake GPU) | Backend selection lives in the TQ init layer — owned by TQ itself, not NeMo-RL. We expose a single `backend` config field. |
| G2 | The TQ implementation itself must be swappable (e.g., later replace with `nv-dataplane`) | Introduce a `DataPlaneClient` ABC inside `nemo_rl/data_plane/`. All call sites in NeMo-RL go through this interface, never `import transfer_queue` directly. |
| G3 | Phase 1: jagged in TQ, materialize to padded only at the model forward boundary | Bridge layer (`materialize(layout="padded")`) — keep existing trainers untouched. |
| G4 | Phase 2 (deferred): migrate trainers to consume jagged natively | Out of scope for now; track as future work. |
| G5 | Stage 1 must enable parallel team work | Stage 1 ships interfaces + factory + smoke test only — no algorithm changes. Teammates can start consuming the API the day Stage 1 lands. |

---

## 1.1 Design Principles

These constrain *how* we build the layers above, not *what* we build.

**P1 — Avoid worker-side caching whenever possible.** TQ is the source of truth. Building a worker-side cache to "amortize" over-fetches reintroduces three problems we don't have today: (a) cache invalidation when a writeback updates a field, (b) low hit rate when stages reshuffle samples across DP ranks (verl's `_balance_batch`, our `shard_keys_by_seqlen`), (c) memory cost on every worker (~100 MB+ at typical batch sizes). Fix the upstream over-fetch instead — see P2.

The exception is **read-only fields that are large, stable, and re-read every step** (e.g., `input_ids` / `position_ids` for repeated model forwards on the same samples). Cache only those, and only if profiling demands it. Default = no cache.

**P2 — Use `tqbridge` (transparent decorator) but always pass `select_fields`.** The decorator pattern is good — it hides the put/get plumbing, keeps worker functions clean, and is a familiar pattern from verl. The footgun is only that verl's current call sites set `KVBatchMeta.fields=None`, so the `select_fields` branch at `transferqueue_utils.py:262` never triggers and every call fetches the full sample record (~10x waste).

We adopt the decorator but **make `select_fields` a required argument**, populated either (a) by the caller setting `meta.fields = [...]` before invoking the decorated function, or (b) auto-derived via `inspect.signature(func).parameters` for kwargs-aligned signatures. Either way, the decorator never falls through to fetching all fields.

**Field-name alignment with native TQ.** Our `KVBatchMeta` mirrors `transfer_queue.metadata.KVBatchMeta` 1:1 — the attribute is `fields: list[str] | None`, not `fields_available`. This keeps the adapter a pure translator (no rename layer) and lets us reuse TQ's `select_fields` validation in `kv_batch_get_by_meta` (`interface.py:595-602`) without re-implementing it.

```python
# Required pattern (caller-provides):
meta.fields = ["input_ids", "position_ids"]
output = self.actor_wg.compute_log_prob(meta)   # decorator fetches only these

# Or kwargs-aligned (cleaner, deferred to Phase 2):
@tqbridge
def compute_log_prob(self, input_ids, position_ids):
    ...   # decorator reads signature, fetches exactly these
```

The decorator must **fail loudly** if `meta.fields is None` and no signature inference is configured. No silent full-fetch fallback.

**P3 — Structured data tensorizes; unstructured data goes out-of-band. No pickle on the bus.** Everything that crosses the `kv_batch_put` / `get_data` boundary must be a tensor at the adapter. This is the rule that makes G1 (single-config-flip backend swap) real — without it, swapping in Mooncake GPU is a multi-week migration, not a config flip.

**Why:** RDMA ships byte buffers. Mooncake GPU specifically requires *device-resident, contiguous, NIC-registered* buffers. Pickle-then-RDMA on the GPU path costs two extra PCIe traversals (H2D before MR registration, D2H on the receiver) plus CPU serialization on both ends — strictly worse than the CPU backend you were trying to upgrade *from*. The CPU backend (SimpleStorage / Mooncake CPU) silently absorbs pickle today, which means the moment a teammate adds a Python leaf "just for this one debug field," the GPU swap quietly becomes useless. Forbid it from day one.

**Three tiers** define where each kind of payload lives:

| Tier | Channel | Examples | Backend behavior |
|---|---|---|---|
| 1 | tensor on the bus | `input_ids`, `logprobs`, `advantages`, `total_reward`, `idx`, `image_grid_thw`, tokenized prompts/responses, `token_loss_mask`, `role_segments` (CSR) | RDMA'd as contiguous device/host buffer; no serialization |
| 2 | `tags` on controller | `prompt_uid`, `step_id`, `dp_rank` hint, `priority`, `task_name` (JSON-serializable primitives) | Lives in TQ controller's tag table; never on storage bus, never RDMA'd |
| 3 | out-of-band, indexed by `idx` | raw `content` strings, `extra_env_info` with mixed types, debug payloads, multi-turn env state, stop-string lists | Ray object store keyed by `idx`; the data plane stores only the `idx` tensor |

**`message_log` migration (the big one).** Current GRPO repeatedly indexes `message_log` for flattening, mask construction, prompt-only extraction, and logging (grpo.py:1444, 1659, 1685, 2048). It mixes load-bearing tensors (per-message `token_ids`, `generation_logprobs`) with non-tensor metadata (`role`, `content` string, optional multimodal fields). We don't punt this — we split it explicitly:

| Sub-field of `message_log[i][j]` | Where it lives | How it's reconstructed |
|---|---|---|
| `token_ids` (per-message) | Tier 1 — concatenated `input_ids` jagged tensor + `role_segments` CSR `(start, end, role_enum)` | Bridge `materialize()` reverses CSR → list of per-message slices |
| `token_loss_mask` | Tier 1 — `token_mask` jagged tensor (already flat) | direct |
| `generation_logprobs` | Tier 1 — `generation_logprobs` jagged tensor | direct |
| `role` (string `"user"`/`"assistant"`/`"system"`) | encoded as `int8` enum inside `role_segments` CSR; vocab shipped via `register_partition(enums=...)` | `StringEnum.decode` |
| `content` (raw text) | Tier 3 — Ray object store, key = `f"content:{idx}"` | Fetched by driver only when needed for logging / `_extract_prompt_only_messages` |
| `multimodal_dict` (e.g. `pixel_values`, `image_grid_thw`) | Tier 1 — declared up-front in `register_partition(fields=...)` superset (R4) | direct |
| `extra_env_info` | Tier 3 — Ray object store, key = `f"env:{idx}"` | Driver-only |

Logging paths (grpo.py:1728, 2053) and `_extract_prompt_only_messages` (grpo.py:1075) become driver-side helpers that fetch Tier 3 strings on demand — they don't run inside DP-sharded workers, so the round-trip is cheap and bounded.

**Tensorizing structured non-tensor data:** structured Python data has clean tensor encodings — use them at the producer:

| Source shape | Tensor encoding |
|---|---|
| `bool` / `int` / `float` | scalar tensor |
| Short fixed-vocab string (`"train"`, `"math"`, env name) | int enum tensor + vocab held by controller (shipped once at `register_partition`, not per sample) |
| Long tokenizable string | int64 token tensor (already what we do for prompts/responses) |
| Raw bytes (image/audio) | `uint8` tensor + length scalar |
| `list[primitive]` | 1D tensor + length |
| `list[list[primitive]]` (variable-length) | CSR: `(flat_values, offsets)` — two tensors |
| `dict` with fixed keys | one tensor per field, declared in `FIELD_SCHEMA` |

Helpers live in `data_plane/codec.py` (Stage 2):

```python
def to_csr(nested: list[list[int]]) -> tuple[Tensor, Tensor]: ...    # variable-length lists
def from_csr(flat: Tensor, offsets: Tensor) -> list[list[int]]: ...

class StringEnum:
    """Producer: str → int. Consumer: int → str. Vocab registered with controller, not per-sample."""
```

`register_partition` grows one optional kwarg to ship vocabs once:

```python
client.register_partition(
    partition_id="train",
    fields=[...],
    num_samples=N,
    consumer_tasks=[...],
    enums={"task_name": ["math", "code", "reasoning"]},   # NEW — controller-side vocab
)
```

**Adapter enforcement (mandatory, not advisory):**

```python
def _to_wire(self, td: TensorDict) -> TensorDict:
    bad = [k for k, v in td.items(include_nested=True, leaves_only=True)
           if not isinstance(v, torch.Tensor)]
    if bad:
        raise TypeError(
            f"kv_batch_put received non-tensor leaves: {bad}. "
            f"Tensorize via codec helpers, use `tags=` for primitives, "
            f"or use Ray object store for arbitrary Python objects."
        )
    td = td.detach().contiguous()
    return td.cpu() if self._wire_device == "cpu" else td
```

No silent pickle fallback — consistent with P2's "fail loudly" stance on `select_fields`. The ABC contract test (`test_interface.py`) must include a "Python leaf rejected" case so any future adapter inherits the same discipline.

**What this affects in later stages:**
- Stage 2 (codec): adds `to_csr`/`from_csr` and `StringEnum` helpers; `FIELD_SCHEMA` table includes an `encoding` column for variable-length / enum fields.
- Stage 3 (GRPO integration): producers (rollout, ref policy) tensorize at write time; no Python leaves leak in.
- Stage 5 (backend swap): swap is a config flip *because of this rule*, not in spite of it. Audit gate: grep adapter for `pickle` / non-tensor branches before declaring G1 verified.

**P4 — Jagged on the bus, padded at the trainer boundary (Phase 1).** Every Tier-1 variable-length field is stored as a `torch.nested.nested_tensor` ("NestedTensor") inside the TensorDict that crosses `kv_batch_put` / `kv_batch_get`. This is the only way to ship variable-length data without a global pad budget per partition (a 1-of-N very long sample would otherwise force every row to that length). Verl already uses this pattern via `TQNestedTensor` — copy it.

Bridge contract:

```python
def materialize(td: TensorDict, layout: Literal["padded", "jagged"] = "padded",
                pad_value_dict: dict[str, int | float] | None = None) -> BatchedDataDict:
    """Phase 1 default: layout='padded' → for each NestedTensor field, call
    nt.to_padded_tensor(pad_value) to produce a regular (B, T) dense tensor.
    Trainers (policy.train, policy.get_logprobs, advantage_estimator) consume
    BatchedDataDict exactly as today — no signature changes.

    Phase 2 (deferred): layout='jagged' returns NestedTensors directly; trainers
    migrate worker-by-worker behind the same flag."""
```

This decouples the wire format (jagged, fixed) from the trainer format (padded today, jagged later). Phase 2 is then a pure consumer-side migration with no producer changes — exactly the alignment the user called out.

**No `requires_grad` on the bus.** NestedTensors with `requires_grad=True` are illegal here; the codec calls `.detach().contiguous()` before put. RDMA backends register the buffer; autograd hooks would be silently dropped.

---

## 2. Architecture Overview

Three layers, top to bottom:

```
┌─────────────────────────────────────────────────────────────────┐
│  GRPO / PPO / SFT pipelines (algorithms/grpo.py, …)             │
│  Use BatchedDataDict like today; call dp_client.batch_put/get   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│  nemo_rl/data_plane/  ← NEW PACKAGE  (Stage 1)                  │
│  ┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐  │
│  │ interfaces.py   │ │ codec.py         │ │ packing.py      │  │
│  │  DataPlaneClient│ │  TensorDict ↔    │ │ KVBatchMeta →   │  │
│  │  KVBatchMeta    │ │  BatchedDataDict │ │  microbatch plan│  │
│  └─────────────────┘ └──────────────────┘ └─────────────────┘  │
│  ┌─────────────────┐ ┌──────────────────┐                       │
│  │ factory.py      │ │ adapters/        │                       │
│  │  build_client() │ │  transfer_queue.py (Stage 1)             │
│  │                 │ │  ray_object.py     (Stage 1, dev/test)   │
│  └─────────────────┘ └──────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│  TransferQueue pip package (transfer_queue==0.1.5) — UNMODIFIED │
│  Backend = SimpleStorage | MooncakeStore (G1)                   │
└─────────────────────────────────────────────────────────────────┘
```

**Key invariant:** Nothing in `nemo_rl/algorithms/`, `nemo_rl/experience/`, or `nemo_rl/models/` imports `transfer_queue` directly. They go through `nemo_rl.data_plane`.

---

## 3. Stages

### Stage 1 — Foundation (parallel-enabling)

**Goal:** Land the interface + factory + simple TQ adapter + smoke test. No algorithm changes. Teammates can start writing against the API immediately.

**Scope:**

```
nemo_rl/data_plane/
├── __init__.py              # public re-exports
├── interfaces.py            # DataPlaneClient ABC, KVBatchMeta, DataPlaneConfig
├── factory.py               # build_data_plane_client(config) → DataPlaneClient
├── adapters/
│   ├── __init__.py
│   ├── transfer_queue.py    # TQDataPlaneClient — wraps transfer_queue.get_client()
│   └── noop.py              # NoOpDataPlaneClient — when enabled=False (passthrough)
├── codec.py                 # placeholder; implemented in Stage 2
└── tests/
    ├── test_interface.py    # ABC contract test (must be implemented by all adapters)
    ├── test_smoke_tq.py     # init + put + get + clear, single field, single sample
    └── test_smoke_multinode.py # 2-node SimpleStorage smoke (Slurm)
```

**Interface (commit this first, freeze for Stage 2 consumers):**

```python
# nemo_rl/data_plane/interfaces.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypedDict
from tensordict import TensorDict

class DataPlaneConfig(TypedDict):
    enabled: bool                                  # default False — gate
    impl: Literal["transfer_queue", "noop"]        # which adapter
    backend: Literal["simple", "mooncake_cpu"]     # backend within TQ
    controller_address: NotRequired[str]
    storage_capacity: NotRequired[int]             # max samples in flight
    num_storage_units: NotRequired[int]
    get_meta_poll_interval_s: NotRequired[float]   # default 0.5
    ack_timeout_ms: NotRequired[int]               # default 5000

@dataclass
class KVBatchMeta:
    """1:1 mirror of transfer_queue.metadata.KVBatchMeta.

    Attribute names match TQ exactly so the adapter does no renaming and
    the `select_fields` validation in TQ's kv_batch_get_by_meta works
    against our object unmodified.
    """
    partition_id: str
    task_name: str | None             # None for direct kv_batch_get/put by keys
    keys: list[str]
    fields: list[str] | None = None   # field names available for these keys
    sequence_lengths: list[int] | None = None      # populated by controller from input_lengths tag
    extra_info: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.keys)

class DataPlaneClient(ABC):
    """Stable boundary between NeMo-RL and any data-plane impl.
    All call sites in algorithms/experience/models go through this.

    Two API groups:
      (A) task-mediated: register_partition / get_meta / get_data /
          check_consumption_status — used by stages that wait for
          upstream production via the per-task consumer counter.
      (B) direct-by-key: kv_batch_put / kv_batch_get / kv_clear — used by
          stages that already know the exact uids (e.g. driver-side
          fan-out to DP ranks). Argument order matches transfer_queue
          1:1 so the adapter is a thin pass-through.
    """

    # ── (A) task-mediated ───────────────────────────────────────────

    @abstractmethod
    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
        enums: dict[str, list[str]] | None = None,   # P3 vocabs (e.g. role)
    ) -> None: ...

    @abstractmethod
    def get_meta(
        self,
        partition_id: str,
        task_name: str,
        required_fields: list[str],
        batch_size: int,
        dp_rank: int | None = None,
        blocking: bool = True,
        timeout_s: float = 60.0,
    ) -> KVBatchMeta: ...

    @abstractmethod
    def get_data(
        self,
        meta: KVBatchMeta,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        """Convenience wrapper around kv_batch_get; resolves select_fields
        from meta.fields when None (P2: must not silently fall through to
        all-fields)."""

    @abstractmethod
    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool: ...

    # ── (B) direct-by-key (TQ-aligned signatures) ──────────────────

    @abstractmethod
    async def kv_batch_put(
        self,
        keys: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> KVBatchMeta:
        """Producer entrypoint. Writing a field automatically flips its
        production_status bit in the TQ controller — this IS the natural
        'stage finished for these keys' signal (see Stage-completion
        design below). Returns the meta that downstream consumers can
        use for direct kv_batch_get."""

    @abstractmethod
    def kv_batch_get(
        self,
        keys: list[str],
        partition_id: str,
        select_fields: list[str] | None = None,
    ) -> TensorDict:
        """Direct fetch by uids. Used by per-DP-rank slice fetches in
        Stage 4. Does NOT advance any per-task consumption counter — that
        only happens via get_meta(mode='fetch')."""

    @abstractmethod
    def kv_clear(
        self,
        keys: list[str] | None,
        partition_id: str,
    ) -> None:
        """keys=None clears the partition's full key set."""

    # ── (C) lifecycle ──────────────────────────────────────────────

    @abstractmethod
    def close(self) -> None: ...
```

**Stage-completion signal — design (load-bearing, freeze with the ABC).**

The `mark_consumed` method we had earlier was misleading: in TQ it is *not* an authoritative post-compute ack. The controller advances the per-task consumption counter **inside `get_metadata(mode="fetch")`** (`controller.py:1352`) — at *fetch* time, not compute time. A worker that fetches and then crashes still leaves its keys marked consumed.

So we use the only signal TQ actually provides authoritatively: **field production**. When a stage calls `kv_batch_put(keys, partition_id, fields={'<output>': ...})`, the controller flips `production_status[sample, output_field] = 1` (`controller.py:503-555`). Downstream consumers waiting on `<output>` only see those samples once they're produced. Field-presence *is* the "stage X done" signal — no separate flag, no separate wire op.

| Question | Answer for Phase 1 |
|---|---|
| **Q1. Do we need an internal "stage done" flag for fault tolerance?** | **No, not in Phase 1.** Field-presence is sufficient for the happy path. Worker crashes are handled by step-level checkpoint restart (the standard NeMo-RL recovery model) — partial-step recovery is out of scope. We don't add a flag we won't use. |
| **Q2. How would we design one when we do need it?** | A reserved `<task>_done: bool` field per consumer task, written by the worker as the *last* `kv_batch_put` of its compute. Consumers wait on `<task>_done` instead of (or in addition to) the payload field. This makes "compute crashed mid-put" detectable: payload field flipped to 1, `<task>_done` not flipped. Recovery uses TQ's `force_fetch` mode (`controller.py:1357`) to re-issue those keys. **Defer to Phase 2** — only build it if/when we want partial-step recovery. |
| **Q3. What about `mark_consumed` on the ABC?** | Drop it from the public ABC. It was only a client-side hint in rl-arena (`dataplane_client.py:240-253`); verl doesn't even call it. The authoritative consumption advance happens in `get_meta(mode='fetch')`. Removes a subtle correctness trap. |
| **Q4. How does the driver know "all samples for stage X are done"?** | `check_consumption_status(partition_id, [task_name])` already does this — it queries the per-task consumption tensor on the controller. We keep this method for the clear-safety check before `kv_clear`. |

**Field-name flexibility — design.** Field names are free-form strings the producer chooses, but we pin them in one place (`schema.py FIELD_SCHEMA`) so:

1. `register_partition(fields=...)` enumerates the superset once per partition (R4 — multimodal-tolerant).
2. The decorator's `select_fields` enforcement (P2) checks against this registry; misspelled field names fail loudly at the worker site, not silently at fetch.
3. Schema additions are a pure-Python edit — no ABC change. Stage-2 codec adds one row to the table per new field, with `(dtype, layout, encoding)`.

This gives us "flexible field names" without losing type safety: producers can add fields without touching the ABC, but every field has exactly one declared encoding. Compare to verl's footgun (`tqbridge` accepts `meta.fields=None` and silently fetches everything) — our decorator never falls through.

**Factory (commit second):**

```python
# nemo_rl/data_plane/factory.py
def build_data_plane_client(cfg: DataPlaneConfig) -> DataPlaneClient:
    if not cfg.get("enabled", False):
        return NoOpDataPlaneClient()
    if cfg["impl"] == "transfer_queue":
        from .adapters.transfer_queue import TQDataPlaneClient
        return TQDataPlaneClient(cfg)
    raise ValueError(f"unknown data_plane impl: {cfg['impl']}")
```

**TQ adapter (commit third — copy/adapt `rl-arena/arena/dataplane_client.py` and `backends.py`):**

The adapter is a *thin* shell:
- `__init__` calls `init_tq(backend=cfg["backend"], ...)` (lifted from `rl-arena/arena/backends.py`)
- Each method translates `KVBatchMeta` ↔ TQ's `BatchMeta` and forwards
- No business logic lives here

**MasterConfig wiring:**

```python
# nemo_rl/algorithms/grpo.py
class MasterConfig(TypedDict):
    policy: PolicyConfig
    loss_fn: ClippedPGLossConfig
    env: dict[str, Any]
    data: DataConfig
    grpo: GRPOConfig
    logger: GRPOLoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    data_plane: NotRequired[DataPlaneConfig]   # NEW — feature-gated, default off
```

**Smoke test (acceptance for Stage 1):**

`tests/test_smoke_tq.py` runs on a single Slurm node:
1. `client = build_data_plane_client({"enabled": True, "impl": "transfer_queue", "backend": "simple", ...})`
2. `client.register_partition("smoke", ["x"], num_samples=4, consumer_tasks=["read"])`
3. `await client.kv_batch_put(keys=["a","b","c","d"], partition_id="smoke", fields=TensorDict({"x": torch.arange(4)}, batch_size=[4]))`
4. `meta = client.get_meta("smoke", "read", ["x"], batch_size=4)`   # advances "read" consumption
5. `data = client.get_data(meta)`
6. `assert torch.equal(data["x"], torch.arange(4))`
7. `assert client.check_consumption_status("smoke", ["read"])`
8. `client.kv_clear(keys=None, partition_id="smoke"); client.close()`

Argument order matches `transfer_queue.kv_batch_put(keys, partition_id, fields, tags)` (`interface.py:467`) — the adapter must not reorder, since the next adapter in line (`nv-dataplane`) will follow the same convention.

**`test_smoke_multinode.py`** — same as above but launched via `RL/ray.sub` over 2 nodes, exactly the way `rl-arena/launch/run_arena.sh` already does. Verifies controller-actor placement and ZMQ across hosts.

**Pip dependency** — add `transfer_queue==0.1.5` to `pyproject.toml` as an optional extra (matches the wheel currently published; bumped only when we cut a new TQ release):

```toml
[project.optional-dependencies]
data-plane = ["transfer_queue==0.1.5"]
```

Same `try/except ImportError` pattern verl uses (`verl/utils/transferqueue_utils.py:35-57`) so NeMo-RL still imports cleanly without TQ installed; failure deferred to factory call when `enabled=True`.

**Setuptools packaging** — current `RL/pyproject.toml` declares `[tool.setuptools] packages = ["nemo_rl"]`, which does NOT pull in subpackages by default. Switch to `find` so `nemo_rl/data_plane/` is included automatically:

```toml
[tool.setuptools.packages.find]
include = ["nemo_rl*"]
```

Otherwise installs from sdist would silently drop the new package and the smoke test would fail with `ImportError: nemo_rl.data_plane`. Verify with `python -c "import nemo_rl.data_plane"` after `pip install -e .` in the Stage 1 PR.

**Stage 1 deliverables checklist:**
- [ ] `nemo_rl/data_plane/{interfaces,factory,adapters/transfer_queue,adapters/noop}.py`
- [ ] `data_plane` optional extra in `pyproject.toml`
- [ ] `data_plane: NotRequired[DataPlaneConfig]` added to `MasterConfig`
- [ ] Single-node smoke test green
- [ ] 2-node Slurm smoke test green
- [ ] Doc: `nemo_rl/data_plane/README.md` with usage example

**Parallel work this unblocks:**
- Teammate A: Phase 2 codec (TensorDict ↔ BatchedDataDict) using the locked `DataPlaneClient` interface
- Teammate B: GRPO Stage 3 integration — can write the put/get call sites against the mocked `NoOpDataPlaneClient` first, swap to real later
- Teammate C: Mooncake CPU backend wiring inside `adapters/transfer_queue.py` (just adds a config branch)

---

### Stage 2 — Schema & Codec (NeMo-RL ↔ TQ wire types)

**Goal:** Convert `BatchedDataDict[DatumSpec]` ↔ `TensorDict` with a stable, declared field schema. Build the jagged-aware materialize() helper so Phase 1 algorithms keep using padded tensors.

**Scope:**

```
nemo_rl/data_plane/
├── schema.py    # FIELD_SCHEMA — names, dtypes, per-sample shapes, layout (jagged/scalar/multimodal)
├── codec.py     # batched_dict_to_tensordict / tensordict_to_batched_dict / materialize
```

**`FIELD_SCHEMA` (mirrors `rl-arena/arena/schema.py`):**

| Field | Dtype | Per-sample shape | Layout | NeMo-RL source |
|---|---|---|---|---|
| `input_ids` | int64 | `[T_full]` | jagged | flatten of message_log |
| `input_lengths` | int32 | `[]` | scalar | sum of token_ids |
| `output_ids` | int64 | `[T_resp]` | jagged | post-rollout response slice |
| `generation_logprobs` | float32 | `[T_full]` | jagged | message_log entry |
| `prev_logprobs` | float32 | `[T_full]` | jagged | policy.get_logprobs |
| `reference_policy_logprobs` | float32 | `[T_full]` | jagged | ref policy forward |
| `advantages` | float32 | `[T_full]` | jagged | broadcast scalar group adv |
| `token_mask` | bool | `[T_full]` | jagged | from message_log token_loss_mask |
| `sample_mask` | float32 | `[]` | scalar | loss_multiplier |
| `total_reward` | float32 | `[]` | scalar | env step |
| `idx` | int64 | `[]` | scalar | DatumSpec.idx (≈ verl uid) |

**`materialize()` — the Phase 1 bridge:**

```python
def materialize(
    td: TensorDict,
    layout: Literal["padded", "packed", "jagged"] = "padded",
    pad_value_dict: dict[str, int | float] | None = None,
) -> BatchedDataDict:
    """Phase 1: jagged TQ → padded BatchedDataDict so existing trainers don't change.
    Phase 2: trainers call this with layout='jagged' or 'packed' and bypass densify."""
```

**Key invariant:** `batched_message_log_to_flat_message()` (the existing NeMo-RL flatten) becomes the reference implementation that `materialize(layout="padded")` must match byte-for-byte. Stage 2 includes a parity test.

---

### Stage 3 — GRPO Lifecycle Integration

**Goal:** Wire all 6 GRPO stages from the design through `DataPlaneClient`. Default off; enabled when `master_config["data_plane"]["enabled"] = True`.

**Stages (ordered to match the *actual* GRPO loop in `algorithms/grpo.py:1700-1816`):**

| # | Stage | Producer | Consumer waits on | TQ ops |
|---|---|---|---|---|
| 0 | register | driver | — | `register_partition(fields=SUPERSET, num_samples=N, consumer_tasks=["prev_lp","ref_lp","train"])` |
| 1 | generation + reward | rollout workers (vLLM/SGLang); reward folded in (already computed by `run_multi_turn_rollout`) | — | put `input_ids, output_ids, generation_logprobs, input_lengths, token_mask, sample_mask, total_reward, idx, role_segments` |
| 2 | prev_logprobs (`policy.get_logprobs`) | policy workers (DP-sharded) | field `input_ids` ready | put `prev_logprobs` |
| 3 | reference_policy_logprobs (`policy.get_reference_policy_logprobs`) | ref-policy workers (DP-sharded) | field `input_ids` ready | put `reference_policy_logprobs` |
| 4 | seq-logprob-error mask (`compute_and_apply_seq_logprob_error_masking`) | **driver (central)** | fields `prev_logprobs, generation_logprobs` ready | put updated `token_mask, sample_mask` |
| 5 | advantage (`adv_estimator.compute_advantage`) | **driver (central)** | fields `prev_logprobs, reference_policy_logprobs?, token_mask, sample_mask, total_reward` ready | put `advantages` |
| 6 | policy update (`policy.train`) | train workers (DP-sharded) | fields `input_ids, advantages, token_mask, sample_mask, prev_logprobs, reference_policy_logprobs?` ready | (no put; loss + optimizer step) |
| 7 | clear | driver | `check_consumption_status(["prev_lp","ref_lp","train"])` ⇒ True | `kv_clear(keys=None, partition_id=...)` |

**Why this order matters (correction from earlier draft):**

1. `compute_and_apply_seq_logprob_error_masking` (`grpo.py:1768`) consumes `prev_logprobs` and `generation_logprobs` to *mutate* `token_mask` and `sample_mask` *before* advantage computation. Skipping this and computing advantage first changes the batch.
2. `adv_estimator.compute_advantage` takes `logprobs_policy` and `logprobs_reference` for the KL-in-reward branch (`advantage_estimator.py:204-214`). Advantage cannot precede them.
3. `policy.train` reads `prev_logprobs` and `reference_policy_logprobs` for the importance ratio and KL penalty (loss function in `algorithms/loss_functions.py`). They must be in the partition before the train stage starts.

**Stages 4 and 5 run centrally on the driver, not on DP-sharded workers.** Matches verl (`main_ppo_sync.py:1135-1198`). Compute is cheap (no model forward). Driver does `kv_batch_get(keys=batch.keys, partition_id=...)` for the full batch, computes, `kv_batch_put` results back.

**Stage 6 (policy update) sharding — uses the new presharded entrypoint (Stage 4).** Driver calls `policy.train_from_dp_meta(meta)` which runs `shard_keys_by_seqlen` (sort-by-seqlen + stride, matching rl-arena's `shard_for_dp` and NeMo-RL's `dynamic_batching_args` branch) over `meta.keys + meta.sequence_lengths` and dispatches per-rank `KVBatchMeta` slices. Each DP worker calls `kv_batch_get(keys=mine)` → constructs its local `BatchedDataDict` → runs the existing per-rank microbatch / optimizer step (factored out as `_train_one_shard`). The internal `shard_by_batch_size` step from `policy.train` is **bypassed** in this entrypoint; the per-rank slice is already balanced. Same applies to Stages 2 and 3 via `get_logprobs_from_dp_meta`. See Stage 4 for the full design, hop accounting, and TP/CP/PP guidance.

**Driver fetches `message_log` Tier-3 fields (raw `content`, `extra_env_info`) only for logging paths** (`grpo.py:1728, 2053`) and `_extract_prompt_only_messages` (`grpo.py:1075`). DP-sharded workers never see them.

**Consumer-task naming.** `consumer_tasks=["prev_lp", "ref_lp", "train"]` — three tasks because three stages each independently advance the per-task consumption counter when they call `get_meta(mode="fetch")`. The driver-only stages (mask correction, advantage) don't get their own task name; they fetch via direct-by-key API which doesn't advance any counter.

**Where the changes land:**
- `algorithms/grpo.py` — orchestration; conditional branch on `data_plane.enabled`. Dynamic-sampling cache stays in driver memory (R11); the TQ seed put happens at the `is_batch_complete` boundary.
- `algorithms/advantage_estimator.py` — driver-side `kv_batch_get` for inputs, `kv_batch_put` for `advantages`; signature unchanged. (Driver-only stage; small batch, low compute → 2-hop is fine here.)
- `models/policy/lm_policy.py` + `models/policy/policy_worker.py` (+ `dtensor_policy_worker.py`) — **add `train_from_dp_meta` / `train_presharded` and `get_logprobs_from_dp_meta` / `get_logprobs_presharded`**, plus the `_train_one_shard` / `_get_logprobs_one_shard` factor-out so both the legacy and presharded paths share the inner per-rank step. Each DP worker grows a `_dp_client` field. This is the bulk of Stage 4 work and it lands in Phase 1, not deferred.
- `experience/rollouts.py` — **unchanged in Phase 1.** Rollout workers still return `BatchedDataDict` to the driver to keep the dynamic-sampling cache path intact. Phase 2 moves the rollout writeback into TQ once dynamic sampling is reworked.

**Backwards compatibility:** if `data_plane.enabled=False`, code path is unchanged from today. The TQ branch is feature-gated everywhere.

---

### Stage 4 — Per-rank fetch entrypoint (mandatory in Phase 1; smaller than I first claimed)

**Goal:** Match the 1-hop pattern that verl and rl-arena already use (TQ storage → DP worker direct, no tensor data through the driver). Add a presharded entrypoint on `Policy` so DP workers fetch their own slice.

**Reference: both verl and rl-arena follow the same 1-hop pattern with different surface plumbing.** Either is a valid template; we pick verl's decorator for NeMo-RL because it composes cleanly with `worker_group`.

**verl's path (~50 LOC of orchestration, decorator-based):**

1. **`_balance_batch`** (`main_ppo_sync.py:998-1022`): driver reads `seq_len` *from tags* (no tensor data!), runs `get_seqlen_balanced_partitions` (Karmarkar-Karp), then `batch.reorder([...])` permutes the keys list in the `KVBatchMeta` in-place.
2. **`actor_rollout_wg.update_actor(batch)`** (`main_ppo_sync.py:1237`): the worker group ships the *meta* (not data) and its dispatch mechanism slices the keys list evenly across DP ranks. Because the keys are pre-permuted into balanced groups, each rank's slice is automatically balanced.
3. **`tqbridge` decorator on the worker** (`transferqueue_utils.py:296-354, 111-126`): wraps the worker function so that on entry it calls `tq_client.get_data(meta)` for that rank's slice (kv_batch_get), and on exit calls `tq_client.put` (kv_batch_put). The wrapped worker function is the *existing* training step — no special entrypoint, just a decorator.

The cleverness is that the worker group's dispatch handles slicing for free, and the decorator handles the TQ I/O for free. The trainer worker doesn't know TQ exists. This is 1-hop because the decorator runs *inside* the worker process — `kv_batch_get` reads TQ storage directly into worker memory.

**rl-arena's path (same idea, explicit-method surface):**

1. **`driver_client.shard_for_dp(meta, dp_world_size)` → `list[KVBatchMeta]`** (`rl-arena/arena/dataplane_client.py:275-314`): driver-side, control plane only, returns one `KVBatchMeta` per rank using sort-by-seqlen + stride. Equivalent to verl's `_balance_batch` + dispatch slicing combined into one call, and the same algorithm NeMo-RL's `BatchedDataDict.shard_by_batch_size(dynamic_batching_args=...)` already applies. Single algorithm, no strategy parameter.
2. **Driver dispatches per-rank: `train_actors[r].update.remote(shards[r])`** (`rl-arena/arena/pipeline.py:158-185`): each train actor is its own Ray actor and receives its `KVBatchMeta` slice directly. No worker_group involved.
3. **Worker calls `self._client.kv_batch_get(keys=shard.keys, partition_id=shard.partition_id, ...)`** (`rl-arena/arena/workers.py:402`): explicit direct-by-key fetch. 1-hop.

Same data flow as verl, just with the TQ I/O written out as a method call instead of hidden behind a decorator.

**For NeMo-RL we adopt verl's decorator path** because `Policy.train` already routes through `worker_group.run_all_workers_sharded_data` — a decorator is the smallest change. But the rl-arena shape would also work and is a good fallback if the decorator hits friction in the NeMo-RL dispatch path.

**Why my "400-600 LOC, load-bearing massive refactor" framing was wrong.** I conflated "needs new code" with "needs to rewrite the trainer." The trainer doesn't change. We need:

| Piece | What | Size |
|---|---|---|
| `shard_keys_by_seqlen(keys, seqlens, dp_world_size)` | Sort-by-seqlen + stride: `order = sorted(range(N), key=seqlens.__getitem__); shards[r] = order[r::dp_world_size]`. Same algorithm as rl-arena's `shard_for_dp` (`rl-arena/arena/dataplane_client.py:275-314`) and NeMo-RL's `BatchedDataDict.shard_by_batch_size(dynamic_batching_args=...)` branch (`batched_data_dict.py:404-414`). One algorithm, no strategy parameter. Operates on `list[str]` + `list[int]`. Does **not** modify `shard_by_batch_size` itself. | ~20 LOC |
| `policy.train_from_dp_meta(meta)` / `get_logprobs_from_dp_meta(meta)` driver-side | Build per-rank `KVBatchMeta` slices via the helper above; dispatch via the existing `run_all_workers_sharded_data` with `in_sharded_axes=["data_parallel"]`. | ~40 LOC each |
| Worker entrypoints `train_presharded` / `get_logprobs_presharded` | Take `KVBatchMeta`, call `self._dp_client.kv_batch_get(keys=meta.keys, partition_id=meta.partition_id, ...)`, run `materialize(layout="padded")`, then call into the **existing** per-rank training/logprob step (the body of today's `train_worker` / `logprob_worker` minus any outer sharding — those workers don't shard internally; sharding happens on the driver). | ~30 LOC each |
| `_dp_client` field on the policy worker | Initialized from the same factory the driver uses; in NoOp mode it's a passthrough. | ~10 LOC |
| Parity tests | New entrypoint vs legacy path: same loss, same grad norms, same metrics on a smoke config. | ~80 LOC |

**Total honest estimate: ~150-250 LOC**, not 400-600. The trainer worker body is reused as-is; we're adding a thin wrapper that does the TQ get on entry and the TQ put on exit, exactly like verl's `tqbridge`.

**Sharding algorithm choice (Phase 1: sort+stride only).** rl-arena's `shard_for_dp` settled on a single algorithm — `order[r::dp_world_size]` after sorting by seqlen — because (a) it's the algorithm NeMo-RL's `dynamic_batching_args` branch already uses, so we get parity for free; (b) it's deterministic and trivially testable; (c) the LPT / Karmarkar-Karp variants only buy a few percent in worst-case imbalance for typical long-tail seqlen distributions, not worth the extra surface in Phase 1. We follow the same choice. The bin-packing branch (`batched_data_dict.py:469-491`, used by NeMo-RL when `sequence_packing_args` is set) is a separate code path inside the worker — it runs *after* the per-rank fetch, on the rank's own slice. Driver-side sharding does not need to know about it.

**Even cleaner alternative — port verl's `tqbridge` directly.** Instead of a separate `train_presharded`, decorate the existing `train` worker with `@tqbridge`. The decorator inspects the first argument: if it's a `KVBatchMeta`, it does `kv_batch_get` and replaces the meta with a `BatchedDataDict`; if it's already a `BatchedDataDict` (legacy path), it passes through. Symmetric on the put side. This means **zero changes to the trainer** and the data-plane path is gated by the type of argument passed to `worker_group.run_all_workers_sharded_data`. Worth considering for Stage 1's interface design — let the decorator pattern be the public contract, not a parallel `_presharded` entrypoint set.

**Hop / shard accounting (corrected):**

| Pattern | Hops (data) | Driver materializes tensors? | Resharding |
|---|---|---|---|
| Today's NeMo-RL | 1 (driver→worker via Ray) | yes (full batch) | once, inside `policy.train` |
| Original plan as written | 2 + double-shard | yes (per-rank slice goes through `policy.train` again) | **twice** — broken |
| Walked-back 2-hop plan | 2 (TQ→driver→worker) | yes (full batch) | once, inside `policy.train` |
| **Phase 1 target (verl/rl-arena-shaped)** | **1 (TQ→worker direct)** | **no** (only `meta.keys + meta.sequence_lengths` cross driver) | once, on driver from metadata |
| rl-arena | 1 (TQ→worker direct via explicit `client.shard_for_dp` + `kv_batch_get`) | no | once, on driver from metadata (sort+stride) |
| verl | 1 (TQ→worker direct via `_balance_batch` + `@tqbridge` decorator) | no | once, on driver from `seq_len` tag (Karmarkar-Karp) |

**`shard_by_batch_size` is fine as-is.** We're not modifying it. The TQ path takes a different route via `shard_keys_by_seqlen` + the per-rank entrypoint; the legacy path keeps using `shard_by_batch_size` unchanged. No double-shard because the TQ path skips the legacy entrypoint entirely.

**`get_meta(dp_rank=R)` — unused.** TQ's `RankAwareSampler` returns disjoint-but-not-balanced shards. Driver-balance from metadata is the only pattern that produces seqlen-balanced shards. The `dp_rank` argument stays on the ABC for forward-compat but no call site uses it in Phase 1 or 2.

**`KVBatchMeta.sequence_lengths`** — populated by TQ from the `input_lengths` tag at `register_partition` / `kv_batch_put` time (verl reads it as a tag at `main_ppo_sync.py:1000`). The driver reads it from the meta object returned by `get_meta` — control plane only, no tensor fetch.

**TP/CP/PP siblings within a DP group — broadcast inside the group, do not fetch independently.** When mcore TP/CP/PP support lands, multiple worker processes share the same DP rank (they are TP/CP/PP siblings of each other). The rule (from rl-arena's README and verl's `_dispatch_data_to_tp` / Megatron's TP data-loading) is:

- Exactly one rank per (TP × CP × PP) group calls `kv_batch_get`. The other siblings receive the tensors via `dist.broadcast` inside the group's process group.
- **CP slicing of the sequence dimension happens in the model forward, not in the data plane.** Each CP rank gets the full sample tensor and slices its own region during the forward pass. TQ does not need a sub-sample slice API on its wire protocol.
- This means `shard_for_dp` / `shard_keys_by_seqlen` only ever produces `dp_world_size` shards — never `dp × tp × cp × pp` shards. The TP/CP/PP fanout is a worker-side concern handled with NCCL collectives, not a TQ concern.

For Phase 1 (FSDP2 only, TP=CP=PP=1), this rule is trivially satisfied since there are no siblings. We document it now so the boundary is set before mcore work begins.

---

### Stage 5 — Backend Swap Verification (G1)

**Goal:** Prove the Mooncake CPU RDMA backend works without code changes outside the adapter.

**Method:**
1. Run Stage 3 GRPO end-to-end with `backend="simple"` — capture wandb metrics.
2. Run identical config with `backend="mooncake_cpu"` — compare metrics.
3. Step-1 and step-N reward curves and loss must match within tolerance.

The whole change should be a single config flip. If it isn't, the abstraction has leaked.

---

### Stage 6 — Native Jagged Migration (deferred)

Trainer worker calls `materialize(layout="packed")` directly and skips the padded round-trip. Each migration is a worker-by-worker change behind a feature flag. Out of scope until Stages 1–5 are stable.

---

### Observability (sublayer, opt-in)

Independent layer over `DataPlaneClient`. Wraps any adapter with a
`MetricsDataPlaneClient` middleware that records `op | partition_id |
n_keys | n_bytes | wall_ms | status | fields` per call to a pluggable
`MetricsSink`. The trainer pulls a flat metrics dict via
`dp_client.snapshot()` once per step and merges into its existing
`logger.log_metrics(...)` payload. Off by default; one config flag opts
in.

```yaml
data_plane:
  observability:
    enabled: true
    sink: memory   # or 'log'
```

Layered design: the middleware is itself a `DataPlaneClient` and stacks
with future layers (integrity check, distributed tracing) without
touching the ABC or the TQ adapter. Future Layer 2 (server-side
controller introspection — `list_partitions`, `partition_stats`,
`queue_depth`) would extend the ABC; Layer 3 (integrity check) would
add a sibling middleware.

Full design: [`data_plane_observability.md`](./data_plane_observability.md).
Code: `nemo_rl/data_plane/observability/`.

---

## 4. Risks (and Mitigations)

### High — sequence packing & DP sharding

**R1. NeMo-RL's `shard_by_batch_size` does DP sharding + dynamic batching + sequence packing in one call.** If the driver pre-balances *and* the data is then fed back through `policy.train`, `shard_by_batch_size` re-shards it — the double-shard failure mode.
- **Mitigation (Phase 1, mandatory):** Add the presharded entrypoint described in Stage 4. The TQ path takes a separate route — driver permutes `meta.keys` via `shard_keys_by_seqlen` (lifted from `batched_data_dict.py:404-414, 469-491` into a metadata-only helper), dispatches per-rank key lists, workers call `kv_batch_get` themselves and skip `shard_by_batch_size`. The legacy path keeps using `shard_by_batch_size` unchanged. No double-shard because the TQ path doesn't traverse the legacy entrypoint. This matches verl's `_balance_batch` + `tqbridge` pattern (`verl/trainer/main_ppo_sync.py:998-1022`, `verl/utils/transferqueue_utils.py:296`). 1-hop, ~150-300 LOC total. The earlier "load-bearing massive refactor" framing was wrong — `shard_by_batch_size` doesn't need to be modified, just bypassed for the TQ path.

**R2. ~~GRPO group integrity~~ — RESOLVED, not a real risk.** Originally I worried that DP sharding could split `n_gens_per_prompt` siblings and break leave-one-out advantage. **Verl resolves this structurally:** `_compute_advantage` runs **centrally on the driver** (`main_ppo_sync.py:1135-1198`) — fetches the entire batch with `tq.kv_batch_get(keys=batch.keys, ...)`, computes per-prompt baselines, writes per-sample advantages back. The DP-sharded stages (old_logprob, ref_logprob, update_actor) only see per-sample advantages by then, so group structure is irrelevant. **Adopt this ordering: balance → old/ref logprob → advantage (central) → balance for training → policy update.** No group-aware sharding needed.

**R3. dp_rank semantics — clarified.** TQ's `RankAwareSampler` (`TransferQueue/transfer_queue/sampler/rank_aware_sampler.py`) keys a dict on `(partition_id, task_name, dp_rank, batch_index)` so TP/PP siblings within a Megatron-Core DP group get **identical** samples (cache hit), while different dp_ranks get **disjoint** samples (consumption marking removes used indices from the ready pool). **No reservation lock exists** — disjointness is from consumption tracking, not locking.
- **Mitigation (Phase 1):** Per Stage 4, the driver runs `shard_keys_by_seqlen` and dispatches per-rank `KVBatchMeta` slices; workers fetch via `kv_batch_get(keys=meta.keys, ...)`. We don't call `get_meta(dp_rank=R)` and don't rely on `RankAwareSampler` for balance. The `dp_rank` argument stays on the ABC for forward-compat.
- **TP/CP/PP siblings within one DP group (mcore future):** the right pattern is **NCCL broadcast inside the group**, not independent TQ fetches per sibling. One rank in the group calls `kv_batch_get`; the rest receive via `dist.broadcast`. CP sequence-dim slicing is done by the model forward, not by the data plane — TQ doesn't need sub-sample slice support on the wire. See Stage 4's TP/CP/PP subsection. This means `RankAwareSampler`'s "TP/PP siblings get identical samples" cache is a *fallback*, not the primary path; even when mcore lands, broadcast inside the group is preferred because it avoids `dp_world_size × tp × cp × pp` independent fetches.

### Medium — schema and lifecycle

**R4. NeMo-RL's `message_log` flattening produces multimodal extra keys dynamically** (grpo.py:1722-1725). `register_partition(fields=...)` requires fields up-front.
- **Mitigation:** Two options:
  - (a) Pre-declare a superset including all multimodal fields (`pixel_values`, `image_grid_thw`, …) at register time; tolerate unused field slots.
  - (b) Allow late field registration: extend the adapter to call `register_partition` lazily on first `kv_batch_put` with new fields.
  - **Choice for Phase 1:** option (a). Simpler, predictable storage layout. Multimodal pipelines are a small minority of runs.

**R5. Pickle vs zero-copy on the ZMQ path.** TQ SimpleStorage serializes via pickle. Tensors with `requires_grad=True`, shared memory, or non-contiguous layout will silently break or copy slowly.
- **Mitigation:** Codec layer (`codec.py`) calls `.detach().contiguous().cpu()` on every tensor before put. Document in `data_plane/README.md`. Add a debug assertion in dev builds.

**R6. Backpressure / OOM on the controller.** `storage_capacity` is fixed. Long-CoT rollouts at large `num_prompts × n_gens × n_steps_in_flight` can exceed it.
- **Mitigation:**
  - Document capacity sizing rule of thumb: `storage_capacity ≥ 2 × num_prompts × n_gens × max_seq_len × bytes_per_token × num_active_fields`.
  - Make `register_partition` fail loudly with a clear error if requested num_samples exceeds capacity headroom.

**R7. partition_id usage — corrected.** I originally proposed `f"{experiment_name}_{step}"` per-step partition IDs. **Verl uses static `"train"` / `"val"` strings** (`main_ppo_sync.py:326, 422, 467, 852`) and clear-and-reuses each step. partition_id is a **logical sample namespace**, not a per-step or per-device tag. The training step number lives in tags, not the partition name.
- **Mitigation:** Use `"train"` / `"val"` static IDs. Per-step partition naming would be required only for pipelined async training (step N+1 rollout overlapping with step N consumption), which is out of scope for Phase 1.

### Low — operational

**R8. Ray actor lifecycle / namespace isolation.** TQController is a global named actor. Two trainers in the same Ray cluster could in principle collide.
- **In practice:** verl's `tq.init()` takes no namespace parameter and the TQ codebase doesn't expose one. Standard Slurm-per-experiment deployment puts one Ray cluster per job, so collisions don't happen. **No mitigation required for Phase 1**; document the one-Ray-cluster-per-experiment assumption in `data_plane/README.md`.

**R9. tqbridge over-fetching (verl footgun).** verl's `tqbridge` decorator works correctly mechanically but fetches all fields because every call site leaves `KVBatchMeta.fields=None`, so the `select_fields` branch (`transferqueue_utils.py:262`) never fires. Cost: every model-forward stage drags the full sample record (`prompts, responses, attention_mask, rollout_log_probs, rm_scores, response_mask, routed_experts, ...`) when it only needs `input_ids, position_ids` — roughly 10× wire-byte waste. Caching does **not** fix this (see P1): per-stage rebalance reshuffles samples across workers, killing hit rate, and writeback fields are cold by definition.
- **Mitigation (per P2):** Adopt the decorator pattern but make `select_fields` required. Two acceptable paths:
  - **Phase 1 (caller-provides):** Every site sets `meta.fields = [...]` before invoking the decorated worker function. ~3 lines per call site, no signature change. Matches verl's direct call sites that already do this (`_compute_old_log_prob:1033`, `_compute_ref_log_prob:1101`, `_update_actor:1258`).
  - **Phase 2 (signature-derived):** Worker functions take field-named kwargs (`def compute_log_prob(self, input_ids, position_ids)`), decorator reads `inspect.signature` to pick the fetch set automatically. Cleaner but requires touching every worker and the dispatch chunking logic. Deferred.
- **Guard:** Decorator must raise if `meta.fields is None` and no signature-based inference is configured. **No silent full-fetch fallback.** Add to ABC contract test in Stage 1.

**R10. ABC drift between `DataPlaneClient` and future `nv-dataplane` implementation.**
- **Mitigation:** ABC contract test (`test_interface.py`) parameterized over all adapters. Any new adapter must pass it before being added to the factory.

**R11. Dynamic sampling / DAPO interaction with the partition lifecycle.** Current GRPO with `use_dynamic_sampling=True` (`grpo.py:803-986`) may run multiple gen sub-batches per training step, filtering each by non-zero std and accumulating into `batch_cache` until enough prompts survive. The naive per-step partition mapping ("one partition = one training step") doesn't fit because the surviving keys come from several rollout sub-batches.
- **Mitigation (Phase 1, minimal change):** Keep dynamic sampling in driver-only memory exactly as today. The data plane is *only* engaged once `is_batch_complete=True` (`grpo.py:1648`). Concrete recipe:
  - Generation, reward, std-based filtering, and `batch_cache` accumulation stay on the driver as `BatchedDataDict`. Rollout workers continue to return `BatchedDataDict` to the driver, **not** to `kv_batch_put`.
  - Once a complete training batch is assembled, the driver does *one* `kv_batch_put` to seed the partition. Stages 2-6 of the lifecycle (prev_lp, ref_lp, mask, advantage, train) run TQ-mediated as designed.
  - Cost: rollout output transits the driver once before going into TQ — same cost as today's path. We lose the "rollout writes directly to TQ" win during Phase 1, but get correctness and zero algorithm changes.
  - Code change: only the entrypoint that *constructs* `train_data` (`grpo.py:1711`) is wrapped; everything upstream is untouched. ~30 LOC.
- **Mitigation (Phase 2, full):** Per-rollout-sub-batch partitions (`partition_id=f"step{N}_gen{g}"`) with explicit cross-partition copy of the surviving keys into a final `step{N}_train` partition. Filter happens on the controller via tag query. Defer until Phase 1 lands.
- **Acceptance gate:** Phase 1 GRPO with `use_dynamic_sampling=True` produces identical metrics with `data_plane.enabled=True` vs `False` — add this to the Stage 5 verification matrix.

**R12. `message_log` carries non-tensor data that current GRPO indexes repeatedly.** Per the Tier-1/3 split in §1.1, only the structured pieces tensorize cleanly; raw `content` strings and `extra_env_info` must live out-of-band. The risk is that some GRPO code path silently expects to round-trip a fully-Python `message_log` through TQ.
- **Mitigation:** Audit all `message_log` access in `grpo.py` (`:1444, 1659, 1685, 2048, 2236-2350, 2734`) before Stage 3. Each access falls into exactly one of three buckets:
  - (a) Reads `token_ids` / `token_loss_mask` / `generation_logprobs` — replace with `materialize(td, layout="padded")` reads of the corresponding Tier-1 fields.
  - (b) Reads `role` for prompt-only extraction or mask construction — replace with `role_segments` CSR (Tier-1 enum).
  - (c) Reads `content` strings or env extras for logging — call `dp_client.fetch_oob(idx_list)` against the Ray object store (driver-side helper to be added in codec.py).
- **Aligned with Phase 1/Phase 2 jagged migration (P4):** Tier-1 fields are NestedTensors on the wire; `materialize(layout="padded")` keeps `policy.*` and `adv_estimator.*` signature-stable. Phase 2 flips trainers to consume `layout="jagged"` worker-by-worker.

**R13. Stage-completion / fault tolerance.** `mark_consumed` is not a real post-compute ack (TQ advances consumption inside `get_metadata(mode="fetch")`, `controller.py:1352`). A worker that fetches and crashes leaves the data marked consumed but un-produced.
- **Mitigation (Phase 1):** Use field-presence as the natural ready signal — when a stage `kv_batch_put`s its output field, the controller flips `production_status[sample, output_field] = 1` (`controller.py:503-555`). Step-level checkpoint restart handles worker crashes; no partial-step recovery. Removed `mark_consumed` from the public ABC; kept `check_consumption_status` for the clear-safety check.
- **Mitigation (Phase 2):** Reserved `<task>_done: bool` per consumer task, written as the *last* `kv_batch_put` of the stage. Recovery uses TQ's `force_fetch` mode (`controller.py:1357`) to re-issue keys whose `<task>_done` bit is 0 even though the payload field is 1. Defer until partial-step recovery becomes a requirement.

---

## 5. Open Questions

1. **~~dp_rank discovery from inside a worker~~ — RESOLVED (driver-broadcast).** Driver computes the balance from `meta.keys + meta.sequence_lengths` and dispatches per-rank `KVBatchMeta` slices via `run_all_workers_sharded_data(in_sharded_axes=["data_parallel"])`; each worker reads its own slice from the dispatched argument, not from a TQ `dp_rank` query. For mcore TP/CP/PP siblings within one DP group: one rank fetches and `dist.broadcast`s inside the group (per Stage 4 TP/CP/PP subsection); we don't use `RankAwareSampler` for that either.
2. **Validation pipeline.** Verl uses `partition_id="val"` and clears after each `_validate` (`main_ppo_sync.py:889`). NeMo-RL's `_validate` iterates `val_dataloader` directly today. Recommend Phase 1: keep validation in-memory (not on the critical hot path); revisit if validation throughput becomes a bottleneck.
3. **Async / sync rollout interaction.** `run_async_multi_turn_rollout` and `run_async_nemo_gym_rollout` already manage their own concurrency. Verify TQ async puts compose cleanly with their event loop — spike in Stage 3.
4. **Mooncake GPU RDMA timeline.** Tracked in `rl-arena/PROPOSAL_lazy_registration.md` and the upstream TQ PR. Out of Phase 1 scope but should not require any NeMo-RL changes when it lands.

---

## 6. Timeline (rough)

| Stage | Effort | Owner | Blocks |
|---|---|---|---|
| 1 — Foundation | 1 week | zhiyul | nothing — kicks off parallel work |
| 2 — Codec | 1 week | teammate A | depends on Stage 1 interface |
| 3 — GRPO integration | 2 weeks | teammate B | depends on Stages 1 & 2 |
| 4 — Per-rank fetch entrypoint (`shard_keys_by_seqlen` + `train_from_dp_meta` / `get_logprobs_from_dp_meta` + thin worker wrappers, OR a verl-style `tqbridge` decorator on the existing trainer) | ~1 week | teammate A | depends on Stage 3; ~150-300 LOC; this is where the 1-hop perf win materializes |
| 5 — Backend swap (Mooncake) | 0.5 week | teammate C | depends on Stages 3 & 4 (otherwise nothing to measure) |
| 6 — Native jagged | TBD | — | deferred |

---

## 7. References

**Data-plane integration patterns (both 1-hop, both valid; we pick verl's decorator for NeMo-RL):**
- **Verl (`tqbridge` decorator + `_balance_batch`):** `data-plane/verl/verl/utils/transferqueue_utils.py`, `data-plane/verl/verl/trainer/main_ppo_sync.py`
- **rl-arena (explicit `shard_for_dp` + direct `kv_batch_get`):** `data-plane/rl-arena/arena/{dataplane_client,pipeline,workers,seqlen_pack}.py`. After the recent updates, rl-arena's per-DP-rank API is verl-shaped — driver-balanced metas + worker-side direct fetch — just exposed as explicit methods instead of a decorator. `shard_for_dp` uses sort-by-seqlen + stride (the same algorithm as NeMo-RL's `dynamic_batching_args` branch).

**Backend stress baseline (orthogonal use of rl-arena):** `data-plane/rl-arena/arena/{backends,jagged_utils}.py` and `configs/`. Used for SimpleStorage / Mooncake CPU / Mooncake GPU comparison and jagged-tensor transport validation.

**Other:**
- **TransferQueue source:** `data-plane/TransferQueue/`
- **NeMo-RL existing packing:** `RL/nemo_rl/distributed/batched_data_dict.py:268` (shard_by_batch_size), `RL/nemo_rl/data/packing/algorithms.py`
- **NeMo-RL design doc:** `RL/docs/design-docs/sequence-packing-and-dynamic-batching.md`
