# NeMo-RL Data Plane Integration Plan

**Owner:** zhiyul
**Date:** 2026-05-01
**Status:** Stage 1 ready to start — designed for parallel team execution
**Reference prototype:** `../rl-arena/`
**Reference integration:** `../verl/verl/utils/transferqueue_utils.py`, `../verl/verl/trainer/main_ppo_sync.py`
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

**P1 — Avoid worker-side caching whenever possible.** TQ is the source of truth. Building a worker-side cache to "amortize" over-fetches reintroduces three problems we don't have today: (a) cache invalidation when a writeback updates a field, (b) low hit rate when stages reshuffle samples across DP ranks (verl's `_balance_batch`, our `seqlen_balanced_shard`), (c) memory cost on every worker (~100 MB+ at typical batch sizes). Fix the upstream over-fetch instead — see P2.

The exception is **read-only fields that are large, stable, and re-read every step** (e.g., `input_ids` / `position_ids` for repeated model forwards on the same samples). Cache only those, and only if profiling demands it. Default = no cache.

**P2 — Use `tqbridge` (transparent decorator) but always pass `select_fields`.** The decorator pattern is good — it hides the put/get plumbing, keeps worker functions clean, and is a familiar pattern from verl. The footgun is only that verl's current call sites set `KVBatchMeta.fields=None`, so the `select_fields` branch at `transferqueue_utils.py:262` never triggers and every call fetches the full sample record (~10x waste).

We adopt the decorator but **make `select_fields` a required argument**, populated either (a) by the caller setting `meta.fields = [...]` before invoking the decorated function, or (b) auto-derived via `inspect.signature(func).parameters` for kwargs-aligned signatures. Either way, the decorator never falls through to fetching all fields.

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
| 1 | tensor on the bus | `input_ids`, `logprobs`, `advantages`, `total_reward`, `idx`, `image_grid_thw`, tokenized prompts/responses | RDMA'd as contiguous device/host buffer; no serialization |
| 2 | `tags` on controller | `prompt_uid`, `step_id`, `dp_rank` hint, `priority` (JSON-serializable primitives) | Lives in TQ controller's tag table; never on storage bus, never RDMA'd |
| 3 | out-of-band | full `message_log` pre-flatten, `extra_env_info`, env state with mixed types, debug payloads | Ray object store or in-actor memory; **not supported by the data plane** |

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
│  TransferQueue pip package (transfer_queue==0.1.6) — UNMODIFIED │
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
    partition_id: str
    task_name: str
    keys: list[str]
    sequence_lengths: list[int] | None = None      # populated by controller from input_lengths field
    fields_available: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.keys)

class DataPlaneClient(ABC):
    """Stable boundary between NeMo-RL and any data-plane impl.
    All call sites in algorithms/experience/models go through this."""

    @abstractmethod
    def register_partition(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
        grpo_group_size: int | None = None,
    ) -> None: ...

    @abstractmethod
    async def kv_batch_put(
        self,
        partition_id: str,
        keys: list[str],
        values: TensorDict,
        tags: list[dict[str, Any]] | None = None,
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
    ) -> TensorDict: ...

    @abstractmethod
    def kv_batch_put_back(
        self,
        meta: KVBatchMeta,
        values: TensorDict,
    ) -> None: ...

    @abstractmethod
    def mark_consumed(self, meta: KVBatchMeta) -> None: ...

    @abstractmethod
    def check_consumption_status(
        self, partition_id: str, task_names: list[str]
    ) -> bool: ...

    @abstractmethod
    def kv_clear(self, partition_id: str) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
```

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
3. `await client.kv_batch_put("smoke", ["a","b","c","d"], TensorDict({"x": torch.arange(4)}))`
4. `meta = client.get_meta("smoke", "read", ["x"], batch_size=4)`
5. `data = client.get_data(meta)`
6. `assert torch.equal(data["x"], torch.arange(4))`
7. `client.mark_consumed(meta); client.kv_clear("smoke"); client.close()`

**`test_smoke_multinode.py`** — same as above but launched via `RL/ray.sub` over 2 nodes, exactly the way `rl-arena/launch/run_arena.sh` already does. Verifies controller-actor placement and ZMQ across hosts.

**Pip dependency** — add `transfer_queue==0.1.6` to `pyproject.toml` as an optional extra:

```toml
[project.optional-dependencies]
data-plane = ["transfer_queue==0.1.6"]
```

Same `try/except ImportError` pattern verl uses (`verl/utils/transferqueue_utils.py:35-57`) so NeMo-RL still imports cleanly without TQ installed; failure deferred to factory call when `enabled=True`.

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

**Stages (from rl-arena pipeline, ported to NeMo-RL):**

| Stage | Producer | Consumer | TQ ops |
|---|---|---|---|
| 0 — register | driver | — | `register_partition(fields, num_samples, consumer_tasks=["adv","train"])` |
| 1 — generation | rollout workers (vLLM/SGLang) | — | `kv_batch_put(input_ids, output_ids, logprobs, input_lengths, total_reward)` |
| 2 — reward | (folded into Stage 1 — already computed by `run_multi_turn_rollout`, write together) | — | merged with Stage 1 put |
| 3 — ref logprob | ref policy workers | driver-balanced shard or `get_meta(dp_rank=r)` | put `reference_policy_logprobs` |
| 4 — advantage | **driver process (centrally)** | `get_meta(blocking=True, batch_size=N_total)` — fetches whole partition | put `advantages, token_mask, sample_mask`; `mark_consumed("adv")` |
| 5 — policy | DP-rank train workers | **driver-side `seqlen_balanced_shard` → `kv_batch_get(keys=...)` per rank** (NOT `get_meta(dp_rank=R)`) | put `prev_logprobs`; `mark_consumed("train")` |
| 6 — clear | driver | `check_consumption_status(["adv","train"])` then `kv_clear` | |

**Stage 4 (advantage) runs centrally on the driver, not on DP-sharded workers.** This matches verl (`main_ppo_sync.py:1135-1198` — `_compute_advantage` calls `tq.kv_batch_get(keys=batch.keys, ...)` with the entire batch on the driver process). GRPO leave-one-out baselines need per-prompt grouping across all `n_samples_per_prompt`; doing it centrally avoids any cross-rank coordination. Compute is cheap (no model forward).

**Stage 5 (policy) uses driver-side global balancing**, not TQ's `dp_rank` cache. The driver does one `get_meta(batch_size=total)` to read all (key, seqlen) pairs, runs `seqlen_balanced_shard` (LPT) to balance tokens across DP ranks, then sends each rank an explicit key list. Each rank does `kv_batch_get(keys=[...])`. This matches both verl (`_balance_batch` at `main_ppo_sync.py:998` reorders the `KVBatchMeta` via Karmarkar-Karp) and rl-arena (`pipeline.py:152-186` + `seqlen_pack.py:68`).

**When `get_meta(dp_rank=R)` is actually used**: only when mcore TP/PP siblings within the same DP group fetch independently (the `RankAwareSampler` cache makes them all see the same data). For NeMo-RL's current FSDP2 path, the driver-broadcast pattern is sufficient — `dp_rank` argument can be deferred until mcore support is added.

**Where the changes land:**
- `algorithms/grpo.py` — orchestration; conditional branch on `data_plane.enabled`
- `experience/rollouts.py` — generation worker writes to TQ instead of returning the full BatchedDataDict
- `models/policy/lm_policy.py` — `get_logprobs` writeback path
- `algorithms/advantage_estimator.py` — read/write through client

**Backwards compatibility:** if `data_plane.enabled=False`, code path is unchanged from today. The TQ branch is feature-gated everywhere.

---

### Stage 4 — Sequence Packing Integration

**Goal:** Make TQ-fetched data work with NeMo-RL's existing `BatchedDataDict.shard_by_batch_size(sequence_packing_args=...)` and `make_microbatch_iterator_for_packable_sequences()`.

**The principal sharding pattern** (validated by both verl and rl-arena):

```
Driver:                                    Workers (DP rank r):
─────────                                  ────────────────────
1. get_meta(batch_size=ALL)                # waits until full partition ready
   → meta.keys, meta.sequence_lengths
2. shards = seqlen_balanced_shard(         # LPT — balanced token counts
       zip(meta.keys, meta.sequence_lengths),
       n_shards=dp_world_size,
   )
3. for r in range(dp_world):
       update.remote(shards[r])  ───────►  receives explicit (key, seqlen) list
                                           kv_batch_get(keys=[...])
                                           local pack_sequences()
                                           local microbatch loop
                                           kv_batch_put(keys=[...], prev_logprobs)
4. mark_consumed("policy_update")
```

**Plan:**
- Controller side: `KVBatchMeta.sequence_lengths` populated by TQ from the `input_lengths` field tag (no tensor fetch).
- Driver side: port `seqlen_balanced_shard` (LPT) from `rl-arena/arena/seqlen_pack.py:68`. Drop-in compatible with NeMo-RL's existing `BatchedDataDict.shard_by_batch_size` once a `BatchedDataDict.from_tensor_dict` adapter exists.
- Worker side: keep NeMo-RL's existing packer (`nemo_rl/data/packing/algorithms.py`). After `kv_batch_get(keys=shards[r])` returns the TensorDict, build a `BatchedDataDict` and run `make_microbatch_iterator_for_packable_sequences()` unchanged.

**Why driver-side balancing instead of TQ's `dp_rank` sampler:** `get_meta(dp_rank=R)` only gives **disjoint** shards (consumption-based). Sequence packing needs **balanced** shards (each rank gets a mix of long+short for equal token counts). One rank getting all the long samples destroys packing efficiency. From `rl-arena/arena/workers.py:386-396`:

> `TQ's dp_rank sampler only gives DISJOINT shards, not BALANCED — defeating sequence packing's purpose. Driver-side global balancing is the only correct pattern (matches verl's seqlen_balancing.py:rearrange_micro_batches).`

**Critical:** keep planning *outside* the controller. Controller exposes lengths via tags; driver computes the balanced split; workers run NeMo-RL's local packer within their slice.

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

## 4. Risks (and Mitigations)

### High — sequence packing & DP sharding

**R1. NeMo-RL's `shard_by_batch_size` does DP sharding + dynamic batching + sequence packing in one call.** TQ-side sharding (`get_meta(dp_rank=R)`) and NeMo-RL-side packing must not duplicate planning.
- **Mitigation:** Driver does global balanced sharding once (`seqlen_balanced_shard` from `rl-arena/arena/seqlen_pack.py:68`); workers receive explicit key lists; NeMo-RL's existing local packer plans microbatches within the rank's slice. Validated in `rl-arena/arena/pipeline.py:152-186`.

**R2. ~~GRPO group integrity~~ — RESOLVED, not a real risk.** Originally I worried that DP sharding could split `n_gens_per_prompt` siblings and break leave-one-out advantage. **Verl resolves this structurally:** `_compute_advantage` runs **centrally on the driver** (`main_ppo_sync.py:1135-1198`) — fetches the entire batch with `tq.kv_batch_get(keys=batch.keys, ...)`, computes per-prompt baselines, writes per-sample advantages back. The DP-sharded stages (old_logprob, ref_logprob, update_actor) only see per-sample advantages by then, so group structure is irrelevant. **Adopt this ordering: balance → old/ref logprob → advantage (central) → balance for training → policy update.** No group-aware sharding needed.

**R3. dp_rank semantics — clarified.** TQ's `RankAwareSampler` (`TransferQueue/transfer_queue/sampler/rank_aware_sampler.py`) keys a dict on `(partition_id, task_name, dp_rank, batch_index)` so TP/PP siblings within a Megatron-Core DP group get **identical** samples (cache hit), while different dp_ranks get **disjoint** samples (consumption marking removes used indices from the ready pool). **No reservation lock exists** — disjointness is from consumption tracking, not locking.
- **Mitigation:** For Phase 1 (FSDP2 only), we **don't use `get_meta(dp_rank=R)` for policy training** at all — driver-side `seqlen_balanced_shard` + explicit `kv_batch_get(keys=...)` is the primary pattern (matches verl + rl-arena). The `dp_rank` cache becomes relevant only when mcore TP/PP support is added; even then, the driver pattern still works (driver broadcasts the same key list to all TP siblings of dp_rank R), and `dp_rank` is only a fallback when workers need to fetch independently without driver coordination.

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

---

## 5. Open Questions

1. **~~dp_rank discovery from inside a worker~~ — RESOLVED (deferred).** Originally a worry. With the driver-broadcast pattern (driver computes `seqlen_balanced_shard`, sends explicit key lists to each rank), workers don't need to know their own dp_rank for policy training — they receive the keys directly. dp_rank threading only matters if/when mcore TP/PP support is added and we want the `RankAwareSampler` cache; even then, the driver-broadcast alternative still works and may be preferable.
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
| 4 — Sequence packing | 1 week | teammate A | depends on Stage 3 |
| 5 — Backend swap (Mooncake) | 0.5 week | teammate C | depends on Stage 3 |
| 6 — Native jagged | TBD | — | deferred |

---

## 7. References

- **Prototype:** `data-plane/rl-arena/arena/{dataplane_client,backends,pipeline,workers,seqlen_pack,grpo_groups}.py`
- **Verl integration:** `data-plane/verl/verl/utils/transferqueue_utils.py`, `data-plane/verl/verl/trainer/main_ppo_sync.py`
- **TransferQueue source:** `data-plane/TransferQueue/`
- **NeMo-RL existing packing:** `RL/nemo_rl/distributed/batched_data_dict.py:268` (shard_by_batch_size), `RL/nemo_rl/data/packing/algorithms.py`
- **NeMo-RL design doc:** `RL/docs/design-docs/sequence-packing-and-dynamic-batching.md`
