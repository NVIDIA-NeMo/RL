# TQ Data Schema — Design Discussion

Status: Draft — architectural direction.
Owner: `@zhiyul`

Scope: how the TQ data-plane adapter should carry _shape / schema_ information
alongside tensor data. What kind of information belongs on the wire, what
belongs beside it, and where the boundary should sit.

## TL;DR

- The wire representation between a TQ writer and a TQ reader has to carry
  enough information for the reader to reconstruct the same tensor the writer
  handed in. Today that information is smeared **per-row** across
  `BatchMeta.custom_meta` as boolean flags — a schema-shaped concern on a
  data-shaped channel.
- Two directions for improving that:
  1. **Per-column raw shape descriptor** — record each field's pre-wire shape,
     per row.
  2. **Per-batch (or per-partition) schema + per-row length hints** — declare
     structural type once, keep only what genuinely varies per row.
- Direction (2) is what every mature data system converges on: Kafka + Schema
  Registry, Arrow's `SchemaMessage`, Parquet's footer schema, Iceberg's table
  manifests, protobuf's `.proto`. Cost scales with schema-change frequency
  rather than data volume.
- The natural home for a per-partition schema in TQ is the **Controller**,
  which already owns partition-scoped state. That is an upstream API
  extension. In nemo-rl's own scope, the equivalent is a dedicated
  **`WireSchemaRegistry`** named Ray actor.
- Verl side-steps the entire problem by not letting the wire see
  rank-ambiguous dense tensors — all per-token fields are nested-jagged.
  Nemo-rl cannot take that route while shipping the `mooncake_cpu` KV
  backend, which forces `(N,) → (N, 1)` promotion.

---

## 1. The problem

TQ storage backends occasionally require the writer to reshape tensors so the
storage layer accepts them (concrete case today: `mooncake_cpu`'s
`KVStorageManager.extract_field_schema` has a 1D schema/data mismatch, so 1D
tensors are unsqueezed to `(N, 1)` on write). The reader has to invert those
reshapes to hand callers the shape they originally provided.

The reshape is lossy in general. For 1D promotion:

```
(N,)   ─ writer unsqueeze ─┐
                           ├──► wire: (N, 1)   ← indistinguishable at read
(N, 1) ────────────────────┘
```

Without an out-of-band signal, the reader cannot tell whether a `(N, 1)`
came from a `(N,)` (should squeeze) or from a genuine `(N, 1)` (must not).
Some sidecar has to travel with (or beside) the data.

The current adapter's sidecar is a set of boolean tags in `custom_meta`, one
per promoted field per row. That is the state we want to improve on.

---

## 2. The two shapes of the fix

### 2.1 Option 1 — Per-column, per-row shape descriptor

Record each field's actual pre-wire shape, per row.

```
custom_meta[0] = {"__wire_shape__": {"reward": [], "logprobs": [512], "attn": [512, 512]}}
custom_meta[1] = {"__wire_shape__": {"reward": [], "logprobs": [478], "attn": [478, 478]}}
```

Reader logic collapses to one generic restore per field:

```python
def _restore_shape(wire_tensor: torch.Tensor, declared_shape: list[int]) -> torch.Tensor:
    # 1. Rank difference from promotion → squeeze trailing 1s
    while wire_tensor.dim() > len(declared_shape) and wire_tensor.shape[-1] == 1:
        wire_tensor = wire_tensor.squeeze(-1)
    # 2. Size difference from padding → slice to declared size
    if tuple(wire_tensor.shape) != tuple(declared_shape):
        wire_tensor = wire_tensor[tuple(slice(0, s) for s in declared_shape)]
    return wire_tensor.contiguous()
```

Covers every current concern with one concept (shape):

| Case                     | Declared    | Wire          | Restored via                   |
|--------------------------|-------------|---------------|--------------------------------|
| 1D promotion             | `[N]`       | `[N, 1]`      | squeeze                        |
| SP padding to mult of 128| `[478]`     | `[512]`       | slice `[:478]`                 |
| Padded 2D per-token      | `[478, D]`  | `[512, D]`    | slice `[:478, :]`              |
| Promoted AND padded      | `[478]`     | `[512, 1]`    | squeeze + slice                |
| Untransformed dense      | `[N]`       | `[N]`         | no-op                          |
| Untransformed jagged     | not written | nested        | no-op                          |

Does **not** cover:

- dtype changes (e.g. fp32 → fp8 quantization on wire)
- reshape / repacking (e.g. quant blocks)
- transpose / layout changes
- any semantic transform that is not shape-only

### 2.2 Option 2 — Per-batch schema + per-row length hints

Once per batch (or per partition — see §4), declare each field's structural
type. Per row, carry only what legitimately varies (e.g. `input_lengths`).

```json
{
  "__nemo_rl_wire_schema__": {
    "v": 1,
    "fields": {
      "reward":     {"rank": 0, "dtype": "f32"},
      "logprobs":   {"rank": 1, "dtype": "f32", "seq_dim": 0, "length_ref": "input_lengths"},
      "advantages": {"rank": 1, "dtype": "f32", "wire_promoted": true},
      "attn_mask":  {"rank": 2, "dtype": "bool", "seq_dim": [0, 1], "length_ref": "input_lengths"}
    }
  }
}
```

```json
// per row
{"input_lengths": 512}
```

Reader dispatches on schema annotations:

- `wire_promoted: true` → squeeze trailing 1
- `length_ref` → look up per-row length, slice off padding
- future: `wire_dtype`, `wire_layout`, `shard_ref` — additive

### 2.3 Side-by-side

| Aspect                                | Option 1 (per-column raw shape) | Option 2 (per-batch schema + hints) |
|---------------------------------------|:-------------------------------:|:-----------------------------------:|
| Concepts                              | 1 (shape)                       | 4+ (rank, dtype, seq_dim, length_ref, wire_promoted, …) |
| Reader logic                          | Generic squeeze-plus-slice      | Per-annotation dispatch             |
| Handles 1D promotion                  | ✅                              | ✅                                  |
| Handles SP padding                    | ✅                              | ✅                                  |
| Handles dtype changes                 | ❌                              | ✅                                  |
| Handles reshape / repacking           | ❌                              | ✅                                  |
| Handles transpose / layout            | ❌                              | ✅                                  |
| Cost scales with…                     | data volume × field count       | schema-change frequency (near zero) |
| Cross-writer mix detection            | Weak (per-row match to wire)    | Strong (invariant schema check)     |
| Cognitive load                        | Low                             | Medium                              |
| Fits industry precedent               | Ad hoc                          | Kafka / Arrow / Parquet / protobuf  |

Both are legitimate. Option 1 is the simplest thing that could possibly work
for today's set of transforms; Option 2 aligns with how mature data systems
have handled the same concern for years.

---

## 3. Alignment with mature data systems

Every widely-deployed data system separates the "schema" channel from the
"data" channel. The reasons are always the same:

- Schemas change rarely; data flows constantly. Bundling them means paying
  per-record for a per-deployment concern.
- Schema needs a single authoritative source of truth. Duplicating per-record
  makes reconciliation impossible under writer disagreement.
- Schema evolution wants explicit versioning; per-record embedding obscures it.

| System                | Data channel                | Schema channel                                            |
|-----------------------|-----------------------------|-----------------------------------------------------------|
| **Kafka (Confluent)** | Message payload             | **Schema Registry** — separate service; message carries small schema-ID |
| **Apache Arrow**      | RecordBatch                 | `SchemaMessage` at stream head; not per record            |
| **Parquet**           | Row groups                  | File-level footer schema; not per row group               |
| **Iceberg / Delta**   | Data files                  | Table manifest / metadata layer; not per data file        |
| **protobuf**          | Wire bytes (tag-delimited)  | `.proto` at compile time; **zero schema on wire**         |
| **gRPC**              | Message frames              | Interface descriptor served separately (reflection API)   |
| **BigQuery**          | Column blocks               | Table schema in metastore                                 |

The pattern is unambiguous: **schema is metadata, and it belongs on the
metadata channel — not the data channel.** The only reason a schema-shaped
concern ever ends up on the data path is because no metadata surface was
available. That is the position TQ writers are in today.

---

## 4. Where should schema live in TQ?

Three metadata surfaces exist. Only one is the correct home.

| Surface                           | Scope           | Durable across processes | Right for schema? |
|-----------------------------------|-----------------|:------------------------:|:-----------------:|
| `BatchMeta.custom_meta` (list)    | Per row         | ✅                       | ❌ (schema does not vary per row) |
| `BatchMeta.extra_info` (Python)   | Per BatchMeta   | ❌ (client-local)        | ❌ (dies at process boundary)     |
| **TQ Controller actor state**     | Per partition   | ✅ (while controller lives) | ✅               |

The Controller is a named Ray actor that already owns per-partition state
(keys, task readiness, consumption tracking). Field schemas naturally belong
here — one authoritative source per partition, queried once per reader.

**This requires an upstream TQ API extension.** Suggested surface:

```python
tq_client.register_partition_schema(partition_id, schema)
tq_client.get_partition_schema(partition_id) -> schema | None
```

### 4.1 `WireSchemaRegistry` — nemo-rl-owned interim

Until upstream ships the API, nemo-rl can host the registry as its own named
Ray actor. Same architectural shape, nemo-rl scope.

```python
@ray.remote(name="nemo_rl_wire_schema_registry")
class WireSchemaRegistry:
    """Partition-level schema registry, separate from TQ's data plane."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict] = {}

    def register(self, partition_id: str, schema: dict) -> None:
        existing = self._schemas.get(partition_id)
        if existing is not None and existing != schema:
            raise RuntimeError(
                f"Schema conflict for partition {partition_id!r}: "
                f"existing={existing} incoming={schema}"
            )
        self._schemas[partition_id] = schema

    def get(self, partition_id: str) -> dict | None:
        return self._schemas.get(partition_id)
```

Writer path (once per partition, idempotent thereafter):

```python
def put_samples(self, sample_ids, partition_id, fields, tags):
    if fields is not None and self._shape_codec is not None:
        schema = _snapshot_schema(fields)
        ray.get(self._registry.register.remote(partition_id, schema))
        wire_fields = _apply_wire_transforms(fields)   # promote 1D, etc.
    tq.kv_batch_put(keys=..., fields=wire_fields, tags=tags)  # tags unmodified
```

Reader path (one Ray RPC on cold cache, local cache after):

```python
def get_samples(self, sample_ids, partition_id, select_fields):
    td = tq.kv_batch_get(...)
    if self._shape_codec is not None:
        schema = self._schema_cache.get(partition_id) or ray.get(
            self._registry.get.remote(partition_id)
        )
        self._schema_cache[partition_id] = schema
        td = _restore_from_schema(td, schema)
    return _from_wire(td)
```

### 4.2 Row-0 convention — considered and rejected

Stamp schema only on `custom_meta[0]`, leave rows `[1:]` empty of schema.
Reader inspects row 0.

Attractive because it costs zero new infrastructure. **Rejected** because:

- Correctness couples to row 0 always being present (breaks if row 0 is
  evicted or dropped by `select_fields`).
- Still on the data path — writers stamp on every write, readers fetch
  through `custom_meta`.
- Does not earn back the architectural improvement.

---

## 5. Cost analysis

### 5.1 Per-batch payload

Rough sizing for a realistic batch:
`rows = 1024`, `tensor_fields = 15`, `mean_rank = 2`.

| Approach                                 | Per-row payload | Batch payload |
|------------------------------------------|:---------------:|:-------------:|
| Per-field boolean tags (baseline today)  | ~200 B          | ~200 KB       |
| Option 1 (raw shape per column per row)  | ~225 B          | ~230 KB       |
| Option 2 (schema JSON per row)           | ~500 B          | ~515 KB       |
| Option 2 (compact schema per row)        | ~200 B          | ~210 KB       |
| Option 4 — Controller / Registry         | 0 (out of band) | ~1 KB total (schema fetched once per reader session) |

The first four options are within an order of magnitude of each other. Only
the registry approach breaks the linear-with-data pattern.

### 5.2 Reader compute

| Approach                | Consistency check                     | Restore per field                  |
|-------------------------|---------------------------------------|------------------------------------|
| Boolean tags (baseline) | O(fields × rows) prefix scans         | O(fields) squeeze                  |
| Option 1                | O(rows) shape-match to wire tensor    | O(fields) squeeze + slice          |
| Option 2 (in-band)      | O(rows) schema-string set-of-1 check  | O(fields) schema-dispatched        |
| Option 4 (registry)     | O(1) — registry rejects on register   | O(fields) schema-dispatched        |

Reader compute is not the bottleneck in any variant; payload size is the
axis that actually matters.

### 5.3 Registry actor overhead (Option 4)

- One Ray RPC per (client × partition) on cold cache. Amortised to zero over
  the client's lifetime.
- Actor state: one dict entry per active partition. Trivial memory.
- Failure mode: actor death loses schema. Mitigation: persist to a durable
  store on register (checkpointing pattern), or self-heal by having any
  writer re-register when a get returns `None`.

---

## 6. Generalisation

The point of separating schema from data is not the immediate concern —
it is _every future wire concern._

| Future concern              | Boolean tags / Option 1 impact           | Option 2 / registry impact          |
|-----------------------------|------------------------------------------|-------------------------------------|
| Add a second wire transform | New tag prefix / new shape field         | New schema field                    |
| FP4 / FP8 quantization      | Cannot express (need dtype)              | `wire_dtype` in schema              |
| Per-field layout change     | Cannot express (need transform id)       | `wire_layout` in schema             |
| Sharded field across ranks  | Cannot express (need shard_ref)          | `shard_ref` + per-row shard offsets |
| Schema evolution / versioning | Ad hoc                                 | Explicit `schema.v`                 |
| Multi-backend generalisation | Per-backend branching                   | Backend-agnostic dispatch on schema |

Every additional wire concern in the ad-hoc-tag approach requires either a
new key family in `custom_meta` (same anti-pattern accreting further) or an
escape hatch to a schema. A schema-first approach handles all of them by
adding one field to the schema — reader machinery unchanged.

This is why Kafka + Schema Registry beat "everyone puts their own schema in
every message" as an industry pattern. Not because the immediate message got
smaller, but because every future evolution stayed cheap.

---

## 7. Why verl does not need any of this

Cross-referencing `data-plane/verl/verl/utils/transferqueue_utils.py`
(pinned to `TransferQueue==0.1.6`):

- No `_promote_1d_leaves` — no 1D promotion at all.
- No shape sidecar, no schema, no `_from_wire`-style densification.
- No `mooncake_cpu` backend on the KV path. Mooncake is used only for the
  checkpoint engine (`verl/checkpoint_engine/mooncake_checkpoint_engine.py`),
  which is a separate data plane.
- Per-token fields (`log_probs`, `entropy`, `values`, `response_mask`) are
  represented as **nested-jagged tensors** end to end.
  `.to_padded_tensor()` is called only at the trainer boundary when a
  dense consumer needs it.

### 7.1 Verl's design decisions that make schema carriage unnecessary

1. **Pins older TQ (0.1.6)** — avoids the v0.1.9 behaviour that reconstructs
   every non-scalar field as a nested tensor.
2. **Does not use `mooncake_cpu` for the KV path** — avoids the
   `KVStorageManager.extract_field_schema` 1D-mismatch bug entirely.
3. **Per-token wire representation is nested-jagged, always** — a rank-1
   `(N,)` and a dense-rank-2 `(N, 1)` never occupy the same slot in the type
   system, so ambiguity has no way to arise.

### 7.2 Head-to-head

| Concern                                     | verl               | nemo-rl                              |
|---------------------------------------------|--------------------|--------------------------------------|
| TQ version                                  | 0.1.6 (pinned)     | 0.1.9                                |
| KV storage backend                          | Default (simple)   | `simple` **and** `mooncake_cpu`      |
| 1D promotion (`_promote_1d_leaves`)         | ❌ absent          | ✅ on `mooncake_cpu` writes          |
| Shape sidecar                               | ❌ absent          | ✅ per-row boolean tags today        |
| Densification of uniform-nested reads       | ❌ absent          | ✅ backend-neutral in `_from_wire`   |
| Per-token wire shape                        | Nested-jagged      | Dense `(N, T)` / `(N,)` / `(N, 1)`   |
| `(N,) vs (N, 1)` ambiguity                  | Impossible by construction | Real; needs a schema signal    |

Verl proves the workaround stack is **avoidable, not fundamental** — but
the avoidance costs specific design choices we do not make: we ship
`mooncake_cpu` for throughput, we run TQ v0.1.9 for checkpointing, and we
hand dense rank-ambiguous tensors to TQ.

### 7.3 Levers not being pulled and why

| Lever                                                | What it kills            | Why we do not pull it              |
|------------------------------------------------------|--------------------------|------------------------------------|
| Roll back to older TQ                                | Densification concern    | Loses checkpointing support        |
| Drop `mooncake_cpu` KV backend                       | Shape sidecar entirely   | Loses large-host throughput        |
| Adopt verl's all-nested per-token convention         | Shape sidecar entirely   | Caller-side API change; larger refactor |
| **Move schema off the data path (this document)**    | Per-row schema smearing  | The tractable lever                |

---

## 8. Migration path

Near-term:

1. Encapsulate the mooncake wire workaround into a `MooncakeShapeCodec` in
   `nemo_rl/data_plane/adapters/_mooncake_shape_workaround.py`. Keeps the
   wire format bit-for-bit identical while it exists; refactor is code
   organisation only.
2. Split `_from_wire` into pure densification (backend-neutral) and a
   codec-owned squeeze step. Removes the tri-state `promoted_1d_fields`
   argument.

Medium-term:

3. Introduce `WireSchemaRegistry` as a nemo-rl-owned named Ray actor. Codec
   writes through registry on `put`, reads through registry on `get` with
   local caching. `custom_meta` returns to being user-tag-only.
4. Move schema declaration from raw shape to structural type + length hints
   (§2 Option 2). Extensible to future wire transforms.

Long-term (upstream):

5. Upstream a partition-schema API on the TQ client / Controller.
6. Rewire `WireSchemaRegistry` as a thin adapter over the upstream API, or
   delete it in favour of upstream.
7. When upstream fixes `KVStorageManager.extract_field_schema` for 1D
   fields, delete `MooncakeShapeCodec` entirely. Schema-first design makes
   this delete one file plus three call sites, not a grep across five module
   functions and three constants.

---

## 9. Open questions

- Is `tq.kv_batch_get` in v0.1.9 a strict pass-through over
  `kv_retrieve_meta → select_fields → get_data`? If yes, the `simple` and
  `mooncake_cpu` read paths can unify into one helper. If no, keep two.
- Does TQ v0.1.9's `custom_meta` round-trip `str`-typed values through
  every backend? Determines whether Option 2's schema-JSON can be stored
  inline as a fallback if a registry actor is deemed premature.
- Is there an upstream tracker for the
  `KVStorageManager.extract_field_schema` fix that the "Drop when upstream
  TQ unifies the schema/data shapes for 1D fields" comment in the adapter
  should link? Filing one would give the workaround stack a documented exit.
- Would upstream TQ accept a partition-schema API contribution? If yes, the
  interim `WireSchemaRegistry` actor becomes a short-lived shim.

---

## 10. References

- `nemo_rl/data_plane/adapters/transfer_queue.py` — current adapter
- `data-plane/verl/verl/utils/transferqueue_utils.py` — verl's approach
- Confluent Schema Registry (Kafka) — the canonical schema-off-data-path pattern
- Apache Arrow IPC format — `SchemaMessage` at stream head
- Parquet file format — footer schema
- Iceberg / Delta Lake — table-level schema separation
