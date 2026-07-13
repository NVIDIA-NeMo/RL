# Follow-up: retire the sparse-native shadow (`_direct_sparse_delta_*_plan`) in `vllm_sparse_delta.py`

**Depends on**: commit `c4fdb34dd` (feat(refit): add layout-agnostic sparse-delta apply via vLLM weight_loader).
**Status**: Planning — not started.
**Cross-ref**: `pr-2444-workstream-retire-per-weight-dispatch.md` §Goal (this is the M4 milestone from that plan).

---

## High-level design: what's added, what's not needed

Refactor target shape. If you're doing the refactor, this is the picture to hold in your head.

### The whole design in one paragraph

The receiver has one job: take a sparse delta `(name, indices, values)` and
apply it to the corresponding vLLM weight, correctly, whatever that weight's
layout is (QKV-packed, gate/up-merged, MoE expert-stacked, plain linear,
embedding, etc.). vLLM already knows every layout — that's what its
per-module `weight_loader` methods encode. So the receiver should scatter
the delta into a dense HF-shape tensor, wrap the call in a context that
turns `Tensor.copy_` into `Tensor.add_` for the target's storage, and hand
the tensor to vLLM's `weight_loader`. That's it. No name-based dispatch, no
per-family plan builders, no arch-reject list.

### What stays / is added by `c4fdb34dd`

Two functions and one file. That is the entire replacement for the shadow.

| Kept from `c4fdb34dd` | What it does | Why it's the right altitude |
|---|---|---|
| `additive_weight_load_context(*targets)` in `vllm_sparse_delta_additive.py` | Monkey-patches `Tensor.copy_` → `Tensor.add_` for the block, scoped to any tensor sharing storage with a target. | The only piece of machinery unique to sparse-delta refit. Its job is to make a `weight_loader` that writes `param.data.narrow(...).copy_(x)` behave additively without knowing about QKV/gate_up/MoE packing. |
| `apply_sparse_delta_via_additive_load(name, indices, values, shape, dtype, model, device)` | Scatter → invoke vLLM's `weight_loader` inside the context. | The whole receiver entrypoint per param. ~15 lines. |
| The 5 unit tests in `test_vllm_sparse_delta_additive.py` | Scope, exception-restore, narrow-view, op-pattern parity, entrypoint. | Guard the two pieces above. |

### What's no longer needed (delete these)

All of it. The point of the additive context is that vLLM's own
`weight_loader` becomes the only entity that ever knows about packing.
`nemo_rl` stops replicating that logic.

| Symbol / file | What it does today | Why it's dead once additive is default |
|---|---|---|
| `_direct_sparse_delta_target_plan` (dispatch) | Routes by name regex to per-family plan builders. | No families to route to. |
| `_direct_sparse_delta_qkv_plan` | Reimplements vLLM's QKV `weight_loader` shard math. | vLLM's `qkv_proj.weight_loader` does it — additive-redirected via storage identity. |
| `_direct_sparse_delta_merged_column_plan` | Reimplements gate/up merged linear packing. | Same for `gate_up_proj.weight_loader`. |
| `_direct_sparse_delta_expert_plan` | Global→local expert ID mapping + MoE shard math. | vLLM's MoE `weight_loader` does it (needs e2e verification per follow-up). |
| `_direct_sparse_delta_mamba2_plan` | Mamba `A_log` log-space update + intermediate/groups/groups segmentation. | Special case: log-delta path — see the Mamba caveat below (one of three options). |
| `_direct_sparse_delta_shard_plan` | Generic `ColumnParallelLinear` / `RowParallelLinear` TP shard math. | vLLM's plain-linear `weight_loader` does it — already verified end-to-end in R5. |
| `_direct_sparse_delta_module` | Locates the target module from its name. | Not needed — additive path uses `dict(model.named_parameters())`. |
| `_make_sparse_delta_target_plan` (factory) | Builds a `_SparseDeltaTargetPlan` per target name. | No plans, no factory. |
| `_local_sparse_delta_update_inputs` | Post-processes shard-local (indices, values) for the shadow apply. | Additive scatters into HF-shape tensor once; no shard-local rewrites. |
| `_SparseDeltaTargetPlan` dataclass | Return type for the factory above. | Nothing returns it. |
| `_EXPERT_WEIGHT_RE` | Regex for the expert-plan dispatch. | No dispatch. |
| `_ADDITIVE_SENTINEL_PLAN` sentinel | Marker for the plan-cache when additive is used. | No plan cache. |
| `_is_plain_linear_name` + `_additive_apply_mode` (my additions) | Per-name mode selection. | No mode selection — additive is universal. |
| Env-var reads (`NRL_REFIT_SPARSE_APPLY_MODE`, `NRL_REFIT_SPARSE_APPLY_ALLOWLIST`) in `__init__` (my additions) | Read the mode at init. | No modes. |
| Arch reject list (`GptOssForCausalLM`, `Gemma3ForConditionalGeneration`) | Explicit rejects because shadow doesn't cover those loaders. | Additive delegates to whatever `weight_loader` vLLM defines. Rejection becomes unnecessary. |
| Verification bookkeeping in `_apply_sparse_weight_deltas` | Runtime self-check: sample a few deltas and verify post-apply. | Nice-to-have; only exists because the shadow was complex enough to warrant it. Additive is simple enough that unit tests + e2e diff replace it. |
| `test_vllm_sparse_delta_m2_envflag.py` (my addition) | Tests the mode-selection machinery. | No modes to test. |
| Per-family test cases in `test_vllm_sparse_delta.py` | `SimpleNamespace`-mocked verification of each plan family. | No plans. |
| `patches.py` NRL env-var whitelist entries (my additions) | Propagate the mode/dump env vars to vLLM internal workers. | No env vars. |

### The `_apply_sparse_weight_deltas` body after refactor (target shape, ~30 lines)

Rough sketch to reach for during refactor. Every path collapses to the same
straight-line call — no dispatch, no plan cache, no mode branching.

```python
def _apply_sparse_weight_deltas(self, payload_tensors, metadata):
    if self._direct_sparse_delta_targets is None:
        model = self.model_runner.model
        self._direct_sparse_delta_targets = (
            dict(model.named_parameters()) | dict(model.named_buffers())
        )
    targets = self._direct_sparse_delta_targets
    raw_locations, raw_values = payload_tensors
    mapper = getattr(self.model_runner.model, "hf_to_vllm_mapper", None)

    with torch.no_grad():
        for item in metadata:
            name = str(item["name"])
            target_name = (
                mapper._map_name(name) if mapper is not None else name
            )
            if target_name is None or target_name.startswith("draft."):
                continue  # skip unmappable / draft-model targets
            target = targets[target_name]
            value_start, value_end = int(item["value_start"]), int(item["value_end"])
            indices = sparse_codec.sparse_locations_for_item(
                item, raw_locations, device="cpu",
            )
            values = raw_values[value_start:value_end]
            apply_sparse_delta_via_additive_load(
                target_name, indices, values,
                tuple(item["shape"]), target.dtype,
                self.model_runner.model, target.device,
            )
```

Everything else that currently lives in this method (arch reject, plan
cache, verification, log-delta branching, range-encoding fast path) either
disappears or becomes a separate concern handled inside
`apply_sparse_delta_via_additive_load` if it needs to survive at all.

### Mamba caveat, revisited under this design

The Mamba `_direct_sparse_delta_mamba2_plan` is the one exception to
"delete everything". It applies `param = current * exp(delta)` instead of
`param = current + delta`. The additive context turns `.copy_` into
`.add_`; it does not know how to do a log-space multiplicative update.

Under this design, that leaves three symmetric choices — see the option
menu further down (M-a keep-Mamba-only, M-b extend-context-for-log-delta,
M-c drop-Mamba). Nothing else about the target shape changes.

---

## What this replaces

Commit `c4fdb34dd` landed the `additive` and `allowlist` modes behind
`NRL_REFIT_SPARSE_APPLY_MODE`, but kept the `plan` mode (default) and the
entire `_direct_sparse_delta_*_plan` shadow intact. That was a deliberate
staging choice: prove the additive path in production behind a flag before
deleting anything.

This follow-up is the actual deletion.

## Delete candidates (line counts measured on `c4fdb34dd`)

| Symbol | Location | Lines | Reason |
|---|---|---|---|
| `_direct_sparse_delta_qkv_plan` | vllm_sparse_delta.py:340–371 | 32 | vLLM's `qkv_proj.weight_loader` handles the packing via storage-identity redirect. |
| `_direct_sparse_delta_merged_column_plan` | :372–417 | 46 | Same for `gate_up_proj.weight_loader`. |
| `_direct_sparse_delta_expert_plan` | :502–564 | 63 | Same for MoE `weight_loader` (unverified — see risks). |
| `_direct_sparse_delta_mamba2_plan` | :418–501 | 84 | See Mamba caveat below — this one may need to stay or be re-homed. |
| `_direct_sparse_delta_shard_plan` | :617–672 | 56 | Plain-linear `weight_loader` handles it via storage-identity redirect. |
| `_direct_sparse_delta_target_plan` (dispatch) | :300–321 | 22 | Not needed once every family goes through additive. |
| `_make_sparse_delta_target_plan` (factory) | :565–616 | 52 | Not needed once no target plans exist. |
| `_direct_sparse_delta_module` | :290–299 | 10 | Same. |
| `_local_sparse_delta_update_inputs` | :673–751 | 79 | Shard-local index/value transforms only used by shadow. |
| `_SparseDeltaTargetPlan` dataclass | :36–49 | 14 | No longer instantiated. |
| `_EXPERT_WEIGHT_RE` | :29–33 | 5 | Only referenced by expert plan + `_is_plain_linear_name`. |
| `_ADDITIVE_SENTINEL_PLAN` | :51 | 1 | Cleanup marker for the plan-cache map. |
| `_is_plain_linear_name` + `_additive_apply_mode` | :96–125 | ~29 | Dispatch machinery — not needed once additive is universal. |
| Env-var reads in `__init__` | :~60–85 | ~15 | `NRL_REFIT_SPARSE_APPLY_MODE` / `NRL_REFIT_SPARSE_APPLY_ALLOWLIST` — not needed once additive is default. |
| Arch reject list in `_apply_sparse_weight_deltas` | :136–141 | ~6 | `GptOssForCausalLM`, `Gemma3ForConditionalGeneration` — additive delegates to vLLM's `weight_loader` which handles those. |
| Dispatch scaffolding in `_apply_sparse_weight_deltas` | :143–195 | ~40 | Plan-cache + `_use_additive` branching become dead. |
| Verification bookkeeping | :226–253 | ~30 | Runtime self-check; only exists for shadow path. |
| **vllm_sparse_delta.py total** | | **~584 lines removable** |
| `test_vllm_sparse_delta_m2_envflag.py` | (whole file) | 208 | Tests the dispatch we're removing. |
| Per-family cases in `test_vllm_sparse_delta.py` | (partial) | ~200 of 288 | Tests each shadow plan against `SimpleNamespace` mocks. |
| **Tests total** | | **~408 lines removable** |
| Env-var whitelist in `patches.py` | +4 lines | 4 | Not needed once env vars are gone. |
| `docs/design-docs/sparse-delta-refit.md` | (partial rewrite) | ~30–50 lines to update | Reflect the simpler model. |

**Net code delta:** ~-1000 lines (implementation + tests), against ~+30 lines to simplify `_apply_sparse_weight_deltas` to unconditionally call `apply_sparse_delta_via_additive_load`. **~-970 net.**

## The Mamba caveat

`_direct_sparse_delta_mamba2_plan` sets `log_delta_transform=True` for
Mamba's `A_log` parameter. On apply, this path does
`current * exp(delta)`, not `current + delta` — a log-space multiplicative
update, not additive. The `additive_weight_load_context` monkey-patches
`Tensor.copy_` → `Tensor.add_`, so calling vLLM's `weight_loader` for
Mamba's `A_log` would apply a **wrong** update.

Options:

- **Option M-a: Keep `_direct_sparse_delta_mamba2_plan` as the sole shadow
  survivor.** Route only Mamba log-delta names through it; everything else
  additive. Simplest correctness story; largest residual shadow.
- **Option M-b: Extend the additive context.** Add a
  `log_delta_weight_load_context` that patches `Tensor.copy_` → an in-place
  `mul_(exp(src))`. Add a per-param mode selector at the call site.
  Cleaner net-diff; more moving parts in the context module.
- **Option M-c: Drop Mamba support.** If Mamba isn't a target of the
  sparse-delta refit path (verify with maintainers), delete the log-delta
  handling entirely. Largest simplification; requires confirmation.

Pick M-a for the first pass unless maintainers confirm M-c.

## Prerequisites (must be true before landing)

1. `c4fdb34dd` (or successor) is merged and observed green in main-branch
   CI for at least one full sweep.
2. E2e parity harness (see `pr-2444-followup-e2e-harness.md`) is checked in
   under `tests/functional/` and green.
3. Per-family e2e parity check for each family whose shadow is being
   deleted (Llama plain-linear + QKV + gate_up is already covered by R5
   here; MoE and — depending on M-a/b/c — Mamba are not).

## Cost estimates

### Code / dev time

| Task | Estimate |
|---|---|
| Refactor `_apply_sparse_weight_deltas` to call additive unconditionally | ~4 hrs |
| Delete shadow methods + dependent tests, silence dead imports | ~2 hrs |
| Grep for callers of every deleted symbol; fix or delete | ~1 hr |
| Handle Mamba log-delta per chosen option (M-a/b/c) | 2 hrs (M-a) / 6–8 hrs (M-b) / 1 hr (M-c) |
| Handle verification (drop or reimplement in additive path) | ~2 hrs |
| Update `docs/design-docs/sparse-delta-refit.md` | ~2 hrs |
| PR + review cycle + revisions | ~1–2 dev-days |
| **Total** | **~2–3 dev-days** end-to-end |

### GPU / Slurm time to validate

Sunk-cost items (already run in this workstream, do not re-pay):

- Llama-3.2-1B plain-linear + QKV + gate_up parity: R5 = 2 nodes × 8 GPUs × ~6 min × 2 modes = **~3.2 GPU-hours (sunk)**.
- 24 unit tests (M1 + M2): 1 node × 2 GPUs × ~4 min = **~0.13 GPU-hours (sunk)**.

Incremental items (must be paid before the deletion PR merges):

| Item | Config | GPU-hours |
|---|---|---|
| MoE parity e2e | Mixtral-8x7B or Qwen-MoE / 4n8g / 2 modes / ~10 min | ~10.5 |
| Mamba parity e2e (if not M-c) | Zamba or similar / 2n8g / 2 modes / ~10 min | ~5.3 |
| FP8 parity e2e (unblocks arch-reject removal) | Any FP8-quantized recipe / 2n8g / 2 modes / ~10 min | ~5.3 |
| GptOssForCausalLM parity e2e | GPT-OSS 20B / 2n8g / 2 modes / ~10 min | ~5.3 |
| Gemma3ForConditionalGeneration parity e2e | Gemma-3 / 2n8g / 2 modes / ~10 min | ~5.3 |
| Nightly regression sweep after the deletion | Existing GRPO nightly.txt | ~20–40 |
| **Total incremental** | | **~50–70 GPU-hours** |

### Runtime overhead (memory + compute + wall-clock)

Not measured this session — R5 was correctness-only. Estimates below are
based on the additive path's structure and the shadow's known dominance
of the apply step. **Milestone 3 benchmarks (plan §M3) are the way to
resolve these — do not treat these numbers as ground truth.**

#### Memory overhead per apply

The additive path allocates one HF-shape dense tensor per param (`torch.zeros(shape)`) and frees it before the next param. Peak transient allocation is bounded by the single largest per-rank tensor, not by the model size.

| Model | Largest per-rank param (bf16) | Additive peak (transient, MB) | Plan peak (sparse buf, MB) | Δ (MB) |
|---|---|---|---|---|
| Llama-3.2-1B, tp=1 | `embed_tokens` = 128256 × 2048 | ~500 | ~25 (5% sparsity of the same param) | +475 |
| Llama-3.1-8B, tp=2 | `embed_tokens` shard = 128256 × 2048 | ~500 | ~25 | +475 |
| Llama-3.1-70B, tp=8 | `embed_tokens` shard = 128256 × 1024 | ~250 | ~13 | +237 |
| DeepSeek-V3 (671B), tp=16 | `embed_tokens` shard ≈ 256256 × 448 | ~220 | ~11 | +209 |

**Bottom line:** per-worker RSS peak grows by O(largest_param / tp_size).
For all realistic configs this is **<1 GiB extra per worker**, and it is
transient (freed before the next param starts). Not a deal-breaker; not
free either.

Steady-state RSS delta after the refit: 0 — the tensor is freed. Only the
watermark matters.

#### Compute overhead per apply

Per param the additive path adds three things beyond the plan path:

1. `torch.zeros(target_shape)` — HF-shape allocation.
2. `dense.view(-1).index_copy_(0, indices, values)` — scatter of the
   sparse subset.
3. vLLM's `weight_loader` doing an HF-shape → shard `.narrow(...).copy_(...)`,
   which our context redirects to `.add_()`.

The plan path skips (1) and (2) and does a shard-local `index_add_`
directly on `param.data`. Cost ratio scales with sparsity:

- At 10% sparsity: additive ≈ 3–5× the apply-only cost of plan.
- At 5% sparsity: additive ≈ 5–10× the apply-only cost of plan.
- At 1% sparsity: additive ≈ 20–50× the apply-only cost of plan.
- At 0.5% sparsity: additive ≈ 50–100× the apply-only cost of plan.

Reason: the scatter + `weight_loader` work is O(HF-shape), not
O(sparse-values). Sparser deltas make the constant HF-shape work
proportionally more expensive.

#### Wall-clock overhead per refit

Refit wall-clock = payload upload (Megatron) + transport (S3/ZMQ) +
payload download (vLLM worker) + decode + apply + Ray sync. Apply is one
of several segments.

If apply is 10–30% of total refit wall-clock (workstream plan's
assumption), then apply getting 5× slower means total refit wall-clock
regresses by ~5–20%. Getting 50× slower means ~40–150% regression.

**Bar to hit**: the plan doc's success criterion (§Success criteria) is
"Default-path apply latency regression ≤ 15% of total refit wall-clock at
typical sparsity." Whether M4 lands with a universal-additive default or
keeps a hot-param allowlist depends entirely on this measurement.

#### Additional per-block overhead of the context manager itself

The context patches `torch.Tensor.copy_` at the class level for the block.
Every `.copy_()` inside the block dispatches through a Python wrapper —
one extra attribute lookup (`self.untyped_storage().data_ptr()`), one set
membership check, one branch. Per-call cost is measured in single-digit
microseconds. For a full refit with ~200 params, this is ~1ms of pure
wrapper overhead. Ignorable.

Global side effect: the patch is process-wide for the block. Any other
tensor `.copy_()` happening concurrently in the same process (e.g., a Ray
callback firing on another thread) would go through the wrapper. Not
observed to be a problem in this session but worth flagging for
maintainers who might parallelize apply in the future.

#### Benchmark cost to resolve these estimates

Executing plan §Milestone 3 as spec'd:

- 4 sparsity levels (0.5%, 1%, 5%, 10%) × 3 model families × 2 transports = 24 configs.
- Llama-3.1-8B at 1n8g: 4 × 2 = 8 configs × ~10 min = **~10.7 GPU-hours**.
- MoE (~70B) at 4n8g: 8 configs × ~15 min = **~64 GPU-hours** (dominant).
- Mamba at 1n8g: 8 configs × ~10 min = **~10.7 GPU-hours**.
- **Full benchmark sweep: ~85 GPU-hours.**

This is in addition to the correctness-validation GPU-hours listed above.

### Risk (silent-breakage cost if we're wrong)

| Family | Risk | Cost if it breaks |
|---|---|---|
| Mamba (option M-a) | Low — path unchanged | 0 |
| Mamba (option M-b) | Medium — new log-delta context is untested | 1 dev-day debug + revert |
| Mamba (option M-c) | Low if truly unused; catastrophic if a downstream Mamba recipe silently corrupts | Confirm before landing |
| MoE | Medium — additive works structurally but expert `weight_loader` may have quirks | 1 dev-day debug |
| FP8 | Low — additive is opt-in; can gate rejection to just this family | 0 if kept gated |
| GptOss / Gemma3 | Low — these were rejected upstream anyway; additive is strictly a widening | 0 |
| Verification drop | Low — was a self-check, not a correctness guarantee | Lose runtime canary |

## Non-goals

- Benchmark suite (workstream M3-5b): separate follow-up.
- Rewriting the sender-side (`weight_transfer_remote_sparse.py`, Megatron
  side) to align with the simpler receiver: separate PR, not blocking.

## Success criteria

- **Code**: `vllm_sparse_delta.py` ends up under 400 lines (currently 895).
- **Tests**: unit tests continue to pass via `L0_Unit_Tests_Vllm_2.sh`.
- **E2e**: each per-family parity check within `atol=1e-5` (matches R5
  bar).
- **Docs**: `docs/design-docs/sparse-delta-refit.md` reflects the "delegate
  to vLLM `weight_loader`" model — no per-family plan tables.

## What to link back to

Post as a follow-up comment on the merged `c4fdb34dd` PR when this ships:

> Follow-up landed: retired the `_direct_sparse_delta_*_plan` shadow that
> `c4fdb34dd` set up the replacement for. Receiver now delegates all
> packing to vLLM's `weight_loader` via storage-identity additive apply.
> ~1000-line net delete. Details / cost / risk analysis:
> `pr-2444-followup-m4-shadow-removal.md` on the review branch.
