# NCCL Reshard Refit (Experimental)

> **Experimental**: `nccl_reshard_refit` is an experimental feature.

The default non-colocated transport broadcasts every **full** parameter tensor from the
training ranks to every generation rank. `nccl_reshard_refit` replaces that for the bulk
of the payload with a **shard-to-shard reshard**: each training rank sends only its local
shard, and each generation rank receives exactly the bytes of its own (differently
parallelized) shard. This is both faster and lighter on memory since no rank ever
materializes or receives the full tensor.

## Enabling It

Add the config key (it is `NotRequired` in `PolicyConfig`, so use `+` when overriding
from the CLI):

```bash
uv run ./examples/run_grpo.py \
  --config <your_config>.yaml \
  policy.generation.colocated.enabled=false \
  ... \
  +policy.nccl_reshard_refit=true
```

At setup, `check_nccl_reshard_refit_support()` validates the configuration and raises a
single `ValueError` listing every violation. The current requirements are:

* **Non-colocated only** — `policy.generation.colocated.enabled=false`. The colocated
  path uses IPC and is unaffected by this feature.
* **Megatron training backend** — `policy.megatron_cfg.enabled=true` (the DTensor
  training backend is not supported yet.).
* **vLLM generation backend** — `policy.generation.backend=vllm` (SGLang and TRTLLM
  backend is not supported yet.).
* Megatron `expert_tensor_parallel_size` (i.e., ETP) must be 1; custom PP layouts
  (`pipeline_model_parallel_layout`, virtual PP > 1, embedding/loss pipeline-split
  accounting) are not supported yet.
* **Precision** must match end to end: BF16 train ↔ BF16 gen, or FP8 train
  (`fp8_param=true` + blockwise recipe) ↔ FP8 gen (`vllm_cfg.precision=fp8`).
  BF16 train ↔ FP8 gen is not supported yet.
* vLLM expert parallelism is supported with the NeMo RL convention
  `expert_parallel_size == tensor_parallel_size`.

Operational knobs:

* `NRL_REFIT_NUM_STREAMS` (default `2`) — number of CUDA streams the generation side
  uses to overlap per-PP-stage bulk reshards. Having higher number can increase
  concurrency of the transportation when PP-size is large, but will have higher
  memory overhead.

## Design Overview

FFN layers are the dominant payload in the weight transfer. Our profiling shows
that the MoE FFN layers account for 97%-98% of the model weights. To balance
performance and software sustainability, we chose a dual-path strategy for the
nccl-reshard-refit implementation:

* **Bulk path** — the FFN projection weights (`gate_proj` / `up_proj` / `down_proj`
  `.weight`, dense MLP and MoE experts alike; see `is_nccl_reshard_param()`). These are
  resharded shard-to-shard with `xferdtensor` over dedicated NCCL communicators. For
  large models this covers the vast majority of the refit bytes.
* **Misc path** — everything else (embeddings, attention projections, layernorms, the
  MoE router, `lm_head`, FP8 `_scale_inv` siblings, FP8 KV-cache scales, …). These ride
  a packed broadcast (conventional `packed_tensor.py` implementation) over the shared
  `model_update_group` and are loaded on the generation side through the backend's
  regular `load_weights` machinery.

The feature is integrated into the `nemo_rl/weight_sync/` framework:
`create_weight_synchronizer(..., nccl_reshard_refit=True)` returns a
`NcclReshardWeightSynchronizer` whose `init_communicator()` performs the one-time setup
and whose `sync_weights()` runs one refit.

### Execution Flow: Setup Time

`NcclReshardWeightSynchronizer.init_communicator()` runs three steps once, before
training starts:

1. **`init_collective()`** — creates the `model_update_group`, a NCCL group spanning all
   training and generation ranks. The bulk path does not use it; it carries the misc
   packed-broadcast (and FP8 KV-cache scales), identical to the conventional collective
   transport.
2. **`init_nccl_reshard_comm_group()`** — creates the bulk-path communicator(s): **one
   NCCL group per training PP stage**, each spanning that stage's training ranks plus
   *all* generation ranks (non-PP is simply `pp_size == 1`, a single group over
   everything). Keeping the bulk path on its own communicators decouples it from the
   misc broadcast.
3. **`prepare_nccl_reshard_refit_info()`** — the metadata exchange. The **training side
   builds a backend-agnostic description** of every bulk parameter
   (`build_nccl_reshard_refit_info()` in `nemo_rl/weight_sync/nccl_reshard_utils.py`),
   keyed strictly by **HuggingFace parameter names**, and ships it to the generation
   side.

The derived metadata (`nccl_reshard_refit_info`) contains, per parameter:

* `name` — the HF parameter name (per-expert MoE weights are grouped into a single
  `...experts.{gate,up,down}_proj.weight` entry of shape `[num_experts, ...]`, tagged
  with `grouped_expert_proj`);
* `global_shape` and `dtype` of the full, unsharded tensor;
* `src_mesh_info` / `src_placements` — the training-side rank mesh (`MeshInfo`) and
  DTensor-style `Shard`/`Replicate` placements, derived from the training parallelism
  (TP/EP/PP; experts live on an EP mesh, everything else on a TP mesh);
* `dst_mesh_info` / `dst_placements` — the same for the generation side (TP mesh, or an
  EP mesh for experts when vLLM expert parallelism is enabled);
* `pp_stage` — which training PP stage owns the parameter (present when `pp_size > 1`),
  used to route it to the right per-stage communicator.

Alongside it, `misc_meta` (an **ordered** dict of `name -> {shape, dtype}`) describes
every misc parameter; the order is load-bearing because producer and consumer walk it in
lockstep during the packed broadcast.

Finally, both sides build their `hf_to_local_param_map`: a mapping from each bulk HF
parameter name to a `LocalParamSpec(base, pre, post)` describing how that parameter is
realized **locally**:

* On the **training side**, a direct parameter's `base` is the live TP/EP-local shard
  (sent as-is); grouped MoE experts get a `pre` hook that stacks this rank's per-expert
  views into a `[num_local_experts, ...]` tensor fresh at each refit.
* On the **generation side**, a direct parameter's `base` is the live vLLM parameter
  (received into in place); a parameter that is a slice of a fused vLLM tensor (dense
  `gate_up_proj`, grouped-expert `w13`/`w2`) gets a `pre` hook that allocates a receive
  buffer for its region and a `post` hook that copies the received shard back into the
  fused parameter.

### Execution Flow: Refit Time

Every training step (with in-flight weight updates, concurrently with generation),
`NcclReshardWeightSynchronizer.sync_weights()` triggers both sides:

* `pre` contains a function that should be executed in-flight before the refit.
* `post` contains a function that should be executed in-flight after the refit.

* The **training side** walks `per_layer_params`, skipping parameters owned by other PP
  stages. For each parameter it resolves the `LocalParamSpec`, runs `pre` (expert
  stacking) if present, wraps the local shard in a `DTensorRef` (which reports the
  *global* shape while holding only the local tensor), and calls
  `xferdtensor(src, src_mesh, src_placements, None, dst_mesh, dst_placements, group,
  stream)`.
* The **generation side** walks the same metadata in the same order — every rank in a
  comm group must issue the same sequence of transfers. Per-PP-stage parameter groups
  are distributed across `NRL_REFIT_NUM_STREAMS` CUDA streams so different stages'
  reshards overlap. For each parameter it runs `pre` (receive-buffer allocation), calls
  `xferdtensor(None, ..., dst, ..., group, stream)`, then `post` (copy back into the
  fused parameter).

### The Misc Path

After the bulk reshard completes, the misc parameters are transferred.
This part is reusing the same code implementation as the conventional packed_tensor refit.

## Decoupling Backend-Agnostic Parts and Backend-Dependent Parts

To facilitate backend extension, the implementation cleanly separates backend-agnostic
components from backend-dependent ones. As a result, extending to a new backend only
requires implementing the backend-dependent components.

**Backend-agnostic** (no knowledge of Megatron or vLLM):

* `nemo_rl/weight_sync/nccl_reshard_utils.py` — the metadata builder
  (`build_nccl_reshard_refit_info`), mesh/placement derivation (`build_mesh_info`,
  `get_placements`, `MeshInfo`), the bulk-path whitelist (`is_nccl_reshard_param`),
  per-expert grouping into HF-convention grouped entries, the config validator, and the
  `LocalParamSpec`, `RefitCtx`, and `HFToLocalParamMap` contracts. All parameter sharding
  required by the different types of parallelism is handled by this utility.
* `nemo_rl/weight_sync/xferdtensor.py` — the transfer entry point and its transport
  dispatch (see below).
* `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py` and the factory routing —
  the lifecycle orchestration.

The glue that makes this work across backends is the **HF naming convention**: the
training side must describe its parameters using HF names and global shapes, and the
generation side maps those HF names onto whatever its own storage layout is.

**Backend-dependent**:

* **Training side** (`megatron_policy_worker.py`): producing the HF-named state-dict
  metadata; building `hf_to_local_param_map` — resolving each HF name to the local
  Megatron tensor view and providing the grouped-MoE `pre` stacking hook; the
  `init_collective` / `init_nccl_reshard_comm_group` bootstrap methods; the
  `nccl_reshard_refit()` send loop; the misc packed-broadcast producer.
* **Generation side** (`vllm_backend.py`): building `hf_to_local_param_map` — mapping HF
  names onto vLLM's fused parameters (`qkv_proj`, `gate_up_proj`, grouped-expert
  `w13_weight`/`w2_weight`) with `pre`/`post` hooks for the slice regions, which is
  deliberately **shape-driven** so the same code handles generation TP and generation
  EP; the comm bootstrap methods; the `nccl_reshard_refit()` receive loop; the misc
  consumer feeding `load_weights`.

**To extend to a new backend**, the only piece with genuinely new logic is
`build_hf_to_local_param_map`. Everything else is boilerplate that follows a fixed
contract and can be copied from the existing backend almost verbatim.

**The one backend-specific implementation — `build_hf_to_local_param_map`:** resolve
each bulk HF name to your local storage as a `LocalParamSpec` — `base` for tensors
sent/received as-is, and `pre`/`post` hooks wherever your layout requires staging
(fused/merged tensors, layout conversions, grouped-expert stacking). This is the *only*
place your backend's parameter layout is encoded; all cross-mesh byte movement is
already handled by the shared metadata and `xferdtensor`.

(A new *training* backend additionally has to produce the HF-named metadata — names,
global shapes, dtypes, and the parallelism description the agnostic builder consumes —
inside its `prepare_nccl_reshard_refit_info`, since only the backend knows how to read
its own weights. A new *generation* backend simply consumes the shipped metadata.)

**Copy-paste boilerplate** (identical in shape to the existing backend; only
names/attributes change):

1. `prepare_nccl_reshard_refit_info` — restore the shipped metadata and call
   `build_hf_to_local_param_map` once.
2. The communicator bootstrap (`init_collective`, `init_nccl_reshard_comm_group`) — the
   same `StatelessProcessGroup` setup; the only requirement is the rank convention:
   training ranks first (per-stage-local for the bulk groups), generation ranks after.
3. The `nccl_reshard_refit()` loop — walk `per_layer_params` in metadata order (grouped
   by `pp_stage` across `NRL_REFIT_NUM_STREAMS` streams), resolve each `LocalParamSpec`,
   run `pre`, call `xferdtensor`, run `post`. It only touches the generic spec/metadata
   contracts, never your layout.
4. The misc producer/consumer — reuses the conventional packed-broadcast path.

## `xferdtensor` Transports

`xferdtensor()` (in `nemo_rl/weight_sync/xferdtensor.py`) is the single entry point both
workers call. It has the 8-argument signature

```python
xferdtensor(src_tensor, src_mesh, src_placement,
            dst_tensor, dst_mesh, dst_placement,
            process_group, stream=None)
```

and dispatches to one of two transports:

* **Core NCCL reshard** — the reshard operation provided by the **nccl4py
  wrapper** (`nccl.m2n.reshard`). When the package is accesible, this is the default:
  the local shards, mesh rank grids, and placements are handed to the NCCL library,
  which executes the cross-mesh redistribution natively.
* **`xferdtensor_python_impl`** (`nemo_rl/weight_sync/xferdtensor_python.py`) — a pure
  Python + nccl4py-collectives **backup implementation** for environments without a
  proper NCCL / nccl4py reshard installation. It computes the exact shard overlaps
  between the source and destination layouts, moves each destination region once via
  batched point-to-point (with striped receives across replica groups), and fans out to
  replicas with cached split-communicator broadcasts. It is a drop-in with the same
  signature and is selected automatically when `nccl.m2n` is not importable.
* **`xferdtensor_golden`** (`nemo_rl/weight_sync/xferdtensor.py`) — a pure function-only
  implementation intended for debugging. This implementation simply broadcasts the full
  tensor to the destination ranks, which then discard the unused parts. While not performant,
  it guarantees functionally correct outputs.

Both transports honor the `stream` argument so the transfer is ordered with the caller's
`pre`/`post` staging work on one CUDA stream.
