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
  policy.generation.refit_transport=nccl_reshard
```

At setup, `check_nccl_reshard_refit_support()` validates the configuration and raises a
single `ValueError` listing every violation. The current requirements are:

* **Non-colocated only** — `policy.generation.colocated.enabled=false`. The colocated
  path uses IPC and is unaffected by this feature.
* **Megatron training backend** — `policy.megatron_cfg.enabled=true` (the DTensor
  training backend is not supported yet.).
* **vLLM or Megatron generation backend** — `policy.generation.backend` must be
  `vllm` or `megatron` (SGLang and TRTLLM are not supported yet).
* Training-side Megatron supports expert tensor parallelism when the generation
  destination is Megatron. A vLLM destination still requires
  `expert_tensor_parallel_size: 1`. Custom PP layouts
  (`pipeline_model_parallel_layout`, virtual PP > 1, embedding/loss
  pipeline-split accounting) are not supported yet.
* **Generation-side ETP with `inference_optimized` is pinned to 1.** Those MoE
  layers do not implement expert tensor parallelism and raise whenever the
  *resolved* ETP exceeds 1 — and an omitted ETP resolves to TP, not 1. So
  `merged_inference_megatron_cfg` pins generation-side
  `expert_tensor_parallel_size` to 1 for that transformer implementation, which
  is what lets generation-side TP > 1 work; the reshard then handles the
  train-ETP → gen-ETP=1 gather. Other transformer implementations retain their
  configured ETP. An explicitly requested generation-side ETP > 1 with
  `inference_optimized` is rejected by config key name rather than surfacing as
  a raw MCore assert at model build.
* **Bulk precision is independent of source and destination storage.** Supported
  Transformer Engine blockwise FP8 and MXFP8 training parameters are dequantized
  using their own quantization metadata. BF16 logical weights are then sent to
  BF16, blockwise FP8, or MXFP8 destination hooks. The destination derives its
  own weight scales when requantizing; training-side scales are not a wire format.
  vLLM blockwise destinations use FP32 inverse scales and the destination's effective
  block grid, including refined MoE grids. Fused gate/up partitions must align with
  that grid; incompatible runtime scale layouts are rejected. Megatron blockwise destinations
  use Transformer Engine storage, while `inference_optimized` supports MXFP8.
  The unchanged misc path still limits blockwise FP8 parameter storage on the
  training side to blockwise FP8 vLLM destinations (`precision='fp8'`, `is_mx=false`):
  misc weights retain their physical FP8 values and inverse scales. This restriction
  comes from misc loading, not from the bulk BF16 transport.
* Megatron generation accepts BF16 or supported Transformer Engine FP8 training
  parameter storage, including blockwise FP8 and MXFP8 with `fp8_param=true`.
  Quantized sources are materialized as logical BF16 for transport; the
  destination either stores the logical weight or quantizes each complete local
  weight into its supported local FP8 storage. Before every refit, an explicit
  parameter sync materializes optimizer
  updates that would otherwise wait for the next overlapped all-gather. When
  MXFP8 parameter all-gather reuses the gradient buffer, that aliased allocation
  stays GPU-resident across refit so
  persistent DDP/autograd views remain valid; ordinary gradient buffers and
  optimizer state are offloaded only when
  `policy.generation.mcore_generation_config.offload_policy_before_refit` is true.
* **Every bulk transfer uses canonical HF layout and `torch.bfloat16`.** This
  contract applies to the source `ctx.buf`, advertised metadata, and receiver
  buffer, including MXFP8 train → MXFP8 gen. It deliberately trades wire size
  and repeated dequantization for a stable interface across storage formats.
  For example, TE and MCore MXFP8 use different representations, and MCore's
  padded, swizzled scale layout cannot be resharded as logical HF weights.
  Precision-specific conversion belongs in the hooks. This BF16 requirement
  does not change the misc path or other refit transports.
* BF16 FlashInfer TRTLLM MoE is supported through vLLM's native
  layerwise-reload path. Its grouped expert weights must use expert-parallel
  destination sharding with linear expert placement; tensor-sharded expert
  destinations and round-robin placement are rejected. This path does not
  support an FP8 KV cache or a co-trained MTP drafter; setup rejects both
  combinations.
* vLLM expert parallelism is supported with the NeMo RL convention
  `expert_parallel_size == tensor_parallel_size`.
* Megatron generation supports expert parallelism; generation-side expert tensor
  parallelism is available only when `transformer_impl` is not `inference_optimized`.
* Megatron generation uses the same top-level selector as other backends:
  `refit_transport=null` selects NeMo-RL's packed collective,
  `refit_transport=mcore` selects Megatron Core's native refit, and
  `refit_transport=nccl_reshard` selects M-to-N. `refit_backend` is consulted
  only for `refit_transport=mcore`. Colocated Megatron generation requires
  `refit_transport=mcore`, because its refit is carried by the in-place
  wake-reshard; the other transports are rejected rather than silently ignored.
* **Generation-side PP > 1 is not supported by this refit transport yet.**
  Megatron-Core and vLLM can run generation with PP, and training-side Megatron
  PP is supported here; the missing piece is generation-stage-aware destination
  routing in `nccl_reshard`.
* **No ModelOpt real quantization** — `policy.generation.real_quant=false`. Real-quant
  rollouts refit through vLLM's layerwise-reload weight loaders, which the bulk
  `xferdtensor` writes bypass.

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
  large models this covers the vast majority of the refit bytes. For the current version of implementation, it only detects `(experts).N.{gate_proj|up_proj|down_proj}` as the subject of this performant transportation path. The coverage will be expanded via future updates.
  Two FFN-named groups are explicitly excluded and ride the misc path instead:
  shared-expert weights (`*.shared_expert.*`, which fuse differently on the vLLM
  side) and co-trained MTP drafter weights (which vLLM keeps in a separate
  drafter module updated through `load_weights`). Co-trained MTP is not supported
  with BF16 FlashInfer TRTLLM; this routing applies to other supported backend
  combinations. MTP weights are recognized two ways: bare-`mtp.`-prefix HF names
  (NemotronH, Qwen3.5) via
  `is_nccl_reshard_param()`, and DeepSeek-style MTP exported as trailing
  `model.layers.N` indices via provenance — the Megatron-side name carries an
  `mtp.` module segment (bare for LM bridges, `language_model.mtp.*` for the VL
  and EXAONE bridges), so the worker excludes those HF layers when building the
  metadata (`_collect_mtp_hf_layer_names()`).
* **Misc path** — everything else (embeddings, attention projections, layernorms, the
  MoE router, `lm_head`, scales belonging to misc weights, FP8 KV-cache scales, …). FP8
  KV-cache scales are supported only by backend combinations that allow an FP8 KV
  cache; BF16 FlashInfer TRTLLM rejects that configuration at setup. These tensors
  ride a packed broadcast (conventional `packed_tensor.py` implementation) over the
  shared `model_update_group` and are loaded on the generation side through the
  backend's regular `load_weights` machinery.

Weight-quantization scale exports associated with bulk weights are excluded from
misc metadata and transfer. Their source values have already been consumed during
dequantization; sending them again could overwrite the scales produced by destination
requantization. The exclusion follows the weights actually selected for bulk transfer,
not a blanket scale-name filter. Scales for misc weights and KV-cache scales retain
their existing handling and dtypes.

The feature is integrated into the `nemo_rl/weight_sync/` framework. For vLLM,
`create_weight_synchronizer(...)` returns an `NcclReshardWeightSynchronizer` directly.
For Megatron generation, the existing `MegatronWeightSynchronizer` retains ownership of
the inference-engine lifecycle and delegates only the transfer to an
`NcclReshardWeightSynchronizer`.

### Execution Flow: Setup Time

`NcclReshardWeightSynchronizer.init_communicator()` runs three steps once, before
training starts:

Bulk preparation always describes logical BF16 weights, regardless of the generation
backend. There is no payload-mode negotiation. Bulk conversion tasks are kept separate
from the existing misc export tasks so changing the bulk contract does not change
attention, embedding, router, or other misc payloads. Megatron workers are assigned
an explicit source or destination refit role and expose the same
`prepare_refit_info`, `build_hf_to_local_param_map`,
`prepare_nccl_reshard_refit_info`, and `nccl_reshard_refit` entry points in either
role.

1. **`init_collective()`** — creates the `model_update_group`, a NCCL group spanning all
   training and generation ranks. The bulk path does not use it; it carries the misc
   packed-broadcast, including FP8 KV-cache scales for backend combinations that support
   them, identical to the conventional collective transport.
2. **`init_nccl_reshard_comm_group()`** — creates the bulk-path communicator(s): **one
   NCCL group per training PP stage**, each spanning that stage's training ranks plus
   *all* generation ranks (non-PP is simply `pp_size == 1`, a single group over
   everything). Keeping the bulk path on its own communicators decouples it from the
   misc broadcast.
3. **`prepare_nccl_reshard_refit_info()`** — the metadata exchange. The **training side
   builds a backend-agnostic description** of every bulk parameter
   (`build_nccl_reshard_refit_info()` in `nemo_rl/weight_sync/nccl_reshard_utils.py`),
   keyed strictly by **HuggingFace parameter names**, and ships it to the generation
   side. Before shipping, `make_nccl_reshard_refit_info_wire_safe()` converts the
   `MeshInfo` rank tensors and `Shard`/`Replicate` placements into plain dicts/lists —
   Megatron patches torch's storage unpickler, so raw tensor pickles would require
   `import megatron` inside the vLLM worker. The generation side rebuilds the objects
   with `restore_refit_info_placements()`.

The derived metadata (`nccl_reshard_refit_info`) contains, per parameter:

* `name` — the HF parameter name (per-expert MoE weights are grouped into a single
  `...experts.{gate,up,down}_proj.weight` entry of shape `[num_experts, ...]`, tagged
  with `grouped_expert_proj`);
* `global_shape` in canonical HF layout and `dtype=torch.bfloat16` for the full,
  unsharded logical tensor;
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

* On the **training side**, each `pre(base)` independently reads current local
  parameter values, dequantizes quantized storage, selects the corresponding HF
  slice, and produces a contiguous BF16 buffer. Selection follows dequantization
  so fused gate/up slices use the complete source's quantization metadata.
  Grouped MoE hooks materialize their ordered expert members and stack them into
  `[num_local_experts, ...]`. There is no shared source cache or cache lifecycle
  in the transfer loop; repeated preparation observes parameter updates immediately.
* On the **generation side**, canonical BF16 storage can receive in place. Other
  storage uses `pre` to allocate BF16 staging and `post` to cast, assemble, or
  quantize the received values. Fused gate/up and grouped expert layouts retain
  the assembly needed to commit complete local weights. Quantized commits update
  weight values and their associated scales together. BF16
  FlashInfer TRTLLM grouped experts instead receive into canonical EP-local staging
  tensors; `post` loads each logical expert with its global expert ID through vLLM's
  native weight loader.

### Execution Flow: Refit Time

Every training step (with in-flight weight updates, concurrently with generation),
`NcclReshardWeightSynchronizer.sync_weights()` triggers both sides:

* `base` identifies local storage; it is not necessarily the transfer buffer.
* `pre(base)` prepares a `RefitCtx` whose `buf` is a canonical HF BF16 shard.
  Omitting `pre` is valid only when `base` already satisfies that contract.
* `post(ctx)` commits received values to local storage, including quantization and
  scale updates. It runs after the transfer on the corresponding CUDA stream.

* The **training side** walks `per_layer_params`, skipping parameters owned by other PP
  stages. For each parameter it resolves the `LocalParamSpec`, runs `pre`
  (dequantization or expert stacking) if present, wraps the local shard in a
  `DTensorRef` (which reports the *global* shape while holding only the local tensor),
  and calls
  `xferdtensor(src, src_mesh, src_placements, None, dst_mesh, dst_placements, group,
  stream)`.
* The **generation side** walks the same metadata in the same order — every rank in a
  comm group must issue the same sequence of transfers. Per-PP-stage parameter groups
  are distributed across `NRL_REFIT_NUM_STREAMS` CUDA streams so different stages'
  reshards overlap. For each parameter it runs `pre` (receive-buffer allocation), calls
  `xferdtensor(None, ..., dst, ..., group, stream)`, then `post` (copy back into the
  fused parameter, requantize local storage, or load staged TRTLLM experts). After every transfer completes, the
  TRTLLM path finalizes vLLM's native layerwise reload once to restore the packed runtime
  layout.

### The Misc Path

After the bulk reshard completes, the misc parameters are transferred.
This part is reusing the same code implementation as the conventional packed_tensor refit.
Its representation is unchanged. Only source scale entries belonging to BF16 bulk
weights are removed; the receiver must retain its newly generated scales for those
weights. This also keeps unrelated weight scales and KV-cache scales on their existing
load path.

Quantized destination handling added for BF16 bulk weights stays in the bulk hooks.
Megatron's packed collective imports, including `refit_transport=null` and misc
weights, retain their existing destination `copy_` behavior. Likewise, vLLM bulk
hooks explicitly opt into rectangular FP8 blocks; the shared quantizer keeps its
square-block default for legacy callers.

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
  Megatron tensor view and providing `pre` hooks for quantized-source
  independent dequantization, BF16 conversion, and grouped-MoE stacking; the
  `init_collective` / `init_nccl_reshard_comm_group` bootstrap methods; the
  `nccl_reshard_refit()` send loop; the misc packed-broadcast producer.
* **Generation side** (`vllm_backend.py`): building `hf_to_local_param_map` — mapping HF
  names onto vLLM's fused parameters (`qkv_proj`, `gate_up_proj`, grouped-expert
  `w13_weight`/`w2_weight`) with `pre`/`post` hooks for slice regions or canonical
  TRTLLM staging, which is deliberately **shape-driven** so the same code handles
  supported generation parallelism; the comm bootstrap methods; the
  `nccl_reshard_refit()` receive loop; the misc consumer feeding `load_weights`; and
  backend-specific finalization after all weights arrive.
* **Megatron generation side** (`megatron_worker.py`): mapping the same canonical
  HF FFN shards to local fused dense/expert views. BF16 destinations receive in
  place; quantized destinations use short-lived BF16 staging buffers, assemble
  fused components, and quantize into persistent storage with fresh scales.
  Misc weights continue through Megatron Bridge's
  packed-broadcast import path.

**To extend to a new backend**, provide a destination map from canonical HF weights
to that backend's local storage. Both backends implement this as
`build_hf_to_local_param_map`; Megatron derives its targets from Bridge conversion tasks.
Everything else follows the fixed transport contract.

**The one backend-specific implementation — `build_hf_to_local_param_map`:** resolve
each bulk HF name to your local storage as a `LocalParamSpec` — `base` for tensors
sent/received as-is only when already canonical BF16, and `pre`/`post` hooks wherever
the local representation requires staging (quantization, fused/merged tensors, layout
conversions, grouped-expert stacking). Backends that
rebuild runtime storage may also need one transport-level finalizer after all specs have
run. These are the only places the backend's parameter layout is encoded; all cross-mesh
byte movement is already handled by the shared metadata and `xferdtensor`.

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


## Expected Performance

| Platform | Model | Precision | Train → Gen mapping | XferDTensor fraction | Refit time |
|--|---|---|---|---:|---:|
| H100 | QWEN3 4B (dense) | BF16 | DP8 → TP8 | 66.9% | 0.21–0.34s |
| H100 | QWEN3 30B | BF16 | EP8×PP2 → TP8×DP2 | 95.0% | 0.74–1.00s |
| H100 | QWEN3 30B | FP8 | EP8×DP2 → TP2×DP8 | 93.0% | 2.90–4.00s |
| H100 | DSV3 | BF16 | PP16×EP16 → TP32×DP8 | 97.6% | 2.39s-2.83s |
| H100 | QWEN3.5 397B | BF16 | TP8xPP8xEP32 -> TP16xDP16 | 97.4% | 1.97s-2.17s |
| GB200 | DSV3 | BF16 | PP16×EP16 → TP32×DP8 | 97.6% | 1.93s-2.59s |
| GB200 | Nemotron Ultra-v3 | BF16 | TP8xEP32xPP2 -> TP8xDP8 | 93.8% | 2.32s |

The feature supports both dense and MoE models. The table above shows the `XferDTensor fraction`, which is the proportion of the refit payload that utilizes the high-performance `bulk` transfer path. As the model size increases, this fraction becomes higher, which is the key to provide a scalable refit time to large models. For FP8 models, the efficiency is currently lower compared to BF16 models.
