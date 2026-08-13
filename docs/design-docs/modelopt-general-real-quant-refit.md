# General ModelOpt Real-Quant Refit

Status: proposal. This design is a prerequisite for adding ModelOpt real-quant
KV-cache refit. It generalizes GEMM real quantization first. Until this design
is implemented, [ModelOpt Real-Quant Refit Architecture](modelopt-real-quant-architecture.md)
describes the current W4A4/W4A16 path.

## Problem

The current NeMo RL real-quant path supports only ModelOpt NVFP4 W4A4 and
W4A16. It works by reproducing deployment behavior in several repositories:

- NeMo RL parses the training recipe into a two-value mode enum and constructs
  a vLLM `quantization_config` by hand.
- Megatron-Bridge discovers quantizer state, derives NVFP4 scales, packs
  weights, and rewrites expert layouts with format-specific code.
- NeMo RL registers custom vLLM quantization classes and translates fused-MoE
  payloads into names and layouts accepted by vLLM.
- NeMo RL selects internal vLLM reload roots and maintains format-specific
  post-load state across refits.

This duplicates ModelOpt export semantics and vLLM checkpoint-loading
semantics. Even another ModelOpt/vLLM format would require another set of
changes across the same layers.

ModelOpt already exports these formats to Hugging Face checkpoints, and vLLM
already loads a subset of those ModelOpt checkpoint formats. The online refit
path should connect those two existing contracts rather than implement a
parallel format.

## Goals

1. Use ModelOpt as the only owner of quantization-format detection, scale
   derivation, packing, tensor suffixes, and deployment quantization config.
2. Use vLLM's native ModelOpt checkpoint loaders and runtime post-processing.
3. Keep Megatron-Bridge as the only owner of Megatron-to-HF mapping and
   TP/PP/EP redistribution.
4. Keep NeMo RL format-agnostic. A new format that fits the same ModelOpt export
   and native vLLM load contracts should not require a new NeMo RL mode or
   quantization class.
5. Preserve the trainable fake-quant policy. Online export must not replace its
   parameters, register deployment buffers on it, or otherwise mutate it.
6. Support dense, routed-MoE, and hybrid models through the same canonical HF
   tensor stream.
7. Preserve repeated refit and CUDA-graph storage stability.

## Non-goals

- Adding kernels or checkpoint formats that vLLM does not support.
- Real-quant KV-cache support. It follows after the GEMM path is general.
- Generalizing real-quant refit to SGLang or TensorRT-LLM in this change.
- Replacing NeMo RL's transport protocols.
- Calling a disk checkpoint exporter during every training step.

## Support definition

The first implementation supports ModelOpt NVFP4 W4A4 and W4A16. The API is
format-neutral so another format can be added without changing NeMo RL, but
broad format coverage is not an initial goal. A format is supported only when
it is in this intersection:

1. ModelOpt can export the quantized policy state to its canonical HF
   deployment representation.
2. vLLM can construct and execute a model from that representation.
3. Megatron-Bridge can map the model's Megatron weights and quantizer sidecars
   to the corresponding canonical HF tensors.
4. Online refit passes the parity and repeated-refit gates in this document.

NeMo RL must not maintain a second list of format names. Other ModelOpt formats
remain unsupported until they pass the same contract without format-specific
MBridge or NeMo RL workarounds. A format-specific requirement is a reason to
defer that format, not to complicate the common interface.

## Rejected approaches

### Invoke `export_hf_checkpoint` for every refit

ModelOpt's high-level HF exporter owns model preparation, module mutation,
state-dict generation, and filesystem output. Its per-weight implementation
replaces the source parameter with packed bytes and registers scale buffers on
the module. That is correct for a terminal checkpoint export and incorrect for
a policy that must continue training.

It also creates a full checkpoint boundary where the online path needs a lazy
named-tensor iterator. Wrapping it with temporary files or restoring the model
afterward would be slow, memory-heavy, and fragile.

### Add more format branches to Megatron-Bridge and NeMo RL

A registry in NeMo RL would make the branches look cleaner but would not
remove duplicated scale and packing semantics. Each upstream format change
would still require coordinated downstream edits.

### Invoke ModelOpt's `GPTModelExporter` directly

ModelOpt's Megatron exporter also owns model-architecture mappings,
distributed collection, full state dictionaries, and checkpoint writing. That
would create a second Megatron-to-HF mapping system beside Megatron-Bridge and
currently covers fewer formats and model structures than the desired
intersection. Reusing its low-level quantization core is useful; making it the
online bridge is not.

### Build a temporary HF model for every refit

Materializing an HF model, copying full weights into it, and running the normal
ModelOpt exporter would duplicate model memory and reintroduce full-model
collection. It does not scale to the target MoE models.

### Send Megatron shards directly to vLLM

Policy and rollout TP/EP layouts can differ. The transport contract must remain
topology-independent, so the handoff representation is the canonical HF
checkpoint layout, not either runtime's local shards.

## Chosen architecture

The handoff is a canonical ModelOpt HF checkpoint represented as the
`config.json`-style `quantization_config` preferred by current vLLM plus a lazy
iterator of named tensors. Do not introduce a NeMo-only config or use the
legacy `hf_quant_config.json` shape as the online contract. No checkpoint is
written to disk.

```text
ModelOpt QAT policy
  | capture quantizer export state (non-mutating)
  v
Megatron-Bridge
  | map and redistribute weight plus quantizer sidecars to canonical HF layout
  | invoke ModelOpt pure packing and config builders
  v
NeMo RL transport
  | descriptor once, named tensors on each refit
  v
vLLM native ModelOpt config + checkpoint loader + layerwise reload
```

### ModelOpt: pure export core

Refactor the existing HF exporter so its format logic is available through a
public, non-mutating API. The exact names can follow ModelOpt conventions, but
the API needs these operations:

1. Capture immutable export state for one quantized weight from its owning
   module and quantizers.
2. Capture the scalar amax values used by the initial W4A4/W4A16 scope.
3. Given a canonical floating-point weight and canonicalized export state,
   return the packed checkpoint weight and its scale tensors.
4. Build the deployment `quantization_config` from the canonical per-layer
   export records.

A representative contract is:

```python
state = capture_quantized_weight_state(module, weight_name="weight")
exported = export_quantized_weight(weight, state, dtype=torch.bfloat16)
quant_config = build_hf_quantization_config(layer_records)
```

`exported` is an ordered mapping of relative checkpoint names such as
`weight`, `weight_scale`, `weight_scale_2`, and `input_scale`. Callers do not
derive these names.

The initial API is deliberately limited to one scalar weight amax and, for
W4A4, one scalar input amax per logical weight. A scalar is max-reduced across
TP shards and copied to every HF projection produced from that logical weight.
Experts remain separate logical weights until after conversion, which preserves
distinct expert calibration without an axis-description language. A quantizer
with per-channel, per-block, per-projection, or packed multi-expert amax state
is rejected rather than inferred from tensor shape.

This narrow contract is sufficient for the initial dynamic NVFP4 W4A4/W4A16
scope and is simpler than publishing unused generic sidecar metadata. A future
format that needs structured sidecars must first extend the public ModelOpt API
with concrete semantics and tests. Relative checkpoint suffixes remain
ModelOpt-owned and are added only after MBridge has produced each canonical HF
weight.

The existing high-level HF exporter then becomes a consumer of the same pure
core: it applies the returned tensors to the terminal export model and writes
them. This parity prevents the online and offline formats from drifting.

### Megatron-Bridge: topology and naming

Megatron-Bridge owns:

- locating the weight and owning ModelOpt quantizers in Megatron modules;
- applying existing Megatron-to-HF conversion tasks;
- splitting fused QKV and gated projections;
- gathering TP, PP, and EP shards;
- max-reducing each scalar amax over the source weight's TP group and preserving
  one state per logical expert;
- invoking the ModelOpt pure exporter after the weight and sidecars are in
  canonical HF layout; and
- yielding the canonical named tensors lazily.

Megatron-Bridge must not derive a scale formula or branch on `nvfp4`, `fp8`, or
`mxfp8`. It may reject a mapping whose sidecar layout it cannot represent, but
that error must identify the mapping and layout rather than inventing a new
format allowlist.

The current `quant_mode` argument is removed. The actual quantizer graph is the
source of truth. Mixed-precision models naturally produce different export
states for different layers.

For MoE, use the canonical HF names produced by the model's normal ModelOpt
export. Any expert batching is a transport optimization and must not create a
NeMo-only checkpoint schema. The first implementation keeps experts separate
through packing; packing before EP gather remains a future optimization.

### NeMo RL: orchestration only

For real quantization, the policy quantizer graph is authoritative. The rollout
worker must not independently infer a deployment mode from a second recipe.

The correctness rule is that the authoritative deployment config must exist
before quantized vLLM model construction. The first implementation obtains it
from the initialized policy quantizer graph. A future ModelOpt API could
pre-resolve the exact same descriptor and recover parallel initialization, but
that optimization is outside this change; NeMo RL must not implement a second
recipe resolver. Initialization is therefore:

1. Reserve generation resources and server ports as needed.
2. Initialize the quantized policy.
3. Ask the policy/Megatron-Bridge path for the small deployment descriptor.
4. For colocated execution, offload the policy before vLLM measures available
   memory and allocates its cache.
5. Put the descriptor's `quantization_config` into vLLM's HF overrides.
6. Construct vLLM with its native ModelOpt quantization method.
7. Restore the policy as required, prepare the canonical tensor manifest, and
   perform the first refit.

This makes real-quant startup sequential even for non-colocated policy and
generation workers. That is a deliberate correctness tradeoff:
exact mixed precision and model-specific exclusions cannot be inferred by a
downstream recipe parser. Non-real-quant startup keeps the current parallel
path. Colocated startup must preserve the current memory contract: vLLM still
observes a policy-free GPU when it sizes its cache, and restoring the policy
must fit under the configured vLLM memory limit.

NeMo RL owns orchestration, its existing IPC/NCCL tensor transport, timeout
policy, manifest validation, and calls that delimit a refit transaction. vLLM
owns checkpoint loading and the native layerwise reload lifecycle. NeMo RL does
not own packing, scale names, MoE checkpoint rewrites, quantization classes, or
format-specific runtime state.

vLLM 0.25.1 has no public abort operation that restores a partially processed
checkpoint update. Any exception after `start_weight_update` is therefore
fatal to that engine instance. NeMo RL must not attempt to reuse or repair its
private reload state. The quantized generation worker shuts down that instance
and propagates the failure; a new worker construction is required before a
later refit can run.

This first implementation supports only NeMo RL's default colocated IPC or
non-colocated collective refit path. Sparse, checkpoint-engine, and
`nccl_reshard` transports do not consume the canonical packed ModelOpt stream
and are rejected during configuration.

The deployment descriptor is initially just the canonical ModelOpt HF
`quantization_config`. Do not introduce a parallel NeMo-specific schema: the
ModelOpt producer metadata in that config identifies the export version, and
vLLM already validates the algorithm and fields it consumes. The tensor
manifest remains separate because shapes and dtypes are already prepared by
the existing refit path.

### vLLM: native load and reload

vLLM owns:

- parsing the ModelOpt deployment config;
- allocating checkpoint-layout tensors;
- loading canonical HF names;
- selecting kernels;
- post-load conversion into runtime layouts; and
- preserving runtime tensor addresses across repeated refits.

NeMo RL keeps its established IPC/NCCL tensor transport and calls vLLM
0.25.1's public `start_weight_update` and `finish_weight_update` methods around
that stream. Between those boundaries, the normal vLLM checkpoint loader
consumes each canonical named-tensor batch. The configured vLLM weight-transfer
engine supplies the native layerwise setup and finalization; NeMo RL does not
use that engine's transport receive method. This limited integration avoids
replacing NeMo RL's transport in the same change while still removing all
private layerwise-reload calls.

NeMo RL must not import vLLM's layerwise-reload helpers, inspect reload metadata
or roots, or retain method-owned kernel objects. The current custom W4A4 and
W4A16 quantization registrations should be retired only after their native
vLLM replacements pass the gates below.

Before deleting an adapter, add focused vLLM characterization tests through
the public lifecycle and normal checkpoint loader for both NeMo IPC and NCCL
transports. Separately characterize vLLM's complete public weight-transfer API
to detect native defects, but do not replace NeMo RL's transport as part of
this refactor. If the native ModelOpt loader fails repeated refit, fix it behind
vLLM's public API rather than copying the loader or calling private reload APIs
from NeMo RL. Delete the NeMo adapter only after the equivalent vLLM fix is
merged, the NeMo vLLM dependency includes it, and the characterization tests
pass against that dependency. A vLLM gap that cannot be fixed cleanly leaves
that model or format unsupported in the first landing.

Megatron policy EP greater than one remains required: MBridge must produce a
topology-independent canonical HF stream from an EP-sharded policy. vLLM
rollout expert parallelism is not part of the first landing. Rollout ranks keep
the current full-expert ownership until native vLLM EP reload has a separate
parity test and explicit support decision.

## Configuration contract

`policy.quant_cfg` and the resulting ModelOpt quantizer graph remain the
quantization source of truth. For real-quant rollout, the deployment config is
captured from the initialized policy graph. The following generation-side
concepts are removed:

- an independently parsed W4A4/W4A16 mode;
- hand-authored `real_quant_ignore` patterns; and
- a requirement to duplicate the policy recipe under generation solely to
  infer the deployment format.

`generation.real_quant: true` remains the explicit opt-in. A generation recipe
may remain for fake-quant rollout, but it is not used to synthesize a
real-quant checkpoint config.

## Invariants and failures

The implementation must enforce these invariants:

1. Capturing a descriptor or exporting a refit does not mutate policy
   parameters, quantizer enablement, buffers, or optimizer-visible objects.
2. Online exported tensor names, shapes, dtypes, and values match ModelOpt's
   offline HF export for the same fixed model state.
3. The deployment descriptor is generated from the same quantizer graph that
   exports the tensors.
4. Every canonical tensor is either loaded exactly once or is explicitly
   excluded by the descriptor.
5. A second refit runs all required post-load processing, preserves runtime
   storage addresses, and changes outputs when source weights change.
6. An unsupported combination fails before training with the first component
   that lacks the capability. NeMo RL does not silently downgrade it.

## Validation gates

### ModelOpt parity

For every initially supported format, compare the pure export core with the
existing offline HF exporter on identical fixed weights and quantizer state.
Require identical tensor keys, shapes, dtypes, and values. Verify that repeated
pure export is deterministic and leaves the source module unchanged.

### Megatron-Bridge parity

Compare streamed MBridge output with an offline ModelOpt HF checkpoint for:

- dense column- and row-parallel linears;
- fused QKV and gated MLP mappings;
- sequential and grouped MoE;
- TP greater than one; and
- Megatron policy EP greater than one with full-expert vLLM rollout ranks.

Start with one format per distinct sidecar layout, then cover every format in
the supported vLLM intersection.

### vLLM refit parity

Using the same exported checkpoint tensors and deterministic prompts, compare:

1. normal vLLM disk load;
2. dummy construction followed by the online refit; and
3. a second online refit after a controlled weight change.

The first two must match within the kernel-appropriate tolerance. The third
must reflect the changed weights while preserving CUDA-graph-referenced tensor
addresses. Inspect the target runtime tensors and scales directly; successful
generation alone is not sufficient.

### NeMo RL smoke tests

Run purpose-specific smoke tests only after the parity gates pass:

- a small dense model for each distinct vLLM ModelOpt loader family;
- Qwen3-30B-A3B for routed MoE; and
- Nano3 for the hybrid MoE/Mamba path.

Each smoke test must complete initialization, two refits, generation, and one
training step with finite nonzero loss where the teacher and student differ.
Long training is not an acceptance test for the export contract.

## Migration result

This is a clean replacement, not a backward-compatible migration. After the
migration, NeMo RL should no longer contain:

- `NVFP4RealQuantMode` or `resolve_nvfp4_real_quant_mode`;
- a hand-built ModelOpt NVFP4 vLLM config;
- custom `nemo_modelopt_nvfp4` or `nemo_modelopt_w4a16_nvfp4` classes;
- fused-MoE ModelOpt checkpoint-name rewriting; or
- format-specific scale packing logic.

Configurations using the removed private mode names or duplicated generation
recipe fields must be updated; no compatibility aliases or translation shim
are retained.

Megatron-Bridge retains distributed conversion code but loses NVFP4 scale
math and the `quant_mode` switch. ModelOpt's offline and online exports share
one pure implementation, and vLLM receives the same checkpoint contract it
already supports from disk.
