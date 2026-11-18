# Proxy enablement

Stage 2 of [the workflow](SKILL.md#stages-and-gates): make GRPO run end to end on a cheap slice of the real checkpoint, so that every NeMo-RL integration path (config loading, generation, refit, logprobs, training step, checkpointing) is exercised before the first full-model allocation.

## Check the reference evidence

Before writing NeMo-RL code, read the validation section of the AutoModel PR that added the model and the vLLM PR that added inference support. Record in the bring-up notes:

| Question | Where to look | If missing |
|---|---|---|
| Does AutoModel match the official or HF reference, and at what KL? | AutoModel PR "Validation" / "Measurements": logits KL, top-1 agreement, per-layer hidden-state error on the first released layers | Record as a risk; a high `gen_kl_error` later starts triage on the AutoModel side |
| Was that evidence produced at the commit NeMo-RL pins? | PR head commit vs the submodule commit; later fixes to the model directory | Re-read the model's commit history; treat evidence from an older head as historical |
| Which precision policy does the reference use (FP32 islands, gate precision, quantized KV or experts)? | AutoModel model config and PR text; vLLM model file and quantization config | Ask the AutoModel and vLLM authors; most precision differences are harmless, but a module the reference keeps in FP32 may be FP32-sensitive |
| Which vLLM features does the model need (`language_model_only`, mamba cache mode, eager vs graph, custom attention backend)? | vLLM PR description and model file | Start from the vLLM PR's own example arguments, with CUDA graphs on |

AutoModel's comparison covers reference vs AutoModel only. AutoModel vs vLLM is first measured by the smoke run below.

## Build the proxy model

A proxy is a slice of the **released** checkpoint, not a random small config: real weights, real geometry, real precision. It exists to exercise every code path at the lowest cost.

Choosing the layers:

- **Homogeneous layer schedule** (every decoder layer is the same type, or types alternate with a short period): keep the bottom contiguous layers, enough to cover one full period. Four is the usual count; AutoModel PRs validate on the same first four released layers. When the first layers differ from the rest (dense MLP layers before the MoE layers), extend the slice past them so it still holds a full period of the repeated layers: Step-5-Preview has dense layers 0-2 and a 3:1 sliding-window to full-attention pattern, so its proxy kept layers 0-7.
- **Heterogeneous schedule** (several layer types, layers that produce state other layers consume, cross-layer KV or index sharing, layer-specific lookups): keep at least one layer of every type, every producer together with its consumers, and the adjacency that the state passing relies on. Read the source-layer fields in both the AutoModel config and the vLLM model file: they decide which layer owns shared state, and a prefix cut breaks the ownership. The slice may be much longer than four layers; renumber the owner fields to the new indices. Write a table of kept layers: new index, source index, layer type, which state it owns or consumes. Example: DeepSeek-V4.1-Flash kept 12 of 40 source layers (0-3, 14-15, 20-21, 24-25, 28-29) to cover SWA, encoder Full/Reuse, decoder Full/Reuse/Reindex and a state refresh.
- **Non-decoder components** (embeddings, LM head, MTP heads, vision tower, n-gram or lookup tables, hyper-connection parameters) are kept unchanged by default. Remove a component only when the RL path does not use it (vision for a text-only recipe, MTP when speculative decoding is off) and both AutoModel and vLLM can be configured to build the model without it; keep any shell parameters the unified loader expects. A text-only recipe is not enough: AutoModel may build the whole multimodal model and then fail on the missing tower weights (MiMo-V2.6-Flash needed its vision and audio towers in the proxy). A proxy can stay large when such a table dominates the checkpoint (the Qwen3.8-Flash-Next 4-layer proxy is about 123 GiB because of its n-gram table).
- **Width reductions are a last resort.** Cutting experts per layer, embedding-table buckets or selection budgets makes a proxy fit on fewer nodes, but it changes routing and lookup semantics and amplifies numerical noise, so it raises the proxy's `gen_kl_error`. Keep head counts, head dimensions, hidden size and low-rank dimensions unchanged so that kernels see production shapes. Slice row-wise tensors along every logical partition: a table laid out as concatenated per-head ranges is cut per head, not by taking the first rows. Slice weights and their scales together.
- Keep the parallel-layout divisibility the recipe needs (heads divisible by vLLM TP, experts divisible by training EP).

Producing it (no generic tool; layer naming and state ownership are model-specific, so write a small script per model and keep it next to the slice):

1. Copy the tokenizer and auxiliary config files unchanged.
2. Edit only the config fields the slice changes: depth (`num_hidden_layers`, under `text_config` for multimodal configs), per-layer schedule lists (`layer_types`, compression ratios), source-layer owner fields, and any component switches or width fields deliberately reduced. If kept layers are not contiguous, renumber them to `0..N-1` in both the config and the tensor names.
3. When only whole layers are dropped, rewrite `model.safetensors.index.json` to the kept tensors and copy or hard-link only the shards that contain them. When tensors are cut, read byte ranges from the safetensors headers and write new shards. Either way, never dequantize and requantize: keep the original bytes, dtypes and scales.
4. Write into a new directory, refuse to overwrite, and write a completion marker only after every check passes: owner closure, weight/scale pairing, shapes against the new config, file lengths against headers, and a read-back checksum of each shard.
5. Write a slice record next to the checkpoint: source checkpoint and revision, the kept-layer table, every config field changed and why, removed components, width reductions and the semantics they change, tensor and shard counts, and the script's checksum.
6. Load the proxy once with AutoModel (forward and backward) and once with vLLM (prefill and decode) before using it in GRPO.

Choose smoke inputs that reach the model's special paths, not just short prompts: lengths that cross sliding-window boundaries, exceed sparse-attention top-k or candidate budgets, and leave odd tails for compression ratios. A length in the config is not proof the path ran; confirm from logs or counters.

## Integration touchpoints

Each item below has caused a failure in a past bring-up. Check each one; most need no change.

### Configuration and architecture

- **Config loading**: if the checkpoint's `model_type` is registered only by vLLM (not by Transformers), `AutoConfig.from_pretrained` fails. NeMo-RL must load the config through vLLM's config resolver on the generation side. Do not ship a rewritten copy of the HF config: it breaks as soon as the upstream type names change. Prefer passing vLLM arguments (for example `language_model_only: true`) and `hf_config_overrides`.
- **Architecture lists**: NeMo-RL keeps hard-coded tuples of architectures that need special handling (for example multimodal architectures in the vLLM worker). Add the new architecture names, including legacy aliases the checkpoint still uses.
- **Version asserts**: an `assert transformers < X` in a patch fails at import time on the new stack. Replace version checks with capability checks (does the registry entry exist).
- **Training-only config overrides**: router auxiliary losses, chat-template switches (`enable_thinking`), MTP or vision heads that RL does not train. Put them in the recipe, not in code.
- **Multimodal processors**: NeMo-RL starts vLLM with `skip_tokenizer_init=True` unless the architecture is on its tokenizer-required list; a vLLM multimodal processor that needs a tokenizer then refuses to start. Add the architecture to that list, or run vLLM text-only (next item).
- **Multimodal checkpoints in a text-only recipe**: vLLM's multimodal wrapper builds the towers even when no image is sent, and a tower's head count can cap the rollout TP (MiMo-V2.6-Flash's vision tower has 8 KV heads, so TP16 failed). Make vLLM resolve the text-only architecture through `hf_overrides` (for example `vision_config: null`), and drop the tower weights in refit when vLLM runs the text-only class; NeMo-RL already filters Gemma4's this way. AutoModel keeps the towers (see [non-decoder components](#build-the-proxy-model)).
- **Overrides that reach vLLM**: `policy.hf_config_overrides` apply to both sides, while some overrides must reach vLLM only (precision, QKV layout, text-only switches). Check that vLLM's startup log shows them: at the time of MiMo-V2.6-Flash, `grpo.py` replaced `vllm_kwargs.hf_overrides` with the policy overrides instead of merging them, and vLLM silently started in FP8. The fix (policy overrides first, vLLM-only overrides layered on top) is on the `mimo-v26-flash-support` branch.
- **Rollout routing replay (R3)**: on by default for MoE models (`policy.router_replay.enabled: true`), so it must work from the first proxy smoke: vLLM returns the routed experts for every MoE layer and the model's AutoModel MoE layers accept them. A new MoE layer type (shared experts, hash or lookup routing, an MTP head with experts) may need wiring for this. The vLLM side of R3 changes between vLLM versions: older vLLM needed `enable_prefix_caching: false` so routes came back for every prompt token, while vLLM with the AuxOutput connector stores routes per KV block and refuses to start unless prefix caching is on. Check the pinned vLLM and set the recipe accordingly; NeMo-RL's vLLM source patches for routed-experts capture are anchored on vLLM code and also need re-checking after a bump. On the training side, R3 replays through the gates AutoModel builds from its MoE config; a model that builds its own gate or its own MoE config can silently drop the replay switch ("model has no RouterReplay gates"). NeMo-RL also reads the expert count from the config, and models name it differently (`n_routed_experts`, `num_experts`, `moe_num_experts`).

### AutoModel model code under RL settings

AutoModel validates a new model under SFT. RL runs the same code with settings SFT does not use, which is where most AutoModel-side bring-up fixes come from:

- **Parameter storage dtype**: NeMo-RL can keep parameters in FP32 with BF16 FSDP mixed precision. Model code that picks its compute dtype from the weight dtype then feeds FP32 activations to BF16 kernels. Take the compute dtype from the FSDP mixed-precision policy.
- **Passes without backward**: logprob passes run forward only. AutoModel's FSDP2 `reshard_after_forward` defaults to False, so every layer stays unsharded after the pass; set `automodel_cfg.moe_parallelizer.reshard_after_forward: true` for large MoE models.
- **Optimizer state in checkpoints**: see the checkpoint OOM case in [full-model-and-longrun.md](full-model-and-longrun.md#fit-in-memory).
- **Wrappers around AutoModel functions**: NeMo-RL wraps some AutoModel loading functions for older checkpoints; a newer AutoModel signature breaks them. Check each wrapper's own removal condition first.

### Weight refit

The refit path is where most new-model failures land, because the model's parameter layout, dtypes and sizes are new.

- **Names and layout**: the AutoModel HF state-dict adapter must emit the names and fused order vLLM's loader expects. Apply layout transforms (transposes, expert regrouping) before gathering a DTensor, not after, to avoid holding two full copies.
- **Dtypes**: send each tensor in the dtype the adapter emits. Casting integer or boolean buffers (hash constants) corrupts them; that is always a bug. Casting all floating tensors to BF16 quantizes FP32 parameters (state-space `A_log` and `dt_bias`, router correction biases, gate weights); usually a small effect, but large for FP32-sensitive modules, and the fix is cheap. vLLM's loader casts on receipt where it needs to.
- **Partial loads**: refit sends weights in buckets. A vLLM model that checks fused weights for completeness (all shards of a fused projection present) rejects the first partial bucket unless the load runs inside vLLM's weight-reload state. Wrap each bucketed load in it.
- **Gathering large expert tensors**: an expert tensor sharded as EP on dim 0 and FSDP on dim 1 is built twice by `full_tensor()`, because gathering along a non-zero dim allocates a buffer and a concatenated copy. Redistribute the non-zero dims first, and run the HF adapter's transposes on the sharded tensor before gathering.
- **Oversized parameters**: a single tensor larger than the refit staging buffer OOMs the streaming path. Send it by direct IPC, release the staging buffers first, and size buffers with `refit_buffer_size_gb` and the refit-buffer environment knobs.
- **Owner-sharded parameters**: a parameter the model keeps sharded across an owner group (huge embedding tables) must not be materialized with `full_tensor()`. Replicate its checkpoint views one at a time, and let each vLLM rank receive the manifest for its own shard. Check that the rollout TP matches the owner sharding; a mismatch must fail with a clear error.
- **Worker-extension names**: methods NeMo-RL adds to the vLLM worker extension can collide with attributes a newer vLLM `Worker` defines; vLLM refuses to start. Check the extension against the pinned vLLM's `Worker` attributes after every vLLM bump.
- **Lost attributes**: `Module.to()` and similar calls can drop Python attributes that mark model-owned parameters. Snapshot such markers by fully qualified name and re-attach them.

### Generation

- vLLM-side workarounds for model features that the async scheduler or the model runner does not yet handle (for example placeholders in token history) are installed only when the model's config requires them, log one line when installed, and have a change-list entry ([branches-and-upstreaming.md](branches-and-upstreaming.md#keep-the-branches-easy-to-split-later)).
- Run with CUDA graphs on (`enforce_eager: false`) from the start. Switch to eager only to debug a hard failure that does not yield otherwise, and record it if the recipe ships eager: eager decode is several times slower and makes every full-model iteration expensive.

### Training memory

- If the optimizer keeps FP32 master weights, load the model in the training dtype; loading in FP32 doubles the parameter memory for nothing.
- `offload_optimizer_for_logprob` and deferred optimizer-state placement keep optimizer state off the GPU during logprob and forward passes.

## Smoke run

Create the recipe first, by copying the closest existing AutoModel DAPO or GRPO recipe in `examples/configs/recipes/llm/` (a similar size and MoE layout), not by inheriting the base config alone. The optimizer settings come with it unchanged: learning rate `1e-6` with a 10-step linear warmup, as in the existing AutoModel recipes. A recipe that sets no scheduler silently inherits the base config's (a 50-step warmup from 0.1x in one bring-up, which left validation accuracy flat for the first 20 steps).

Set the optimizer the full model will use from the first smoke: optimizer class, master weights and moment dtype from the model's AutoModel SFT recipe (for large MoE models, TE FusedAdam with master weights and BF16 moments; see [plan the node count](full-model-and-longrun.md#plan-the-node-count)). The proxy's checkpoint and resume row then tests the optimizer state the long run will save. Starting with the base config's AdamW and switching later repeats the proxy rows and, on the full model, OOMs in refit from FP32 parameters.

Then run that recipe with overrides, never a separate hand-written config, so that the smoke tests the configuration that will ship. The first two rows are required; the others only when the recipe uses that feature.

| Run | Overrides | Proves |
|---|---|---|
| Colocated 1 to 2 steps (required) | proxy checkpoint; `max_num_steps` 1 to 2; small prompts × generations; the production TP/EP if it fits, else the smallest layout with the same TP; R3 on for MoE | generation, refit, logprobs, train step, routed experts reach the trainer |
| Checkpoint and resume (required) | save at step 1, relaunch, run step 2 | checkpoint save/load of model and optimizer, dataloader state |
| Parallel variant (when the production layout cannot run on the proxy) | a second EP (or TP for vLLM) value | sharding and refit manifests are layout-independent |
| Eval and refit lifecycle (when the recipe validates during training) | validation before step 1 and at every step | generation after vLLM sleep/wake always receives a fresh refit; eval does not run on stale or freed weights |
| Non-colocated (if the recipe uses it) | separate generation workers | cross-process refit transport |

## Reading the smoke run

Pass signals, all required:

- The run reaches "Max number of steps has been reached" (or the equivalent final step) and the job exits 0.
- Any model-specific install line expected (generation workaround installed, owner-sharded refit path taken) appears in the driver or worker log.
- `gen_kl_error` is reported at every step, is finite, and does not grow between step 1 and step 2.
- Refit time is recorded.

The proxy's `gen_kl_error` is a regression baseline, not an absolute gate. It can read higher or lower than the full model: a slice with reduced widths or quantized caches amplifies small numerical differences, while a short full-width slice can read far lower. Compare it only with earlier runs of the same proxy:

| Model | Proxy | Proxy `gen_kl_error` | Full model |
|---|---|---|---|
| Qwen3.8-Flash-Next | 4 bottom layers, full width | 0.0016-0.0019 | 0.0006-0.0013 |
| DeepSeek-V4.1-Flash | 12 layers, experts 384→32, reduced n-gram table | about 0.007 after fixes (0.44 before the precision and routing fixes) | on the order of `1e-4` |
| Step-5-Preview | bottom 8 layers, all 352 experts, R3 on | about `1.6e-5` | about `5e-4` |

How to act on it:

- Above about `5e-2`: a real mismatch is likely. Go to [mismatch-triage.md](mismatch-triage.md) before spending full-model allocations.
- Below about `5e-2`: proceed to the full-model short run, which is the gate that decides; record the value as this proxy's baseline.
- After the first pass, any rise against the proxy's own baseline at the same commit of the recipe is a regression to explain.

Ignore on a proxy: reward, validation accuracy and generation length (a 4-layer model generates until the length cap).

Known log noise that does not indicate failure: Ray session-name assertion messages from re-attached clusters, and `undefined symbol` import errors from optional kernel extensions built against another Torch ABI (DeepEP, NIXL, DeepGEMM) when the recipe does not use them. Confirm the recipe really does not use them before ignoring them.

## Gate

Stage 2 passes when all smoke rows required by the recipe pass with the signals above, and the bring-up notes list each touchpoint above as "unchanged" or "changed in <commit>" with the generic or model-specific classification.
