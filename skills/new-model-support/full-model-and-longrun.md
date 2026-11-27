# Full-model short run and long run

Stages 3 and 4 of [the workflow](SKILL.md#stages-and-gates). This is the most expensive and usually the longest part of a bring-up: problems that only appear at real depth, real width or real sequence length surface here. They fall into three groups: new architecture features (sparse or linear attention, hyper-connections, multimodal towers), mixed precision (FP32 islands, quantized experts or caches), and scale (100B+ parameters, long context, OOM).

## Full-model short run

### Plan the node count

Estimate memory per GPU before the first allocation, so the node count is decided once:

- Training state per parameter: BF16 weights 2 B, FP32 master weights 4 B, two Adam moments. FP32 moments make it 16 B; TE FusedAdam with master weights and BF16 moments makes it about 10 B. Use the optimizer the model's AutoModel SFT recipe uses; for 600B parameters this halved the node count (16 instead of 32 H100 nodes).
- Colocated generation adds the vLLM weights (BF16, sharded by its TP and EP) and the KV cache share set by `gpu_memory_utilization`.
- Check the largest single refit allocation (one layer's fused expert tensor across all experts) against what is left during refit.

Validate a new optimizer setting on the proxy first (same `gen_kl_error` and gradient norm, lower peak memory), then use it on the full model.

### Choose the topology

- vLLM tensor parallel size must divide the attention head count; for MoE models with sequence parallel experts, expert parallel size usually equals TP. Start from the vLLM PR's own serving arguments.
- With R3 on, vLLM cannot use pipeline parallelism: it refuses to return routed experts when PP is above 1. A model whose BF16 weights do not fit one node's rollout engine therefore runs vLLM with TP spanning nodes (MiMo-V2.6-Flash: about 620 GB of BF16 weights, TP16 across two H100 nodes), not TP×PP.
- Training expert parallel size must divide the expert count. Large MoE models typically use FSDP2 with EP equal to the GPU count and TP 1.
- Set `cluster.num_nodes` to exactly the number of nodes the scheduler allocates. A mismatch does not fail; Ray placement waits forever.
- Start at a short context (about 1K prompt plus 2K response) with CUDA graphs on, and extend only after the long run passes. Longer budgets and eager decode make every full-model iteration slower without testing anything new.

### Fit in memory

Apply in this order, one change per run, and record the effect of each:

| Knob | Effect |
|---|---|
| `vllm_cfg.gpu_memory_utilization` | the rollout's share of GPU memory in colocated mode; lower it when training or refit OOMs |
| Optimizer with FP32 master weights, BF16 moments | load the model in the training dtype; keep master weights in the optimizer |
| `offload_optimizer_for_logprob` | keeps optimizer state off the GPU during logprob and forward passes |
| `activation_checkpointing` | trades recompute for activation memory |
| `logprob_chunk_size`, dynamic batching | bounds logits memory for large vocabularies |
| `refit_buffer_size_gb` and refit buffer environment knobs | bounds the staging memory used to stream weights into vLLM |
| `enforce_eager` | removes CUDA-graph memory, at a large decode-speed cost; a last resort, not a default |

Before changing a knob, read the OOM traceback for the phase and the allocation, and compare the allocation size to what the phase should hold. Two cases from past bring-ups where no knob above helped:

The fixes for the cases below are on support branches, not on `main`. Before the first full-model allocation, check whether your base has them; if not, port them instead of rediscovering them. They live in the repository named with each case.

- **Logprob OOM with tens of GiB allocated (AutoModel):** FSDP2 `reshard_after_forward` defaults to False in Automodel. A logprob pass has no backward, so every layer stays unsharded and each GPU ends up holding its whole EP-local model. Set `automodel_cfg.moe_parallelizer.reshard_after_forward: true` (a recipe setting, no code change).
- **Refit OOM on one multi-GiB allocation (AutoModel, large MoE):** the refit gathers one layer's fused expert tensor for all experts. An all-gather along a non-zero dim allocates the result twice, and the HF adapter's transposes add more copies ([gathering large expert tensors](proxy-and-enablement.md#weight-refit)). Check which line allocates before lowering `gpu_memory_utilization`: vLLM's KV cache is already released during refit. With AdamW and FP32 parameters the same tensor is twice as large; switching to the FusedAdam setup in [plan the node count](#plan-the-node-count) removed this OOM for MiMo-V2.6-Flash with no code change.
- **Checkpoint save or load OOM with a low-precision optimizer (AutoModel, TE FusedAdam):** training steps fit, then the first save or the first resume fails. FusedAdam's `state_dict()` converts BF16 moments into new FP32 tensors on the GPU mesh (+8 bytes per parameter), and torch `Optimizer.load_state_dict` casts all loaded state to the parameter device at once. Moving state to CPU does not help the save, because the copies are re-wrapped with `DTensor.from_local` on the CUDA mesh. The fix saves BF16 moments as stored (lossless, about a third smaller) and loads optimizer state with the parameters on CPU. Failures here are rank-dependent: one rank can sit at 9 GiB while another OOMs at 69 GiB, so read the failing rank's log. Two implementations exist: in NeMo-RL's AutoModel checkpoint code on the `step5-support` branch (commits `1b5af0be0`, `7de63ea46`, `b3f1b4a4b`, `b31aa8c27`, `6aefe2930`), and in AutoModel's checkpoint wrappers on the AutoModel branch `rl/mimo-v26-support` (commit `a1e7d533d`, save only). The second was written because the first was not found; check both before writing a third.

### Learning rate

Keep the learning rate and warmup the recipe inherited from the existing recipe it was copied from ([smoke run](proxy-and-enablement.md#smoke-run)); do not tune them during the bring-up. Scheduler state is saved in checkpoints, so a resume keeps the old schedule even after the config changes; a schedule fix needs a restart from the base model.

### Lengths

`max_new_tokens` must equal `max_total_sequence_length - max_input_seq_length`. If it is larger, vLLM silently caps generation per sample at `max_model_len - input_length`, so the generation budget varies with prompt length.

### Read the short run

Run several GRPO steps and record per step: `gen_kl_error`, step time and its breakdown (generation, logprobs, training, refit), peak memory per phase, generation length and reward.

Save a checkpoint at the last step of the short run and resume from it for one more step. The proxy's save and resume prove the code path, not the memory: a 6-layer proxy cannot show that saving a 300B model's optimizer state OOMs. In one bring-up that OOM surfaced only at step 10 of the long run, costing the run slot and the requeue.

Gate: no OOM, hang or error, including the save and the resume; `gen_kl_error` around `1e-3` at every step with no upward trend; refit and step time recorded. If `gen_kl_error` is well above `1e-3`, go to [mismatch-triage.md](mismatch-triage.md) before the long run.

## Long-run validation

### Why 100 steps

Some train-inference mismatches are invisible at the start. In one bring-up, BF16 training against FP8 rollout looked healthy for about 100 steps and then `gen_kl_error` grew until training collapsed; the fix (train-side FP8 fake quantization matching the rollout's rounding) was found only through the long run. The long run exists to catch that kind of slow drift.

### Gate

All required:

- 100 or more GRPO steps with no crash in the RL loop. A crash is any job ending with an error in generation, refit, logprobs, training or checkpointing (OOM, NaN, assertion, NCCL error, hang). A scheduled stop at the partition time limit followed by resume from checkpoint is not a crash. Failures outside the RL loop (node failure, filesystem outage, missing image) are rerun and recorded, and do not count toward or against the steps.
- After a crash, fix the cause and count again from the fixed commit.
- `gen_kl_error` stays around `1e-3` across the whole run, with no upward trend. Judge the trend over the last 50 steps, not single spikes; investigate any spike that does not return to the baseline.
- Reward and validation accuracy improve over the run. If they do not improve with the inherited recipe, stop: that needs a human, not more runs.
- Other training metrics stay healthy: finite loss and gradient norm, entropy not collapsing to zero, generation length not pinned at the cap.

### Run it

- Chain jobs within the partition time limit, each resuming from the last checkpoint. Save checkpoints often enough that a time-limit stop loses at most a few steps.
- Check that resume restores the step counter, optimizer state and dataloader position; a resume that silently restarts from step 0 invalidates the count.
- Keep the code commit and image fixed for the whole counted run. A code change mid-run starts a new count unless it is provably outside the RL loop.
- Record each job id with the step range it covered.
- Size checkpoints against the storage quota before the run: weights plus optimizer state, times `keep_top_k`, plus the one being written. Set `checkpointing.metric_name: null` with `keep_top_k: 1` so the latest checkpoint, not the best-scoring one, survives for resume.
- Set `checkpointing.checkpoint_must_save_by` a little under the partition time limit, so each job saves and exits on its own instead of being killed mid-step.
- The full model's first save and resume happen in the short run. After any change to how checkpoints are written, run a save, then a load and one more step on the proxy, and compare `gen_kl_error` and gradient norm with the steps before the save.

## Settle the recipe

When the long run passes, the recipe in `examples/configs/recipes/llm/` must equal the validated run's configuration, apart from the checkpoint path placeholder and the step limit. Compare the resolved config of the run with the recipe file; any difference is either copied into the recipe or stated in the guide as unvalidated. Continue with [docs-and-release.md](docs-and-release.md).
