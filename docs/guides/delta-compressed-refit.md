# Delta-Compressed Collective Refit

Delta-compressed collective refit reduces repeated non-colocated vLLM weight
syncs by sending a full baseline periodically and sparse additive deltas between
baseline refreshes.

Use it when policy and vLLM generation run on separate GPU resources and the
same generation workers are refitted many times during training.

## Support

Supported:

- non-colocated vLLM generation
- collective NCCL refit
- floating-point model weights

Not supported:

- colocated IPC/ZMQ refit
- SGLang generation
- ModelOpt quantized vLLM weights
- vLLM FP8 model weights

## Configuration

Enable the feature under `policy.generation.delta_compression`:

```yaml
policy:
  generation:
    backend: "vllm"
    colocated:
      enabled: false
      resources:
        num_nodes: 1
        gpus_per_node: 8
    delta_compression:
      enabled: true
      dtype: "bfloat16"
      full_sync_interval: 20
      sparse_bucket_size_bytes: 1073741824
      delta_load_batch_size_bytes: 536870912
```

| Field | Description |
|---|---|
| `enabled` | Enables delta-compressed collective refit. |
| `dtype` | Dtype used for delta tensors. Typical values are `"bfloat16"` or `"float32"`. |
| `full_sync_interval` | Force a dense full sync every N successful syncs. The first sync is always full. With `20`, sync 1 is full, syncs 2 through 20 are delta-eligible, and sync 21 is full. |
| `sparse_bucket_size_bytes` | Maximum sparse payload bytes to bucket before broadcasting. Start with `536870912` or `1073741824`. |
| `delta_load_batch_size_bytes` | Maximum decoded delta tensor bytes to batch before calling the vLLM weight loader. Start with `536870912`. |

Floating-point deltas use sparse-index payloads. Non-floating tensors are sent as
full-update chunks. Benchmark with representative weight changes before enabling
this by default for a recipe.

## Verify

`tools/refit_verifier.py` creates policy and vLLM workers, performs refits,
generates with vLLM, and compares generated-token logprobs against the policy
backend. Delta compression requires `--non_colocated`.

Example two-node Qwen3-30B-A3B run with one 8-GPU policy node and one 8-GPU vLLM
node:

```bash
uv run --extra mcore python3 tools/refit_verifier.py \
  --model_name /path/to/Qwen3-30B-A3B-Base \
  --non_colocated \
  --policy_num_nodes 1 \
  --generation_num_nodes 1 \
  --policy_gpus_per_node 8 \
  --generation_gpus_per_node 8 \
  --tp_size 1 \
  --ep_size 8 \
  --pp_size 1 \
  --vllm_tp_size 8 \
  --vllm_ep_size 8 \
  --vllm_pp_size 1 \
  --max_new_tokens 1 \
  --max_sequence_length 128 \
  --num_refits 3 \
  --enable_delta_compression \
  --delta_sparse_bucket_size_bytes 536870912 \
  --delta_load_batch_size_bytes 536870912 \
  --vllm_gpu_memory_utilization 0.8
```

Expected success markers:

```text
Collective refit initialized
Refit pass 1/3
Refit pass 2/3
Refit pass 3/3
Model refitting completed
Script completed successfully!
```

The verifier also prints mean and maximum generated-token logprob differences.
Use those values to confirm vLLM still matches the policy backend within expected
backend and precision tolerance.

## Benchmark

Run the same verifier once without delta compression and once with it enabled.
Keep the model path, topology, sequence lengths, and number of refits fixed.

For the full-transfer baseline, remove:

```bash
--enable_delta_compression \
--delta_sparse_bucket_size_bytes 536870912 \
--delta_load_batch_size_bytes 536870912
```

The first delta-compressed refit is a full baseline sync by design. Compare
later refits when measuring delta-transfer behavior. Repeating refit without
weight changes mostly exercises the control path; benchmark after real optimizer
steps or otherwise change policy weights between refits.

`tools/refit_verifier.py` does not print refit timings by default. To time
verifier refits, use a temporary wrapper around `weight_sync.sync_weights()` or
use the training timer below.

For end-to-end GRPO runs, compare
`prepare_for_generation/transfer_and_update_weights` in the training log. In
async or exposed-generation workflows, also check any `weight_sync` timing entry
because those paths can report the synchronizer phase under that label.

## Slurm

When launching the verifier through `ray.sub`, mount both the repo and model
path into the container and pass the verifier command as `COMMAND`:

```bash
COMMAND="<verifier command>" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD,/path/to/models:/path/to/models" \
sbatch \
  --nodes=2 \
  --gres=gpu:8 \
  --account=YOUR_ACCOUNT \
  --partition=YOUR_PARTITION \
  --time=2:00:00 \
  ray.sub
```

Dependency import failures for CUDA bindings, NCCL bindings, vLLM, Megatron,
Ray, or logging libraries indicate an environment issue, not a
delta-compression issue.

## Tuning

Start with:

- `full_sync_interval: 20`
- `sparse_bucket_size_bytes: 536870912` or `1073741824`
- `delta_load_batch_size_bytes: 536870912`

Lower `sparse_bucket_size_bytes` can let sparse decode and vLLM weight loading
start earlier while later sparse broadcasts are still running. Too small a value
can increase header, packing, and loader overhead. Use end-to-end refit timing
to choose the best value for the model, sparsity pattern, and interconnect.

The feature keeps an additional CPU baseline copy on the policy side for
floating-point tensors it delta-encodes. During a sync, it also uses transient
GPU buffers for staged baseline reads, sparse payloads, decoded deltas, and vLLM
load batches. Transient memory is bounded by normal refit chunking and the
bucket/load-batch settings above.

For overlap and timeline analysis, use the NeMo RL Nsight Systems workflow in
[Nsight Systems Profiling](../nsys-profiling.md).

## Troubleshooting

- `--enable_delta_compression requires --non_colocated`: delta compression only
  works with collective refit.
- Delta refits are not faster: confirm weights changed between refits, try
  512 MiB or 1 GiB sparse buckets, and compare against a full-transfer run with
  the same topology. High-density deltas can be slower because sparse-index
  encoding adds index payloads and decode work.
- Out of memory during baseline staging: reduce bucket/load sizes, reduce other
  vLLM memory pressure such as `vllm_gpu_memory_utilization`, or reduce model
  parallel shard size if possible.
- Logprob differences are high: confirm the same model path, tokenizer,
  precision, TP/EP/PP settings, prompt, and sequence lengths are used for both
  policy and vLLM.
