# Train DFlash and DSpark Draft Models Online

DFlash and DSpark are block-parallel speculative decoders. NeMo RL can train
either draft model alongside a Megatron policy and refit the updated policy and
draft weights into vLLM for rollout generation.

This guide covers the supported online co-training path. For GRPO setup, see
the [GRPO guide](grpo.md). For the Eagle3 workflow, which has different packing
constraints, see [Train with Eagle3 Speculative Decoding](eagle3-speculative-decoding.md).

## Supported Topologies

The initial supported topology is intentionally narrow. NeMo RL rejects other
layouts before training.

| Component | Supported configuration |
| --- | --- |
| Training backend | Megatron Core |
| Target tensor parallelism | TP2 |
| Target pipeline parallelism | PP1 with no virtual pipeline parallelism |
| Target context parallelism | CP1, CP2, or CP4 |
| Sequence packing | Optional at CP1; required at CP2 and CP4 |
| Target sequence parallelism | Supported only with sequence packing |
| DFlash/DSpark body sequence parallelism | Disabled |
| Generation topology | CP1 and PP1 |

Online DFlash and DSpark training does not support fused linear log-probability
computation. When training with CP2 or CP4, use the split
begin/microbatch/finish training path. Setting `data_plane.enabled=true` with
`examples/run_grpo.py` selects the synchronous TransferQueue trainer that uses
that path.

## Configure DFlash

The repository includes a one-node Qwen3-8B recipe at
`examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dflash.yaml`.
The core DFlash configuration is:

```yaml
policy:
  model_name: Qwen/Qwen3-8B
  megatron_cfg:
    enabled: true
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    sequence_parallel: false
    use_fused_linear_logprobs: false
  dtensor_cfg:
    enabled: false
  sequence_packing:
    enabled: false
  draft:
    speculator_type: dflash
    enabled: true
    model_name: z-lab/Qwen3-8B-DFlash-b16
    loss_weight: 1.0
    gamma: 5
    anchors_per_sample: 2
    mask_token_id: 151669
    target_hidden_state_layer_ids: [1, 9, 17, 25, 33]
    num_layers: 5
    seed: 13
    vocab_tile_size: 256
    position_decay: 1.0
    max_cp_boundary_exclusion_fraction: 0.25
  generation:
    backend: vllm
    vllm_kwargs:
      speculative_config:
        method: dflash
        model: ${policy.draft.model_name}
        num_speculative_tokens: ${policy.draft.gamma}
        draft_tensor_parallel_size: 1
```

`gamma` is the number of masked speculative positions after the anchor, so a
training block contains `gamma + 1` positions. `anchors_per_sample` controls how
many training blocks are sampled from each logical sequence. Blocks never cross
packed sample boundaries.

## Configure DSpark

DSpark uses the same block-parallel body and adds a Markov head plus an optional
confidence head. Use `block_size`, not the DFlash-only `gamma` field:

```yaml
policy:
  model_name: Qwen/Qwen3-8B
  megatron_cfg:
    enabled: true
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    sequence_parallel: false
    use_fused_linear_logprobs: false
  dtensor_cfg:
    enabled: false
  sequence_packing:
    enabled: false
  draft:
    speculator_type: dspark
    enabled: true
    model_name: deepseek-ai/dspark_qwen3_8b_block7
    model_revision: 03326e5043815da1f81b109078b2889737c26017
    loss_weight: 1.0
    block_size: 7
    anchors_per_sample: 2
    mask_token_id: 151669
    target_hidden_state_layer_ids: [1, 9, 17, 25, 33]
    num_layers: 5
    markov_rank: 256
    markov_head_type: vanilla
    confidence_enabled: true
    confidence_with_markov: true
    ce_loss_weight: 0.1
    tv_loss_weight: 0.9
    confidence_loss_weight: 1.0
    loss_decay_gamma: 4.0
  generation:
    backend: vllm
    vllm_kwargs:
      speculative_config:
        method: dspark
        model: ${policy.draft.model_name}
        revision: ${policy.draft.model_revision}
        num_speculative_tokens: ${policy.draft.block_size}
        draft_tensor_parallel_size: 1
```

DSpark online training uses the live target vocabulary and target-owned
embedding/output weights. Omit `draft_vocab_size`, or set it to exactly the
target vocabulary size. A different draft vocabulary is rejected because the
online remapping path is not supported.

## Enable Packed Context Parallel Training

The following command turns the DFlash recipe into a one-node TP2 x CP4 packed
run. CP4 and TP2 consume all eight training GPUs, while colocated vLLM generation
uses CP1 and PP1.

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-qwen3-8b-1n8g-megatron-dflash.yaml \
  data_plane.enabled=true \
  policy.megatron_cfg.tensor_model_parallel_size=2 \
  policy.megatron_cfg.pipeline_model_parallel_size=1 \
  policy.megatron_cfg.context_parallel_size=4 \
  policy.megatron_cfg.sequence_parallel=true \
  policy.sequence_packing.enabled=true \
  policy.make_sequence_length_divisible_by=16
```

Use `context_parallel_size=2` and `make_sequence_length_divisible_by=8` for
TP2 x CP2. Megatron packed CP requires this value to be a multiple of
`2 * TP * CP`. Keep packing enabled. Target sequence parallelism may be disabled
while retaining packed CP, but it must not be enabled for an unpacked run.

Packed sequences carry logical sample IDs and cumulative sequence boundaries.
NeMo RL uses those boundaries to reconstruct target sequence-parallel captures,
gather projected key/value tensors across CP, and assign every draft window to
exactly one CP rank. The application supplies no additional layout fields.

## Loss, Diagnostics, and Optimizer Settings

The total objective is the policy loss plus `policy.draft.loss_weight` times the
provider-specific draft loss. NeMo RL logs the normalized value as
`train/draft_loss`.

Set `policy.draft.update_probe_enabled=true` for a short diagnostic run. After
the optimizer step, NeMo RL checks that the draft model had nonzero gradients
and that at least one draft parameter changed. Disable the probe for ordinary
training.

To use a separate learning-rate schedule for draft parameters, set:

```yaml
policy:
  draft:
    optimizer:
      lr: 1.0e-5
      min_lr: 1.0e-6
      weight_decay: 0.01
```

If `optimizer` is `null`, draft parameters use the policy optimizer settings.

## Checkpoint and Refit Behavior

- `policy.draft.model_name` initializes the trainer-owned draft model.
- Training checkpoints include the draft model and its optimizer state when
  draft training is enabled.
- `policy.generation.vllm_kwargs.speculative_config.model` initializes the vLLM
  drafter before the first refit.
- Each successful refit updates both target and draft weights. The generation
  `method` must match `policy.draft.speculator_type`.
- DFlash and DSpark generation requires PP1. Keep generation CP at 1 even when
  policy training uses CP2 or CP4.

## Common Configuration Errors

- `context_parallel_size > 1` with packing disabled
- target sequence parallelism with packing disabled
- target TP other than 2, or training PP/virtual PP greater than 1
- generation CP or PP greater than 1
- `use_fused_linear_logprobs=true`
- a generation method that does not match the training speculator
- `gamma` in a DSpark block or `block_size` in a DFlash block
- a DSpark `draft_vocab_size` different from the live target vocabulary

These combinations fail during configuration or worker setup instead of
starting a training step with incompatible ownership semantics.
