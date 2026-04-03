# Train with Eagle3 Speculative Decoding

Eagle3 speculative decoding speeds up rollout generation by running a smaller draft model in vLLM and having the policy model verify its proposals. In NeMo RL, you can either use a fixed Eagle3 draft model only for generation, or train that draft model online during RL so it stays aligned with the policy.

This guide covers the NeMo RL-specific runtime and training path. For a high-level overview of speculative decoding, see [An Introduction to Speculative Decoding for Reducing Latency in AI Inference](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/). For GRPO fundamentals, see the [GRPO guide](grpo.md). For asynchronous rollout collection, see the [Async GRPO guide](async-grpo.md).

## Offline vs Online

- **Offline draft model**: vLLM uses a fixed Eagle3 checkpoint for speculative decoding, but the RL training loop does not update that draft model.
- **Online draft training**: NeMo RL attaches an Eagle3 draft model to the Megatron policy worker, trains it alongside the policy, and refits both policy and draft weights into vLLM.

Use the offline path when you already have a good drafter and only want faster rollouts. Use the online path when the policy is changing during RL and you want the drafter to track those updates.

## Draft Checkpoint

For the best results, start from an Eagle checkpoint that was already pretrained as a draft model, then use NeMo RL's online draft loss to keep it aligned with the policy during RL. For training or adapting an Eagle checkpoint, see the [Model Optimizer speculative decoding example](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/speculative_decoding/README.md).

NeMo RL now keeps a trainer-owned draft LM head. If the draft checkpoint contains
`lm_head.weight`, NeMo RL loads it into the draft model. If that weight is absent,
NeMo RL initializes the draft LM head from the current policy output layer instead.

## Enablement

### Generation Only

```yaml
policy:
  generation:
    backend: "vllm"
    vllm_kwargs:
      speculative_config:
        method: "eagle3"
        model: /path/to/eagle3-draft
        num_speculative_tokens: 3
```

This enables Eagle3 in vLLM, but the trainer does not own or update the draft model.

### Online Draft Training

```yaml
policy:
  megatron_cfg:
    enabled: true
  dtensor_cfg:
    enabled: false

  draft:
    enabled: true
    model_name: ${policy.generation.vllm_kwargs.speculative_config.model}
    loss_weight: 1.0

  sequence_packing:
    enabled: false

  generation:
    backend: "vllm"
    vllm_kwargs:
      speculative_config:
        method: "eagle3"
        model: /path/to/eagle3-draft
        num_speculative_tokens: 3
        draft_tensor_parallel_size: 1
```

> [!NOTE]
> Online draft training currently requires the Megatron backend and does not support sequence packing yet. Set `policy.megatron_cfg.enabled=true`, `policy.dtensor_cfg.enabled=false`, and `policy.sequence_packing.enabled=false`.

## How It Works

### Rollout Path

During generation, vLLM runs the Eagle3 drafter from `policy.generation.vllm_kwargs.speculative_config`. When `policy.draft.enabled=true`, NeMo RL refits both:

- the policy weights into the main vLLM model
- the `draft.*` weights into the vLLM drafter

That keeps the rollout drafter aligned with the latest RL-updated policy instead of a stale checkpoint.

### Training Path

During the policy forward pass, NeMo RL captures:

- token input embeddings
- a small set of intermediate hidden states from auxiliary policy layers

Those captured activations are the Eagle inputs. NeMo RL captures an early/middle/late-style set of policy layers for Eagle3, then the draft model predicts logits with its own draft LM head. That LM head is loaded from the draft checkpoint when `lm_head.weight` is present and otherwise initialized from the current policy output layer.

### Draft Loss and Time-Step Alignment

The draft loss compares draft logits against detached policy logits, but only after aligning both sides to the same next-token event.

Suppose the policy input sequence is:

```text
[BOS, The, cat, sat]
```

The policy forward pass produces hidden states and logits at those positions:

```text
position:            0      1      2      3
input token:       [BOS]  [The]  [cat]  [sat]
hidden state:        h0     h1     h2     h3
policy logits:       p0     p1     p2     p3
predicts:           The    cat    sat    EOS
```

For Eagle training, NeMo RL does not compare raw `p0, p1, p2, p3` directly to the raw draft output. Instead it shifts the draft inputs and teacher outputs so draft position `t` predicts the teacher distribution for position `t+1`.

First, it rolls the captured input embeddings left by one token:

```text
original embeddings: e(BOS)  e(The)  e(cat)  e(sat)
shifted embeddings:  e(The)  e(cat)  e(sat)    -
```

Then it rolls the detached teacher logits left by one position:

```text
original teacher logits:  p0      p1      p2      p3
rolled teacher logits:    p1      p2      p3       -
teacher meaning:         cat     sat     EOS       -
```

So the aligned draft-training pairs become:

```text
(h0, e(The)) -> p1
(h1, e(cat)) -> p2
(h2, e(sat)) -> p3
```

In words:

- use the hidden state at position `t`
- use the embedding of the token at position `t+1`
- predict the teacher distribution for position `t+1`

After this alignment, the draft loss is:

$$
L_{draft} = \mathbb{E}*t \left[- \sum_v \mathrm{softmax}(z*{policy,t})*v \log \mathrm{softmax}(z*{draft,t})_v \right]
$$

Here `z_{policy,t}` and `z_{draft,t}` refer to the aligned tensors after rolling, truncation, and masking, not the raw unshifted outputs of the forward pass.

This has the same student gradient as forward KL from the policy distribution to the draft distribution, up to the teacher entropy constant. The total training objective is:

$$
L_{total} = L_{policy} + \lambda \cdot L_{draft}
$$

where `lambda` is `policy.draft.loss_weight`.

## Important Knobs

- `policy.draft.enabled`: attach and train the Eagle draft model
- `policy.draft.model_name`: checkpoint used to initialize the draft model
- `policy.draft.loss_weight`: weight on the auxiliary draft loss
- `policy.generation.vllm_kwargs.speculative_config.model`: draft checkpoint used by the vLLM drafter
- `policy.generation.vllm_kwargs.speculative_config.draft_tensor_parallel_size`: tensor parallelism used by the drafter inside vLLM
- `policy.generation.vllm_kwargs.speculative_config.num_speculative_tokens`: number of speculative tokens proposed by vLLM

## Notes

- When online draft training is enabled, NeMo RL logs `draft_loss`.
- Resume checkpoints include the nested draft model state when `policy.draft.enabled=true`.
- If speculative decoding is enabled without trainer-owned draft weights, vLLM must load real draft weights at startup. When the trainer owns the draft model, the first refit pushes both policy and draft parameters.
- Online draft training does not currently support `policy.sequence_packing.enabled=true`.
