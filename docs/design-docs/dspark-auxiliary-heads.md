# DSpark Auxiliary Heads

`nemo_rl/models/megatron/draft/dspark.py` defines two heads for DSpark block
drafting. It does not define a training objective, trainer integration, or
rollout execution.

## Inputs and Outputs

The target vocabulary supplies previous-token IDs. The draft vocabulary
defines the output-logit width. The two vocabularies can have different sizes.

| Module | Inputs | Output |
| --- | --- | --- |
| `DSparkMarkovHead` | Base logits with shape `[..., local_draft_vocab_size]`, previous-token IDs, and a Boolean valid-slot mask | Corrected logits with the same shape as the base logits |
| `DSparkConfidenceHead` | Hidden states, a Boolean valid-slot mask, and Markov embeddings when `with_markov=True` | One float32 confidence logit per slot |

The caller supplies the base logits. The Markov head owns the previous-token
embedding. It does not own the draft backbone's embedding or language-model
(LM) output head. For valid slots, it adds a low-rank previous-token bias to
the base logits. Both heads return zero for invalid slots. The confidence head
returns logits, not probabilities.

## Tensor Parallelism

The Markov embedding `markov_w1` is replicated across tensor-parallel (TP)
ranks. The output projection `markov_w2` can split its draft-vocabulary rows
across TP ranks. Each rank must receive base logits for its local vocabulary
shard.

Set `reduce_across_tensor_parallel=True` when Markov embeddings feed the
vocabulary-sharded projection. Set it to `False` when they feed a replicated
confidence projection. This prevents the replicated confidence path from
counting the same gradient more than once.

## Checkpoint Contract

Attach the heads as `markov_head` and `confidence_head` on the draft model.
Keep the parameter names below. The pinned
`deepseek-ai/dspark_qwen3_8b_block7` artifact uses `with_markov=True` and one
`vocab_size` of `151936`. Both Markov weights have `151936` rows. The test
fixture records the artifact at revision
`03326e5043815da1f81b109078b2889737c26017`.

| Checkpoint key | Shape |
| --- | --- |
| `markov_head.markov_w1.weight` | `[target_vocab_size, markov_rank]` |
| `markov_head.markov_w2.weight` | `[draft_vocab_size, markov_rank]` before TP sharding |
| `confidence_head.proj.weight` | `[1, hidden_size + markov_rank]` |
| `confidence_head.proj.bias` | `[1]` |

`markov_w1` remains replicated in the Megatron checkpoint. The checkpoint
stores the TP shard axis of `markov_w2` as axis 0. The pinned test fixture
checks checkpoint keys and shapes. It does not download checkpoint tensors or
test an end-to-end DSpark run.

The confidence head also supports `with_markov=False`. In that mode,
`proj.weight` has shape `[1, hidden_size]`. This shape does not match the pinned
checkpoint. Distinct target and draft vocabularies also require matching
checkpoint weights.
