# Shared-prefix execution for GRPO (experimental)

GRPO generates several completions for each prompt. With ordinary sequence packing, the policy runs that prompt again for every completion when it computes log probabilities and when it trains. Shared-prefix execution is a sequence-packing option that runs the prompt once per forward for the rows that start with it.

The option is `policy.sequence_packing.share_prefixes`, and it is off unless it is set to `true`. It targets single-controller GRPO with a text-only hybrid model on the Megatron backend. A hybrid model is a model that Megatron Bridge builds through `HybridModelProvider`. The option does not change the GRPO objective, the sampled completions or rollout generation.

## Terms

- A *group* is the set of rows generated from one prompt. The row tag `GROUP_ID_TAG` identifies it.
- A *tree* is one prompt followed by the completions of the rows that start with it. It is a one-level prefix tree. The Megatron-LM type is `SharedPrefixTree`.
- A *pack* is one forward that holds several trees and runs each prompt once. The Megatron-LM type is `SharedPrefixPack`.
- A *dense bin* is one forward of ordinary sequence packing, in which every row carries its own prompt.
- A *forward* is one model call on one microbatch. It is either a pack or a dense bin.
- The *packed length* of a tree counts the prompt once and every completion once. The *full length* counts the prompt in every row.
- The *padding multiple* is `policy.make_sequence_length_divisible_by`. Every row and every pack is padded to a multiple of it.

A tree is a prefix tree: its rows start with the same tokens. It is not the call tree that `nemo_rl/environments/call_selection.py` reads from parent links. A pack is unrelated to forest training (`token_capture.forest_training`), in which one rollout owns several rows.

## Configuration

Add one key to the sequence-packing block of a single-controller GRPO recipe for a text-only hybrid model. The fragment shows the key next to the settings it depends on. The token budgets are illustrative.

```yaml
policy:
  sequence_packing:
    enabled: true
    algorithm: modified_first_fit_decreasing
    train_mb_tokens: 32768
    logprob_mb_tokens: 32768
    share_prefixes: true
  make_sequence_length_divisible_by: 16  # a multiple of 2 * TP * CP
  megatron_cfg:
    enabled: true
    tensor_model_parallel_size: 2
    context_parallel_size: 4
    sequence_parallel: true
  generation:
    top_p: 1.0
    top_k: null
data_plane:
  enabled: true
token_capture:
  enabled: true
```

Run the recipe with the single-controller launcher:

```sh
uv run examples/run_grpo_single_controller.py --config <recipe>.yaml
```

`examples/run_grpo.py` rejects the key at setup, because that launcher does not write the row tags that planning reads. The other launchers do not check the key. A run with one of them packs densely when the key is `true`. A Megatron worker of such a run still runs the model check at setup.

`examples/configs/recipes/llm/grpo-nano-v2-12b-1n8g-megatron-single-controller-token_capture-shared-prefix.yaml` is a complete recipe.

On-policy distillation teachers and the inference-only policy of non-colocated Megatron generation copy the policy config. They set the key to `false` in their copy, so they always pack densely. A value model also packs densely.

## How it works

The rollout reassembler tags every row. The driver plans the forwards of every data-parallel rank from those tags. Each Megatron worker builds the packs of its plan. The table lists where each part lives.

| Module | Role |
|---|---|
| `nemo_rl/experience/rollout_reassembler.py` | Stamps the prompt length and the prompt hash on every row. |
| `nemo_rl/data/packing/shared_prefix/row_metadata.py` | Names of the row tags, the plan entry and the batch columns; `UNSHARED_ROW`; `prompt_hash`. |
| `nemo_rl/data/packing/shared_prefix/plan.py` | `plan_shared_prefix_shards`, the planner. Pure Python. |
| `nemo_rl/data_plane/preshard.py` | `shard_meta_for_dp` calls the planner and ships the plan to each rank. |
| `nemo_rl/models/policy/shared_prefix.py` | `validate_shared_prefix_config`, the config check, and `bound_by_full_length`, the budget rule. |
| `nemo_rl/models/megatron/shared_prefix/setup.py` | `validate_shared_prefix_model`, the model check at worker setup. |
| `nemo_rl/models/megatron/shared_prefix/microbatches.py` | Builds the model inputs of one pack. `SharedPrefixParams` is the counterpart of `PackedSeqParams`. |
| `nemo_rl/models/megatron/shared_prefix/logprobs.py` | Turns the logits of a pack into per-row logprobs and into the loss input. |
| Megatron-LM, `megatron.core.datasets.shared_prefix` | The tree and pack types, and `build_shared_prefix_inputs`, which builds one pack's tensors. |
| Megatron-LM, `megatron.core.models.hybrid.shared_prefix` | `validate_shared_prefix_config`, which holds the model rules, and the model forward of a pack, which `HybridModel.forward(..., shared_prefix_layout=...)` calls. |

The planner and the config check import no Megatron-LM module, so the driver plans without a Megatron-LM install. Only `nemo_rl/models/megatron/shared_prefix/` imports the Megatron-LM shared-prefix modules, and it imports them from `megatron.core`.

### 1. The rollout reassembler tags every row

The rollout reassembler stamps two row tags on every row it writes. `PROMPT_LENGTH_TAG` is the number of prompt tokens at the start of the row. `PROMPT_HASH_TAG` is a hash of those tokens. The reassembler writes the tags whether or not the option is on, and it writes no extra data-plane column.

The prompt length is 0 for a placeholder row, for a row without a verified prompt, and for a row whose `token_mask` is nonzero inside its prompt. A row with prompt length 0 never shares.

### 2. The driver plans every rank's forwards

When the option is on, `TQPolicy._packing_args` adds two entries to the sequence-packing arguments of each logprob dispatch and each train dispatch: `share_prefixes` and `bound_by_full_length` (see [Token budget](#token-budget)). `shard_meta_for_dp` reads them. When every row of the dispatch carries both tags, it calls `plan_shared_prefix_shards`. Otherwise it packs densely, as it does without the option. A dispatch that holds rows from a data-plane checkpoint written before the tags existed therefore runs dense.

The planner reads four values per row: the length, the group, the prompt length and the prompt hash. It does no data-plane I/O. It plans one global batch at a time, as dense packing does, with these rules:

- Two rows share a prompt when they have the same group, the same prompt length and the same prompt hash, and the prompt is shorter than the row. Two groups with the same prompt do not share. A prompt that differs in one token does not share.
- A tree holds at most 16 rows (`MAX_TREE_ROWS`). A larger set of rows is split evenly. For example, 17 rows become trees of 9 and 8 rows. The rows of one prompt are also split when they do not fit the token budget of one forward.
- A row that shares with no other row runs in a dense bin.
- A pack holds trees up to the token budget of the stage. A dense bin holds rows up to the same budget.
- Every rank runs the same number of packs and the same number of dense bins, and it runs its packs first. The forward at a given index therefore has the same kind on every rank. The two kinds can run different collectives, so ranks must not mix them at one index (see [Token budget](#token-budget)).
- The number of packs is rounded up to a multiple of the number of ranks, and so is the number of dense bins. The planner adds no dummy rows to reach these counts.
- When there are fewer trees than packs, the planner halves the largest tree that has at least 4 rows. When no tree can be halved, the smallest tree stops sharing, and its rows run in dense bins.
- When there are fewer dense rows than dense bins, the smallest tree also stops sharing.
- The forwards of one kind are dealt longest first to the rank that has the fewest tokens so far.

Each rank's `KVBatchMeta` carries its plan in `extra_info`. Three entries are the ones dense packing already ships: `MICRO_BATCH_INDICES`, `MICRO_BATCH_LENGTHS` and `ELEM_COUNTS_PER_GB`. One entry is new: `SHARED_PREFIX_TREE_IDS` holds one integer per row of the rank. The integer is the row's tree within its forward, or `UNSHARED_ROW` (-1) for a row of a dense bin. The rows of a tree are contiguous.

### 3. The worker builds each pack

The worker fetches its rows through the same path as dense packing. The Megatron worker then adds two batch columns from the plan and the row tags: `SHARED_PREFIX_TREE_IDS` and `SHARED_PREFIX_PROMPT_LENGTHS`. The columns travel with every microbatch slice.

The microbatch iterator looks at the tree ids of each slice:

- All ids are `UNSHARED_ROW`: the slice is a dense bin. It runs through the ordinary packed path, unchanged.
- All ids are 0 or greater: the slice is a pack. The builder in `microbatches.py` calls Megatron-LM's `build_shared_prefix_inputs`, which lays out each prompt once, followed by the completions of its tree. The pack is padded to the padding multiple.
- Anything else raises.

The driver matched prompts by length and hash only. `build_shared_prefix_inputs` compares the prompt tokens of the rows of each tree and raises if they differ.

In a pack, each completion attends to its prompt and to its own earlier tokens. It cannot attend to a sibling completion or to another tree. Megatron-LM's `docs/developer/shared_prefix.md` describes the model side.

### 4. Logprobs are copied back and the loss runs per row

`logprobs.py` turns the logits of a pack into next-token logprobs for every row. The logprobs of the prompt are computed once per tree and copied to every row of the tree. The logprob of a row's first completion token is read at the last prompt position.

The loss then runs per row on those logprobs, with each row's own mask, advantage and ratio. No row owns the shared tokens. In the backward pass, the gradients of all rows of a tree add up in its one prompt. `microbatches.py` raises if `token_mask` is nonzero inside the shared prompt of any row, so a shared token never carries loss.

Each worker writes its results back under each row's own key. The order of the rows within a rank therefore does not matter, and nothing restores it.

## Token budget

Each forward is limited by the token budget of its stage, as in dense packing: `logprob_mb_tokens` for a logprob pass and `train_mb_tokens` for training.

- A dense bin counts every row at its padded length.
- A pack counts every tree at its packed length.
- Exception: a training pack counts every tree at its full length when multi-token prediction (MTP) may run. MTP processes every row with its own copy of the prompt.

The driver decides this per stage with `bound_by_full_length`:

| Stage | Length a tree is counted at |
|---|---|
| Logprob pass | Packed length. |
| Training, `policy.megatron_cfg.mtp_num_layers: 0` | Packed length. |
| Training, any other value, or the key absent | Full length. An absent key leaves the number of MTP layers to the checkpoint, so the driver assumes that MTP runs. |

When trees are counted at full length, sharing lowers the tokens that each training forward runs. It does not lower the number of training forwards.

A logprob pass runs the model in evaluation mode. In that mode Megatron-LM skips the MTP block in a pack, and it runs the MTP block in a dense bin, as it does without the option. For a model with MTP layers, a logprob pass therefore does MTP work for the rows of dense bins and none for the rows of packs. The logprobs do not use the MTP output. This difference between the two kinds of forward is why every rank runs the same kind at the same index. Skipping MTP in dense evaluation forwards is not part of this feature.

## Requirements and rejected configurations

The option needs every item below. A config that misses one is rejected at setup.

The launcher:

- The single-controller launcher, `examples/run_grpo_single_controller.py`. Its setup requires `data_plane.enabled: true` for every run. `examples/run_grpo.py` raises when the key is `true`. The other launchers do not check the key and pack densely (see [Configuration](#configuration)).

Checked once on the driver by `validate_shared_prefix_config`, before any worker starts. Each rule is one error that names the key to change:

- `token_capture.enabled: true`. Only the rollout reassembler writes the two row tags.
- `policy.sequence_packing.enabled: true` and `policy.dynamic_batching.enabled: false`.
- `policy.sequence_packing.pair_grouping_key` and `policy.sequence_packing.max_sequences_per_bin` unset. The planner applies neither.
- `policy.megatron_cfg.enabled: true`. Other backends do not build packs.
- `policy.megatron_cfg.pipeline_model_parallel_size: 1`.
- `policy.megatron_cfg.peft.enabled: false`.
- `policy.megatron_cfg.use_fused_linear_logprobs: false`.
- `policy.megatron_cfg.moe_hybridep_prepad_packed_inputs: false`. That setting runs a collective for every dense bin and none for a pack.
- `policy.megatron_cfg.zero_train_gen_mismatch: false` and `policy.megatron_cfg.batch_invariant_mode: false`. Generation runs every row in full, so a pack cannot reproduce its arithmetic exactly.
- `policy.generation.top_p: 1.0` and `policy.generation.top_k: null`. Logprobs of a pack are not filtered by top-k or top-p.
- `policy.router_replay.enabled: false`.
- `policy.is_vlm: false`.
- `policy.draft.enabled: false`.
- `loss_fn.score_centering: false` and `on_policy_distillation.full.enabled: false`. A pack gives the loss next-token logprobs, not logits over the vocabulary.
- `loss_fn.seq_logprob_error_in_loss: false`.
- `loss_fn.positive_example_nll_weight: 0`. That term is a mean over the tokens of one loss call. A dense bin without `fuse_loss` calls the loss once per row, and a pack calls it once per forward, so the same rows would give a different term in a pack.

Checked on each Megatron worker at setup by `validate_shared_prefix_model`, because these rules need the built model config:

- The model comes from Megatron Bridge's `HybridModelProvider`.
- The model takes the packed, context-parallel-sharded inputs that the worker builds. A model that packs its own sequences, or slices its own inputs for context parallelism, is rejected.
- The model settings pass Megatron-LM's `validate_shared_prefix_config`, which owns the model rules. The driver check repeats two of them, the pipeline-parallel size and batch-invariant mode, so that they fail before a worker starts.
- `policy.make_sequence_length_divisible_by` is a padding multiple that Megatron-LM accepts for packs at the model's TP and CP sizes (`resolve_shared_prefix_padding_multiple`).
- A model with MTP layers has `calculate_per_token_loss` enabled. Megatron-LM requires it in the training forward of a pack, and this check fails earlier.
- The built model has no MTP layers when `policy.megatron_cfg.mtp_num_layers: 0` makes the driver count training packs at packed length.

## Forest training

Forest training (`token_capture.forest_training: true`) is supported. In forest training the reassembler writes one row per selected path of a rollout, so the groups of a step have different row counts. The planner accepts any row count per group.

- The prompt of a row is the prompt of the first model call on its path.
- Rows of one group share when that prompt has the same length and the same tokens. These are the first rows of the rollouts and the rows that branch under the same first call.
- A row that starts after a context rewrite, such as a compaction, has a different prompt. It runs in a dense bin unless another row of the group starts with the same rewritten prompt.
- The rows of an invalid rollout and the execution-padding rows have prompt length 0. They run in dense bins and carry no loss.
- Advantages are computed per rollout on the driver before packing, so each rollout keeps one advantage.
- Rows that branch below the prompt share only the prompt. The tokens of the calls they have in common still run once per row.

## What is not supported

- Sharing past the prompt. Rows with the same first assistant turn, or branch rows with the same parent call, share only the prompt.
- Sharing across groups.
- Sharing inside MTP. MTP runs on every row in full.
- Backends other than Megatron, Megatron models that are not hybrid models, and multimodal models.
- Every setting that [Requirements and rejected configurations](#requirements-and-rejected-configurations) rejects.
- FLOPs and MFU metrics that follow the packed length. They count every row at its full length, so they overstate the work of a run that shares prompts.
- Warm-up of pack forwards. The synthetic first training pass (`async_rl.trainer_warmup`) runs dense bins only.

## How to test

Most unit tests of the feature are in files under `tests/unit/` whose name contains `shared_prefix`.

- The driver-side tests need neither Megatron-LM nor a GPU. They cover the planner (`tests/unit/data/packing/`), the plan that `shard_meta_for_dp` ships (`tests/unit/data_plane/`), the config rules (`tests/unit/models/policy/`), the `examples/run_grpo.py` guard (`tests/unit/algorithms/`), and a training step that loses a prompt group (`tests/unit/single_controller/`).
- The tests of the row tags that the rollout reassembler writes, including the rows of forest training, carry the `nemo_gym` marker (`tests/unit/data_plane/`, `tests/unit/experience/`).
- The Megatron-side tests carry the `mcore` marker and run in the Megatron lanes. They cover the pack microbatches, the logprobs, the worker setup check and the worker's batch columns.
- `tests/unit/models/megatron/test_shared_prefix_logprob_oracle.py` compares the logprobs, the loss and the gradients of packs with those of the dense rows they replace. It uses a table-lookup model whose logits depend only on causal history, so a correct implementation must match dense exactly. One case has the row shapes of forest training. It needs one GPU, and its TP and CP variants need up to four:

```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --extra mcore --group test pytest --mcore-only tests/unit/models/megatron/test_shared_prefix_logprob_oracle.py
```

The Megatron lane scripts (`tests/unit/L0_Unit_Tests_Megatron_*.sh`) export `NRL_REQUIRE_SHARED_PREFIX_TESTS=1`. With it, a `--mcore-only` run fails when the installed Megatron-LM lacks the shared-prefix modules. The oracle then fails instead of skipping when it has no GPU, and so do its TP 2 and CP 2 variants when they have fewer than two GPUs. The variant with TP 2 and CP 2 together needs four GPUs. It skips in a lane that has fewer.

A pair of test-suite scripts runs the feature end to end on one node with Nemotron-Nano-12B-v2. The two runs differ only in `policy.sequence_packing.share_prefixes`, and a fixed few-shot block makes the prompt most of each row:

- `tests/test_suites/llm/grpo-nano-v2-12b-1n8g-megatron-single-controller-token_capture.sh` is the dense arm. It checks `token_mult_prob_error`, the reward at step 30 against a floor, and that no rollout became a placeholder row.
- `tests/test_suites/llm/grpo-nano-v2-12b-1n8g-megatron-single-controller-token_capture-shared-prefix.sh` is the shared arm. It runs the checks of the dense arm. It then checks that `timing/train/policy_and_reference_logprobs` and `timing/train/policy_training` stay below 0.8 times the dense arm's medians.

Both scripts are listed in `tests/test_suites/disabled.txt`, so the nightly suite does not run them. The reward floor and the two dense medians are values in the scripts, taken from a run of the dense arm. A script fails after training while one of its values is empty.
