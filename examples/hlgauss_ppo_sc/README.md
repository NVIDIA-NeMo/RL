# HL-Gauss PPO on SingleController

`binhu/hlgauss_ppo_sc` is the **SingleController** branch of HL-Gauss PPO.
The separate `binhu/hlgauss_ppo` branch retains the legacy PPO implementation.
Here, `sc` in the branch name means SingleController, not that score centering
must be enabled. Every recipe below uses SingleController PPO and its `TQValue`
critic. The entrypoint is named `run_grpo_single_controller.py` for historical
reasons; a `ppo:` configuration selects PPO, GAE and the critic.

The upstream repository still contains other algorithms and shared legacy
helpers. They are not alternate entrypoints for these recipes. Do not launch
these recipes with `examples/run_ppo.py`.

## Configuration layers

| Recipe | Additional behavior |
|---|---|
| `nano_ctrunc.yaml` | HL-Gauss critic, independent critic row masks, privileged critic, subagent segment rows, existing loop reward penalties; actor retains overlong filtering |
| `nano_sc.yaml` | Score centering, sampler top-32 capture, fused packing; actor CP4, critic CP2 |
| `nano_sc_failures.yaml` | Failure classification and selective context-limit unmasking; poisoned infrastructure samples excluded |
| `nano_sc_token_credit.yaml` | Optional post-normalization malformed/tool-failure advantage caps |

All defaults resolve to files in this checkout. There is no runtime dependency
on an experiment snapshot, another local branch, a private launch script or an
experiment's W&B name. Model weights, prepared datasets, compatible dependencies
and a sandbox service are still required.

The example topology is 24 nodes with 4 GPUs/node: 8 training and 16 generation
nodes, TP4/EP8; actor/critic each update for one epoch. PPS128/GPP1/GBS128 is
explicit. These are Nano example settings, not inferred hardware requirements
for other models. For PPS64/GPP4/GBS256, override both batch sizes and set
`async_rl.max_inflight_prompts=64`, `async_rl.max_buffered_rollouts=128` and
`async_rl.min_groups_for_streaming_train=64`; otherwise changing GPP can multiply
in-flight rollout concurrency unexpectedly.

## Start and resume

Follow the root repository installation guide and initialize the pinned
submodules (`git submodule update --init --recursive`). Use a matching Megatron /
vLLM environment. The model-specific token IDs in the SWE base recipe are for
Nano; they are not portable to Qwen or a different tokenizer unchanged.

Inside your allocated environment, with the checkout visible on all workers:

```bash
export MODEL_PATH=/path/to/actor-hf
export TRAIN_DATA_PATH=/path/to/prepared-swe-train.jsonl
export CRITIC_CHECKPOINT=/path/to/compatible-pretrained-critic-checkpoint
export OPENSANDBOX_DOMAIN=https://your-sandbox-endpoint
# Load OPENSANDBOX_API_KEY from your own credential file/environment.
bash examples/hlgauss_ppo_sc/run.sh nano_ctrunc.yaml \
  checkpointing.checkpoint_dir=/path/to/new-run \
  logger.log_dir=/path/to/new-run/logs
```

The wrapper launches the driver; it does **not** allocate or submit Slurm jobs.
Your scheduler/container launcher supplies those resources. `ENV_FILE`, if set,
is sourced before unsetting `RAY_LOG_SYNC_FREQUENCY`. Do the same before starting
Ray in any outer launcher. Never enable periodic Ray log copying between Lustre
directories. No job identifiers, account names or credentials are shipped here.

For network-disabled sandboxes, configure the offline OpenCode installation
cache/binary supported by the Gym provider in your deployment; the base SWE
recipe documents those fields. Model and sandbox services must be reachable.

`CRITIC_CHECKPOINT` is a **weights-only warm start**; it does not imply inheritance
of that checkpoint's optimizer. Leave it unset only when intentionally starting
with a fresh critic. Full PPO continuation uses the same
`checkpointing.checkpoint_dir` and restores actor/critic, optimizers and saved
training state through SingleController. Do not seed a new run by changing a
warm-start path and expect a full resume.

Offline critic pretraining uses **SingleController rollout banks**, not legacy
PPO dumps:

```bash
unset RAY_LOG_SYNC_FREQUENCY
uv run examples/run_critic_pretrain_single_controller.py \
  --config examples/hlgauss_ppo_sc/critic_pretrain.yaml \
  policy.model_name=/path/to/model \
  'critic_pretrain.bank_dirs=[/path/to/rollout-bank]' \
  checkpointing.checkpoint_dir=/path/to/critic-run
```

The pretraining entrypoint allocates critic workers only. Its supervision uses
the bank's saved labels; adding online failure rules to a recipe does not
retroactively reclassify old bank data. Collect/rebuild appropriately labeled
banks when changing penalties.

The head is explicitly HL-Gauss101 on [-0.1, 1.1], default 0.75-bin sigma and
`prior_success=0.5`. This initialization option adds a bias parameter; it does
not force all later values to 0.5. A warm-start checkpoint must match its head
shape, support and bias signature. Do not silently reinterpret an MSE head.

## Mask, reward and advantage semantics

- `critic_train_on_env_masked` and `critic_train_on_truncated` affect the critic
  row weights only. They do not change actor masks or reward labels. Original
  placeholder/padding and sequence-logprob-error exclusions remain in force.
- Context-limit handling in `nano_sc_failures.yaml` preserves the verifier
  reward (`reward: null`) and removes the corresponding mask. The recipe also
  disables the final blanket overlong filter, which would otherwise mask it again.
- A timeout **without** context-limit remains actor-masked. When both rules
  fire, the configured context-limit rule also removes the timeout rule.
- Infrastructure, provider OOM and incomplete verifier/agent failures matching
  `invalid_sample_failure_kinds` remain invalid. Poisoned receipts become
  placeholders; ctrunc must not restore them as valid critic observations.
- A row being actor-masked does not by itself mean it is absent from GAE's
  `normalize_over=all_rows` statistics. This recipe preserves that baseline
  normalization choice; it does not silently switch to valid-row whitening.
- Optional token credit caps apply **after** batch normalization, only to actor
  training tokens: malformed turns have an upper bound of -2; configured tool
  failures lose positive advantage. Do not normalize again afterward. Critic
  returns and row masks are unchanged. This recipe replaces whole-rollout
  malformed zeroing rather than stacking it with the token cap. Loop reward
  zeroing is a separate rule and remains inherited.

Score centering includes sampler head/tail handling, ordinary and fused shared-
logits backward protection, and CP-block top-k transport. The Nano packing path
uses zigzag CP layout. This is not an NT4/MIMO contiguous-layout port. The
supported SC recipe uses temperature1, top-p1 and no top-k sampling filter;
filtered-distribution sampler replay is not included. SC does not imply adding
sampler KL, Muon, a different TIS threshold or forcing PPO ratios to one.

## Scope and provenance

The base is Jiaqi's public SingleController SWE lineage at
`5666961588e02fe7e3c038d7e07d736e592858c8`, before local cluster migration notes.
The HL-Gauss/ctrunc/diagnostics and score-centering backport are consolidated
from the exercised Nano implementation. Jiaqi's NT4 backport was audited at
`8ca42d1994a13e6275cdbbdee88a53d8186e9a4d`. Initial SC work originated in
Tugrul's `tkonuk/rl-research`; Jiaqi adapted it and added fixes. This is a
functional Nano backport, not a merge of the NT4 model/optimizer recipe.

The pinned Gym dependency already provides context-limit reporting and token
capture. Local evaluation-only certificate/network regrading and experimental
Gym behavior annotations are not part of this training branch.

B0, rootbound, rootcal, sibling-value corrections, group/length reweighting,
critic FIFO replay and turn-decay research variants are not enabled or imported
by this branch's recipes. Existing upstream shared helpers are retained where
SingleController depends on them.

Unit tests cover categorical targets/decoding, masks, rule precedence, token
caps, CP transport and SC loss/gradient parity. Recipe tests validate the actual
SingleController PPO schema. CPU tests are not a full GPU/Slurm smoke test;
hardware, dependency and sandbox deployment parity must be checked when moving
to another cluster.

## Consolidation validation (2026-11-18)

- 289 CPU cases across HL-Gauss, token credit, failure rules, rollout banks,
  offline critic pretraining and SingleController PPO setup/recipes passed
  (the corrected pretraining recipe was rechecked in the six-recipe/mask suite).
- Seven additional SC loss, top-k gather and shared-logits backward cases passed.
- Ruff checks on imported Python files and the new recipe tests, shell syntax
  checks and `git diff --check` passed.
- The login test environment uses PyTorch2.9.1+cpu. CP block broadcast tests
  passed there, but microbatch/fetch tests require `torch.distributed.all_gather_single`,
  which that installation lacks. Full multi-rank CP/SC parity was not certified
  in this consolidation. Use the matching training environment; no compatibility
  shim was added to the algorithm to hide this dependency.
- No GPU/Slurm training smoke or new evaluation was launched by this branch
  consolidation. The unchanged upstream submodule pins are retained.
