# Claude Code subagent training

Train one model as both a coordinator and its child agents through Gym's existing
Claude Code harness. The coordinator delegates two independent calculations, waits
for the children, and combines their results. Gym grades the final sum locally.
The eight bundled tasks are synthetic arithmetic examples, not a benchmark.

**Runtime validation passed:** this configuration completed 20 training steps with
W&B and a saved-tensor audit of coordinator and child calls. It still requires the
Gym prerequisites listed below; an unmodified Gym checkout is not qualified.

## What is trained

```yaml
token_capture:
  enabled: true
  forest_training: true
  selection_policy: all_captured
```

All completed, eligible captured calls from a valid rollout are selected, including
child contexts absent from the coordinator's returned transcript. Independent roots
become separate training paths; shared generated prefixes receive loss once. The
final-answer reward and rollout advantage apply to the coordinator and its children.
This example does not assign different rewards to individual agents.

`all_captured` also selects completed discarded attempts; it does not infer whether
a harness liked each generation. Incomplete capture can invalidate the entire owner.
The prompts ask for exactly two children, but the model can ignore that instruction.
A run without observed children does not validate subagent training.

Foreground means Claude Code returns the child result to the coordinator before it
continues. It does not merge their contexts. This avoids the background-notification
loop seen in the earlier experiment. Background execution and forced compaction are
not qualified by this example. If the harness compacts, the same forest machinery
handles the resulting context discontinuities.

## Prerequisites

- A NeMo-RL training environment with the `nemo_gym` extra and the checked-out Gym
  submodule, plus the dependencies required by `resources_servers/math_with_judge`.
- A local Nano Omni HF checkpoint and its chat template. This recipe retains the
  previous Nano Omni topology: three nodes with eight GPUs each; two learner nodes
  (TP8/EP16), and one generation node (four TP2 replicas). It is not a hardware-minimal
  recipe. Changing the model requires reviewing its template, parsers and topology.
- Git for the sandbox repository used by native worktree-isolated subagents.
- Node.js/npm for Gym's installation of the pinned Claude Code CLI **2.1.303**.
  Model calls go through Gym to the local vLLM workers, not an Anthropic model.
- A Python executable in the allocated runtime that imports Gym, FastAPI, aiohttp,
  Pydantic and OmegaConf; this is the sandbox runner interpreter.
- The checkout and example assets available at identical absolute paths on each
  node. This example uses Gym's **local subprocess provider**, so it is intended
  for these trusted bundled tasks, not untrusted coding tasks.

This example needs the following Gym compatibility fixes (tracked separately from
this examples change):

1. Permit Messages requests on source-history capture.
2. Canonicalize captured Messages output with the same converter as the next request,
   so ordinary continuations retain their captured parent.
3. Pass capture mode into `harness_agent`'s inner collector, without credentials.
4. Resolve relative local-provider working directories under their sandbox workspace.

The capture-aware Claude transcript parser is already in that Gym revision. No new
harness or agent loop is introduced by this example. The NeMo-RL dependency pin and
lockfile are unchanged; the prerequisite fixes must be integrated before this can be
advertised as runnable on an unmodified checkout.

## Launch

Run from the NeMo-RL root **inside a GPU allocation**, using the repository's existing
Ray/container launch workflow. For multi-node Slurm, put the following environment
and command in that workflow; do not run training on a login node.

```bash
export SUBAGENT_EXAMPLE_DIR="$PWD/examples/nemo_gym/subagent_training"
export SUBAGENT_RUN_DIR="$PWD/logs/subagent-training"
export SUBAGENT_MODEL=/absolute/path/to/nano-omni-checkpoint
export SUBAGENT_CHAT_TEMPLATE="$SUBAGENT_MODEL/chat_template.jinja"
export SUBAGENT_SANDBOX_PYTHON=/absolute/path/to/runtime/bin/python
mkdir -p "$SUBAGENT_RUN_DIR/tmp" "$SUBAGENT_RUN_DIR/sandboxes"

uv run python examples/run_grpo_single_controller.py \
  --config "$SUBAGENT_EXAMPLE_DIR/grpo.yaml"
```

W&B is enabled. Authenticate it through the normal runtime environment; optionally
set `WANDB_ENTITY` and `SUBAGENT_RUN_NAME`. All nodes need the configured paths and
runtime executable. The setup commands stage the coordinator and solver profiles
into each sandbox and commit them in a new sandbox-local Git repository, so native
worktree isolation can also load the profiles. No personal Claude configuration is needed.

The learner, serving defaults, and Gym request settings all set
`enable_thinking: false` and `truncate_history_thinking: false`. Keep them aligned.
The output budget is 2,048 tokens per model request. Child `maxTurns: 1` does not
prevent Claude Code's native token-limit recovery requests. The CLI episode has a
1,800-second timeout and the wrapper has a 1,900-second timeout.

The in-order sampler uses `max_lookahead_versions: 0`: each rollout batch is
generated with the learner's current weights before its update. This keeps the
example's numerical validation separate from policy-version lag.

Checkpoints are enabled every five steps, including data-plane state so unconsumed
rollouts can be recovered. The inherited training time budget is
explicitly 3h40m; a clean early shutdown at that budget is **not** a 20-step pass.
Use the existing checkpoint/resume workflow if the allocation cannot finish.

## Validation

When changing this recipe, validate it for 20 steps with W&B and retained rollout dumps.
Check all of the following; scheduler completion and a successful final answer alone
are insufficient:

- Native Claude observations show the two child invocations and their model calls.
- Those child calls and coordinator calls appear in selected training paths, with
  generated tokens contributing to loss exactly once, including shared ancestors.
- All 20 optimizer steps and their training dumps exist. Report nonzero gradients
  when sibling rewards differ; an all-zero-gradient run does not demonstrate learning.
- Report valid-owner fraction and any capture timeouts, rather than silently
  dropping invalid episodes from the success claim.
- Report aggregate and maximum sequence logprob error **before masking**, plus the
  number of masked rows. Forest masking applies the 1.05 threshold to the
  token-weighted error across each logical rollout, then masks all its rows together.
  Physical-row maxima are separate diagnostics.

The paired-runtime validation on November 5, 2026 passed:

| Check | Result |
| --- | --- |
| Training steps | 20, including 12 nonzero-gradient updates |
| Logical rollouts | 80 valid; all 80 native CLI runs completed successfully |
| Captured calls used in training | 357, including 175 child calls |
| Native child coverage | Every consumed rollout had at least two observed child invocations |
| Loss accounting | 23,785 generated tokens, each counted exactly once |
| Serving/learner version lag | Zero for every real training row |
| Numerically masked rows | Zero |
| Raw aggregate TMPE from saved tensors | 1.00459 |
| Maximum whole-rollout TMPE | 1.01936 |
| Maximum physical-row TMPE | 1.16357; whole-rollout masking uses the pooled value above |

[W&B validation run](https://wandb.ai/nvidia/nemo-rl-subagent-training/runs/kat6fu2v).
The training job completed all steps; its original audit hit a deleted temporary
child-output symlink. A corrected audit ran separately on the same saved tensors,
lineage manifests and persistent native observations and passed. Training was not
repeated to obtain that audit result.

The GPU run used RL `95095b7124f9b2526c12be772b04fac918ba0a47` and Gym
`5911d628c744d0bf43fe1d9ddf3bda72d59b0f9a` plus the prerequisite fixes. The PR
branches are rebased onto newer main revisions; their focused checks are recorded
separately from this earlier GPU result.

This validates the paired Gym prerequisite changes and this three-node configuration,
not arbitrary models, background agents, forced compaction, or separate child rewards.
An earlier one-batch-lookahead run masked one rollout after a policy update; that
run is retained as diagnostic evidence and is not the zero-lag qualification above.
