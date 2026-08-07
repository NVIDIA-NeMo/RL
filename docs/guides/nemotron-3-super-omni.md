# Nemotron 3 Super Omni

This guide covers asynchronous multimodal GRPO for the Nemotron 3 Super Omni
120B-A12B model. The migration recipe uses a Megatron policy, non-colocated
vLLM generation, and NeMo Gym resources for math, multiple choice, GUI
coordinates, and string matching.

## Migration recipe

Use
[`vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.yaml`](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.yaml).
Its intended production topology and batching match the Super V3 training
handoff:

| Setting | Value |
|---|---|
| Total allocation | 16 nodes, 8 GPUs per node |
| Policy parallelism | TP=8, EP=16, CP=2 |
| Generation allocation | 8 non-colocated nodes |
| Prompts and generations | 256 prompts, 16 generations per prompt |
| Train global batch size | 4096 |
| Maximum sequence length | 16,384 |
| Async policy lag | at most one version |
| MTP and speculative decoding | disabled |

The recipe enables raw vLLM log probabilities, truncated importance sampling
in the range 0.5 to 2.0, in-flight weight updates, and the existing NeMo RL
policy-to-vLLM refit path.

The visual encoder and projection remain trainable. The sound encoder and
projection are frozen for this image-and-text workload. The model's own chat
template is passed to both the Hugging Face tokenizer and the vLLM chat server.

## Launch

The launcher deliberately has no site-specific checkpoint, data, container,
cache, or Slurm-account defaults. Export each required path explicitly. The
container's Python and Ray versions must match the repository lock; when an
older image is unavoidable, `CLUSTER_VENV` can point at a shared venv containing
the locked driver runtime and the launcher will use its base Python for rebuilt
actor environments:

```bash
MODEL_PATH=/path/to/nemotron-super-omni-hf \
TRAIN_PATH=/path/to/super-omni-gym.jsonl \
CONTAINER=/path/to/nemo-rl-super-omni.sqsh \
SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
PERSISTENT_CACHE=/shared/cache/nemo-rl-super-omni \
SLURM_ACCOUNT=your_account \
SLURM_PARTITION=batch \
CLUSTER_VENV=/shared/venvs/nemo-rl-driver \
NRL_FORCE_REBUILD_VENVS=true \
GYM_SKIP_VENV_IF_PRESENT=false \
bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh
```

The default rebuilds Gym service venvs rather than trusting image-baked
dependencies. Set `GYM_SKIP_VENV_IF_PRESENT=true` only when the image's Gym
venvs have already been validated against the locked Python and Ray versions.
Submitted runs use a commit-suffixed, locked snapshot and record source,
submodule, container, model, and runtime metadata beside it.

### Weights & Biases

The launcher turns W&B logging on, and the driver creates the run before any
worker starts, so a missing credential ends the job about two minutes into a
full allocation. Clusters that do not mount `/home` into the training
container cannot read a `wandb login` credential from `~/.netrc`; only
`WANDB_API_KEY` in the submitting environment reaches the job. Pick one:

```bash
export WANDB_API_KEY=<key>       # log live
export WANDB_MODE=offline        # log locally, `wandb sync <run-dir>` later
EXTRA_HYDRA_ARGS="logger.wandb_enabled=false"  # skip W&B entirely
```

The launcher checks this before submitting and refuses to burn an allocation
on a run that cannot log.

Set `DRY_RUN=true` to print the complete training command and `sbatch`
invocation without submitting. Use `EXTRA_HYDRA_ARGS` for Hydra overrides, for
example a short validation run:

```bash
DRY_RUN=true \
EXTRA_HYDRA_ARGS="grpo.max_num_steps=1 checkpointing.enabled=false" \
bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh
```

### Fast 8-node optimizer-step validation

This smoke keeps the policy topology intact while using 4 generation nodes,
4 policy-training nodes, 2 prompts, and one optimizer step:

```bash
EXP_NAME=smoke-super-omni-mtp-off-8n \
SBATCH_NUM_NODES=8 \
SLURM_TIME_LIMIT=00:30:00 \
EXTRA_HYDRA_ARGS="cluster.num_nodes=8 \
policy.generation.colocated.resources.num_nodes=4 \
grpo.max_num_steps=1 \
grpo.num_prompts_per_step=2 \
grpo.num_generations_per_prompt=16 \
policy.train_global_batch_size=32 \
policy.generation.max_new_tokens=2048 \
policy.megatron_cfg.mtp_num_layers=0 \
policy.generation.mcore_generation_config.num_speculative_tokens=0 \
checkpointing.enabled=false \
logger.wandb_enabled=false \
logger.tensorboard_enabled=false \
logger.monitor_gpus=false" \
bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh
```

Treat the run as successful only if its driver log shows generation, policy
logprob, backward/optimizer, and policy-to-vLLM refit completion. A Slurm
`COMPLETED` state alone is not sufficient.

## Image MOPD

The public image-MOPD recipe extends the MTP-disabled Super configuration with
a non-colocated Super teacher:

[`mopd-nemotron-super-omni-120ba12b-10n8g-megatron-tp8ep16cp2-async-gym.v1.yaml`](../../examples/configs/recipes/vlm/mopd-nemotron-super-omni-120ba12b-10n8g-megatron-tp8ep16cp2-async-gym.v1.yaml).

It uses ten 8-GPU nodes: one for vLLM generation, one for teacher logprobs, and
eight for the packed TP=8, EP=16, CP=2 policy. The default teacher is the same
checkpoint as the student, making the first-step OPD advantage a near-zero
plumbing check. Override
`on_policy_distillation.teacher_model_by_agent_name.circle_count_simple_agent`
with a stronger Omni checkpoint for real distillation.

Generate deterministic NeMo-Gym `circle_count_simple_agent` rows with one
`input_image` and the matching `agent_ref`:

```bash
python examples/nemo_gym/nemotron-3-super-omni/prepare_circle_count_mopd_data.py \
  --out /shared/data/circle_count_train.jsonl \
  --num-samples 512
```

Then launch:

```bash
MODEL_PATH=/path/to/nemotron-super-omni-hf \
TEACHER_MODEL_PATH=/path/to/nemotron-super-omni-teacher-hf \
TRAIN_PATH=/path/to/circle_count_train.jsonl \
CONTAINER=/path/to/nemo-rl-super-omni.sqsh \
PERSISTENT_CACHE=/shared/cache/nemo-rl-super-omni \
SLURM_ACCOUNT=your_account \
WANDB_ENTITY=your_entity \
WANDB_API_KEY=... \
bash examples/nemo_gym/nemotron-3-super-omni/run_mopd_circle_count.sh
```

`SANDBOX_CONTAINER` is optional and only needed by Gym resources that launch a
NeMo-Skills sandbox; `circle_count_simple_agent` does not.

`TEACHER_MODEL_PATH` is optional. When omitted, the teacher defaults to
`MODEL_PATH` for a self-distillation plumbing check. A separate teacher must use
a tokenizer, processor, chat template, vocabulary, and multimodal architecture
compatible with the student.

Before the full run, use the four-node, three-step exact-stack smoke:

```bash
EXP_NAME=mopd-super-omni-circle-smoke \
CONFIG_PATH=examples/configs/recipes/vlm/mopd-nemotron-super-omni-120ba12b-4n8g-smoke.v1.yaml \
SBATCH_NUM_NODES=4 \
bash examples/nemo_gym/nemotron-3-super-omni/run_mopd_circle_count.sh
```

The implementation forwards the same multimodal rows to student and teacher,
including teacher routing and data-parallel padding, and invalidates vLLM's
weight-dependent encoder-output cache after each refit. Validate at least three
steps: with one-step async staleness, step three is the first unambiguous
training batch generated after the first weight update.

## Migration notes

The old training directory carried local vLLM and Megatron patches. This
integration uses the repository-pinned Gym, Megatron Bridge, Megatron-LM, and
vLLM paths instead. The model-specific behavior that remains necessary is
expressed through typed configuration and runtime code: RADIO CPE controls,
vision and sound freeze flags, explicit MTP disablement, refit buffer sizing,
the model's serving chat template, and Super-recipe-gated normalization of its
dynamic-resolution image tensors.

Use the four-node smoke command above as the minimum functional acceptance test.
It completed three optimizer/refit steps in Slurm job `1633404`. The production
topology subsequently completed seven consecutive optimizer/refit steps in
ten-node Slurm job `1637034`.
