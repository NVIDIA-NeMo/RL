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
| MTP and speculative decoding | disabled (see the MTP variants below) |

The recipe enables raw vLLM log probabilities, truncated importance sampling
in the range 0.5 to 2.0, in-flight weight updates, and the existing NeMo RL
policy-to-vLLM refit path.

The visual encoder and projection remain trainable. The sound encoder and
projection are frozen for this image-and-text workload. The model's own chat
template is passed to both the Hugging Face tokenizer and the vLLM chat server.

## MTP variants

Two recipes inherit the base above and add Multi-Token Prediction. Both keep
its parallelism, data, and batch settings, so the table applies unchanged
apart from the rows below.

| Recipe | MTP head | Speculative decoding |
|---|---|---|
| [`...-async-gym.v1.yaml`](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.yaml) | absent | disabled |
| [`...-async-gym-mtp.v1.yaml`](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym-mtp.v1.yaml) | trained (`mtp_loss_scaling_factor: 0.1`) | disabled |
| [`...-async-gym-mtp-specdec.v1.yaml`](../../examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym-mtp-specdec.v1.yaml) | trained | MTP drafter, 5 tokens |

The checkpoint ships one physical MTP layer, so `mtp_num_layers: 1` with
`mtp_use_repeated_layer: false` maps directly onto it. `mtp_detach_heads: true`
keeps MTP gradients out of the backbone, and also tags the MTP parameters into
their own gradient-norm group, so they are clipped separately rather than
against the backbone's norm. Set `mtp_loss_scaling_factor: 0.0` to keep the
head present for speculative decoding without training it.

The speculative-decoding variant additionally sets `mamba_cache_mode: align`.
Super is a hybrid Mamba model, so the recurrent SSM state has to roll back
when draft tokens are rejected; the default (`none`) does not do that.

Point the launcher at either recipe with `CONFIG_PATH`:

```bash
CONFIG_PATH=examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym-mtp.v1.yaml \
bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh
```

## Launch

The launcher ships placeholders rather than defaults: `MODEL_PATH`,
`TRAIN_PATH`, `CONTAINER`, `SANDBOX_CONTAINER`, `PERSISTENT_CACHE`, and
`SLURM_ACCOUNT` all start as `/path/to/...` or `your_slurm_account`
(`VAL_PATH` follows `TRAIN_PATH` unless set). A preflight check rejects any
still holding a placeholder, listing them, before it reaches `mkdir` or
`sbatch`.

The container is one of them — there is no working default image. Build or
supply one with the Python 3.13 runtime the current main branch requires; the
older Python 3.12 handoff image is not compatible with the current lockfile.

Export a real value for each:

```bash
MODEL_PATH=/path/to/nemotron-super-omni-hf \
TRAIN_PATH=/path/to/super-omni-gym.jsonl \
CONTAINER=/path/to/nemo-rl-super-omni.sqsh \
SANDBOX_CONTAINER=/path/to/nemo-skills-sandbox.sqsh \
PERSISTENT_CACHE=/shared/cache/nemo-rl-super-omni \
SLURM_ACCOUNT=your_account \
SLURM_PARTITION=batch \
bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh
```

### Weights & Biases

The launcher turns W&B logging on, and the driver creates the run before any
worker starts, so a missing credential ends the job about two minutes into a
full allocation. Clusters that do not mount `/home` into the training
container cannot read a `wandb login` credential from `~/.netrc`; only
`WANDB_API_KEY` in the submitting environment reaches the job. Pick one:

```bash
export WANDB_API_KEY=<key>       # log live
export WANDB_MODE=offline        # log locally, `wandb sync <run-dir>` later
EXTRA_HYDRA_ARGS="logger.wandb_enabled=false"   # skip W&B entirely
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

## Migration notes

The old training directory carried local vLLM and Megatron patches. This
integration uses the repository-pinned Gym, Megatron Bridge, Megatron-LM, and
vLLM paths instead. The model-specific behavior that remains necessary is
expressed through typed configuration and runtime code: RADIO CPE controls,
vision and sound freeze flags, explicit MTP disablement, refit buffer sizing,
the model's serving chat template, and Super-recipe-gated normalization of its
dynamic-resolution image tensors.

The matching validation driver is
[`vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.sh`](../../tests/test_suites/vlm/vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-tp8ep16cp2-async-gym.v1.sh).
It remains a manually invoked functional test because the production topology
uses 16 nodes. The 8-node command above completed an end-to-end optimizer step
and refit in Slurm job `1545077`.
