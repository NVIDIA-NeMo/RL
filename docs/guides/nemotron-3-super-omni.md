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

The launcher defaults to the checkpoint, blended Gym data, Slurm account, and
cache settings used by the migration source script. Its default NeMo RL image
uses the Python 3.13 runtime required by the current main branch; the older
Python 3.12 handoff image is not compatible with the current lockfile. Every
value can be overridden with an environment variable:

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
EXTRA_OVERRIDES="logger.wandb_enabled=false"   # skip W&B entirely
```

The launcher checks this before submitting and refuses to burn an allocation
on a run that cannot log.

Set `DRY_RUN=true` to print the complete training command and `sbatch`
invocation without submitting. Use `EXTRA_OVERRIDES` for Hydra overrides, for
example a short validation run:

```bash
DRY_RUN=true \
EXTRA_OVERRIDES="grpo.max_num_steps=1 checkpointing.enabled=false" \
bash examples/nemo_gym/nemotron-3-super-omni/super_omni_launch.sh
```

### Fast 8-node optimizer-step validation

This smoke keeps the policy topology intact while using 4 generation nodes,
4 policy-training nodes, 2 prompts, and one optimizer step:

```bash
EXP_NAME=smoke-super-omni-mtp-off-8n \
SBATCH_NUM_NODES=8 \
SLURM_TIME_LIMIT=00:30:00 \
EXTRA_OVERRIDES="cluster.num_nodes=8 \
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
