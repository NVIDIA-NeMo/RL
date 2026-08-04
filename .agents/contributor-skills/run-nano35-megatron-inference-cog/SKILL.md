---
name: run-nano35-megatron-inference-cog
description: Run the Nemotron-3.5-Nano ("nano v3.5") model on the OCI HSG GB200 cluster with NeMo-RL's Megatron (mcore) inference/generation backend, using the `cog` CLI and the prebuilt nightly image (no image build). Loads the local Megatron dist-checkpoint at /lustre/.../nemotron-3.5-nano-swe-step25/without_mtp and drives a GRPO/nemo-gym run from examples/nemo_gym/grpo_nanov3.yaml with policy.generation.backend=megatron.
when_to_use: "Run nano3.5 / nemotron-3.5-nano on GB200 with the megatron inference backend"; "smoke-test mcore inference for the nano v3.5 checkpoint on oci"; "run grpo_nanov3.yaml with megatron generation on the cluster"; reproducing/iterating on Megatron-Core generation for the nano-3.5 mamba-hybrid MoE model.
---

# Running nano-3.5 with the Megatron inference backend on GB200 via `cog`

Goal: run the **Nemotron-3.5-Nano** model (a mamba/attention-hybrid MoE) on
GB200 using NeMo-RL's **Megatron generation backend** (`DynamicInferenceEngine`),
loading the existing Megatron dist-checkpoint instead of converting from HF.

## Cog and cluster prerequisites

- Use `cog` from `~/cog` with the NeMo-RL repo profile installed. The profile
  must resolve `~/RL` to the prebuilt `nvcr.io/nvidian/nemo-rl:nightly` image
  and reuse `/opt/nemo_rl_venv`; do not build a new image or venv.
- Use the registered `oci-hsg` cluster. Its GB200 nodes are aarch64 and have
  4 GPUs per node. The QOS requires whole-node GPU allocations, so always pass
  `--gpus 4`, including jobs that use fewer GPUs internally.
- Run NeMo-RL commands from `/opt/nemo-rl`, which contains the complete
  submodules and consistent environment. The synced workspace has empty
  `3rdparty` submodule directories; copy local `examples/` and `nemo_rl/`
  changes into `/opt/nemo-rl` before running.
- Run `cog prepare-image` once for each new nightly image. The image is large;
  if import is terminated by the default Slurm limit, ensure the local cog
  enroot import uses `-t 04:00:00` or set `COG_IMPORT_TIME=04:00:00`.
- If workspace sync reports `Unsupported worktree entry type`, reinstall the
  local `~/cog` checkout containing the NeMo-RL submodule snapshot fix.

## Credentials

Whenever a Hugging Face, W&B, NGC, or other token is needed, first run:

```bash
source /Users/shanmugamr@nvidia.com/tokens
```

Use the exported environment variable required by the command. Never put token
placeholders or literal token values in this skill, and never print the file's
contents.

## Inputs (facts that matter)

- **Weights (Megatron dist-ckpt, torch_dist)**:
  `/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3.5-nano-swe-step25/without_mtp`
  → load via `policy.pretrained_checkpoint.format=megatron_lm`.
- **HF config + tokenizer** (architecture + tokenizer resolution):
  `/lustre/fsw/portfolios/llmservice/users/dmosallanezh/nemo-evaluator-rundirs/nano_v35/conversions/geshen_ultra_rl_v5_kd600_step30_fixedpath_20260520_1130/hf`
  → `policy.model_name` and `policy.tokenizer.name`.
- Both live under `/lustre/fsw/portfolios/llmservice`, **not** the cog scratch
  root (`/lustre/fsw/portfolios/coreai/...`). cog only auto-mounts scratch, so
  the llmservice path must be added with `COG_EXTRA_MOUNTS` (see below).
- **Reference config**: `examples/nemo_gym/grpo_nanov3.yaml`
  (entrypoint `examples/nemo_gym/run_grpo_nemo_gym.py`). It already carries a
  `policy.generation.mcore_generation_config` block; we flip
  `policy.generation.backend` to `megatron` and repoint paths + parallelism.

> The reference `grpo_nanov3.yaml` is sized for the 30B-A3B nano v3 across many
> nodes (train tp=2/ep=8/pp=2/cp=4, cluster 32×8). nano-3.5 differs in
> architecture and here we run a **small footprint** to exercise mcore
> inference, so the parallelism/cluster values below are overridden. Scale up
> deliberately, not by copying the base yaml's values.

## Model architecture → NeMo-RL config mapping

The nano-3.5 Megatron model provider args (from the raw mcore inference command)
map onto NeMo-RL config as follows. Architecture (hidden size, experts, mamba
dims, hybrid pattern, squared-relu, sigmoid router, etc.) is resolved by
megatron-bridge from the **HF `config.json`** at `policy.model_name`, so most of
these need no CLI flag — they must simply match the HF checkpoint. The ones that
are NeMo-RL knobs (parallelism + inference behavior) are listed explicitly.

| mcore arg | NeMo-RL setting |
|---|---|
| `--model-provider mamba`, `--hybrid-layer-pattern ...`, `--mamba-*`, `--hidden-size 2688`, `--ffn-hidden-size 1856`, `--num-experts 128`, `--moe-ffn-hidden-size 1856`, `--moe-router-topk 6`, `--moe-shared-expert-intermediate-size 3712`, `--squared-relu`, `--moe-router-score-function sigmoid`, `--moe-router-enable-expert-bias`, `--moe-router-topk-scaling-factor 2.5`, `--num-query-groups 2`, `--kv-channels 128`, `--position-embedding-type none`, `--untie-embeddings-and-output-weights`, `--vocab-size 131072`, `--make-vocab-size-divisible-by 128` | Come from the HF `config.json` at `policy.model_name` (the `/hf` dir). If the HF config is missing/mismatched, patch via `policy.hf_config_overrides.<field>=...`. Do **not** hand-set these unless a mismatch surfaces. |
| `--tensor-model-parallel-size 1` | `policy.generation.mcore_generation_config.tensor_model_parallel_size=1` |
| `--expert-model-parallel-size ${WORLD_SIZE}` | `...mcore_generation_config.expert_model_parallel_size=<#gen GPUs>` (128 experts must be divisible by it; 4 → 32 experts/rank) |
| `--expert-tensor-parallel-size 1` | `...mcore_generation_config.expert_tensor_parallel_size=1` |
| `--inference-dynamic-batching-num-cuda-graphs -1` | `++policy.generation.mcore_generation_config.num_cuda_graphs=-1` (base default is 4; `-1` captures the full set of graph sizes → less decode padding, faster generation). |
| `--cuda-graph-impl local` | `++policy.generation.mcore_generation_config.cuda_graph_impl=local` (already the default). |
| `--cuda-graph-scope full_iteration_inference` | `++policy.generation.mcore_generation_config.inference_cuda_graph_scope=block`. Megatron **migrates** the deprecated `full_iteration_inference` scope to `inference_cuda_graph_scope=block` (`megatron/core/transformer/cuda_graph_config.py`), so `block` is the equivalent NeMo-RL value. |
| `--transformer-impl inference_optimized` | `...mcore_generation_config.transformer_impl=inference_optimized`. **Only takes effect for NON-colocated generation.** A colocated run reuses the *training* model (which is forced to `transformer_engine`), so this key is silently ignored and you must instead set `moe_pad_experts_for_cuda_graph_inference=true` (see the "colocated vs inference_optimized" caveat below). |
| `--moe-grouped-gemm`, `--inference-grouped-gemm-backend vllm` | `policy.megatron_cfg.moe_grouped_gemm=true`, `policy.megatron_cfg.inference_grouped_gemm_backend=vllm` — set on `megatron_cfg` so they apply to the colocated model (the `mcore_generation_config` copies only apply non-colocated). |
| `--moe-token-dispatcher-type alltoall` | `policy.megatron_cfg.moe_token_dispatcher_type=alltoall` (base default `allgather`). Optionally `policy.megatron_cfg.inference_moe_token_dispatcher_type=alltoall` for the inference path. |
| `--moe-router-dtype fp32` | `policy.megatron_cfg.moe_router_dtype=fp32`. |
| `--mamba-inference-ssm-states-dtype fp32` | `++policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=float32`. |
| `--inference-dynamic-batching-buffer-size-gb 20` | `++policy.generation.mcore_generation_config.buffer_size_gb=70` (full scale). On the 4-GPU colocated smoke this OOMs — keep `buffer_size_gb=20` there. |
| `--seq-length / --max-position-embeddings / --inference-max-seq-length 73728` | `policy.max_total_sequence_length` (+ `...mcore_generation_config.max_model_len`). 73728 is large; start smaller (e.g. 2048) for a smoke, raise as memory allows. |
| `--parsers deepseek-r1-reasoning qwen3-coder-tool` | `...mcore_generation_config.parsers=[deepseek-r1-reasoning,qwen3-coder-tool]`. |
| `--moe-shared-expert-overlap` | `policy.megatron_cfg.moe_shared_expert_overlap=true` |
| `--tokenizer-type HuggingFaceTokenizer --tokenizer-model <hf>` | `policy.tokenizer.name=<hf dir or hub id, e.g. nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16>` |
| `--pretrained-checkpoint <dir>`, `--load <dir>`, `--use-checkpoint-args`, `--dist-ckpt-strictness log_unexpected` | `++policy.pretrained_checkpoint.path=<dir>` + `++policy.pretrained_checkpoint.format=megatron_lm`. The bridge reads model args from the dist-ckpt and tolerates extra/missing keys (no separate `--load`/strictness flag needed). |

`--inference-use-synchronous-zmq-collectives` needs no flag — NeMo-RL's worker
always builds the `InferenceConfig` with `use_synchronous_zmq_collectives=True`
(`megatron_worker.py`), so it is effectively always on. `--model-provider mamba`
is the standalone Megatron-LM inference-server entrypoint (not present in this
repo/image); NeMo-RL resolves the mamba/hybrid provider from the HF config
instead. `--mtp-use-repeated-layer` is moot: the `without_mtp` checkpoint is
already MTP-stripped.

## Run it

```bash
# 0. Sanity: profile resolves to nemo_rl + image = nightly
cog profile --repo ~/RL --run-name nano35-mcore-gen --cluster-name oci-hsg

# 1. One-time per image: import the nightly sqsh (cached afterwards).
cog prepare-image --repo ~/RL --cluster-name oci-hsg

# 2. Mount the llmservice checkpoint tree (weights + HF config/tokenizer) into
#    the container. cog only auto-mounts the coreai scratch root, so export this
#    before `cog submit` (it appends to --container-mounts for one-shot srun jobs).
export COG_EXTRA_MOUNTS='/lustre/fsw/portfolios/llmservice/users/ksanthanam:/lustre/fsw/portfolios/llmservice/users/ksanthanam,/lustre/fsw/portfolios/llmservice/users/dmosallanezh:/lustre/fsw/portfolios/llmservice/users/dmosallanezh'

# 3. Submit. Run from the image's /opt/nemo-rl (full submodules + consistent
#    venv); the synced workspace has empty 3rdparty submodules and `uv run`
#    fails there. Overlay the synced example config so local edits to
#    grpo_nanov3.yaml are picked up.
#
#    IMPORTANT: grpo_nanov3.yaml ships PLACEHOLDER data paths
#    (data.train.data_path=/path/to/train.jsonl) and a multi-env
#    env.nemo_gym.config_paths list. You must supply a real nemo-gym dataset.
#    The smoke command below prepares the workplace_assistant dataset in-job via
#    `ng_prepare_data` (needs HF_TOKEN + compute-node internet; GB200 nodes have
#    it) and points training at it — mirroring
#    tests/test_suites/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation-async-gym.sh.
#    For a real nano-3.5 run, point data.train/validation at the actual curated
#    dataset and keep the base yaml's env.nemo_gym.config_paths.
source /Users/shanmugamr@nvidia.com/tokens  # exports HF_TOKEN
MODEL_DIR=/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3.5-nano-swe-step25/without_mtp
HF_DIR=/lustre/fsw/portfolios/llmservice/users/dmosallanezh/nemo-evaluator-rundirs/nano_v35/conversions/geshen_ultra_rl_v5_kd600_step30_fixedpath_20260520_1130/hf

cog submit \
  --repo ~/RL \
  --cluster-name oci-hsg \
  --run-name nano35-mcore-gen \
  --gpus 4 --nodes 1 --ntasks-per-node 1 \
  --partition batch --time 02:00:00 \
  --job-name nano35-mcore-gen \
  --command "export HF_TOKEN=$HF_TOKEN; cp -rf examples/. /opt/nemo-rl/examples/ 2>/dev/null || true; cd /opt/nemo-rl && \
    ( [ -f 3rdparty/Gym-workspace/Gym/env.yaml ] || echo \"hf_token: \$HF_TOKEN\" > 3rdparty/Gym-workspace/Gym/env.yaml ) && \
    ( cd 3rdparty/Gym-workspace/Gym && uv run --no-sync ng_prepare_data '+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]' +output_dirpath=data/workplace_assistant +mode=train_preparation +should_download=true +data_source=huggingface ) && \
    jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/train.jsonl > /tmp/wa_train.jsonl && \
    jq -c '.responses_create_params.tools |= (.[0:1])' 3rdparty/Gym-workspace/Gym/data/workplace_assistant/validation.jsonl > /tmp/wa_val.jsonl && \
    uv run --no-sync python examples/nemo_gym/run_grpo_nemo_gym.py \
    --config examples/nemo_gym/grpo_nanov3.yaml \
    policy.model_name=$HF_DIR \
    policy.tokenizer.name=$HF_DIR \
    ++policy.pretrained_checkpoint.path=$MODEL_DIR \
    ++policy.pretrained_checkpoint.format=megatron_lm \
    policy.generation.backend=megatron \
    policy.max_total_sequence_length=2048 \
    policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    policy.generation.mcore_generation_config.expert_model_parallel_size=4 \
    policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
    policy.generation.mcore_generation_config.pipeline_model_parallel_size=1 \
    policy.generation.mcore_generation_config.context_parallel_size=1 \
    policy.generation.mcore_generation_config.buffer_size_gb=50 \
    policy.megatron_cfg.tensor_model_parallel_size=1 \
    policy.megatron_cfg.expert_model_parallel_size=4 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 \
    policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.moe_shared_expert_overlap=true \
    policy.generation.colocated.enabled=true \
    data.train.data_path=/tmp/wa_train.jsonl \
    data.validation.data_path=/tmp/wa_val.jsonl \
    'env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml]' \
    grpo.max_num_steps=1 grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 grpo.val_period=0 grpo.val_at_start=false \
    cluster.gpus_per_node=4 cluster.num_nodes=1 \
    logger.wandb_enabled=false logger.tensorboard_enabled=false"
```

Notes on the overrides:
- `--gpus 4` is mandatory (oci-hsg QOS requires whole-node GPU jobs); the 4 GPUs
  match `cluster.gpus_per_node=4` and `expert_model_parallel_size=4`.
- `expert_model_parallel_size` must divide `num_experts=128` and equal the number
  of colocated generation GPUs.
- Keep `pretrained_checkpoint` under `policy` (not `checkpointing`). Use `++` to
  add the key since the base yaml doesn't define it.
- Start with `max_total_sequence_length=2048` and step count 1 for a smoke; raise
  toward 73728 (the tuned standalone config's seq length) / more steps once it
  runs clean and memory allows.
- To make generation match the tuned standalone mcore config, append the tuned
  knobs from "Optimized mcore inference settings" (num_cuda_graphs=-1, alltoall
  dispatcher, vllm grouped-GEMM, fp32 router/mamba-ssm dtype, buffer_size_gb=20).
- If the run needs more than one node, add `--launcher ray` and raise `--nodes`,
  `cluster.num_nodes`, and the parallelism together (they must multiply to the
  total GPU count). Keep `--ntasks-per-node 1`. See "Multi-node runs
  (`--launcher ray`)" below — a plain multi-node `cog submit` (torchrun launcher)
  will **not** form a cross-node Ray cluster.

## Multi-node runs (`--launcher ray`)

NeMo-RL is **not** torchrun-launched: it needs a single driver on a head node
that fans work out to Ray actors spanning every physical node. cog's default
launch model (torchrun/SPMD — run the same command on every task, inject
`MASTER_ADDR`/`WORLD_SIZE`/`RANK`) does **not** build a cross-node Ray cluster,
so a plain `cog submit --nodes 2` for NeMo-RL either hangs on the placement
group (each node starts its own isolated single-node Ray cluster and can't place
`gpus_per_node * num_nodes` bundles) or silently degrades to duplicate
single-node runs.

Use **`cog submit --launcher ray`** (added to cog specifically for this). On an
`N`-node allocation it reproduces `ray.sub` semantics inside cog's single-`srun`
model: it starts `ray start --head` on node 0 with the fixed non-ephemeral ports
and NeMo-RL's custom `--resources` (`worker_units`, `nvlink_domain_<uuid>`,
`topo_rank`, `slurm_managed_ray_cluster`), starts `ray start --address=...` on
every other node, sets `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` + `ulimit -Sn 65535`,
polls `ray status` until `worker_units == gpus * nodes`, then runs the driver
**exactly once** on the head (which attaches via `ray.init(address="auto")`).
Nodes coordinate through sentinel files under
`$RUN_DIR/ray_launcher/$SLURM_JOB_ID`, so no overlapping `srun` or persistent
named containers are required.

Rules for `--launcher ray`:
- Always `--ntasks-per-node 1` (cog enforces this; one Ray launcher task per
  node). Still pass `--gpus 4` (oci-hsg whole-node QOS).
- Put the driver in `--command` (it runs only on the head).
- Put per-node prep in `--setup-command` (runs on **every** node before Ray
  starts). Overlay local `nemo_rl/`/`examples/` here so workers use the same
  code the driver does — the head's `--command` cp only affects the head.
- Scale `cluster.num_nodes`, parallelism, and `--nodes` together. For 2 nodes
  ×4 GPUs, `expert_model_parallel_size=8` (must divide `num_experts=128`).

Example — a colocated GRPO smoke on **2 nodes** (EP=8 across 8 GPUs):

```bash
export COG_EXTRA_MOUNTS='/lustre/fsw/portfolios/llmservice/users/ksanthanam:/lustre/fsw/portfolios/llmservice/users/ksanthanam,/lustre/fsw/portfolios/llmservice/users/dmosallanezh:/lustre/fsw/portfolios/llmservice/users/dmosallanezh'
MODEL_DIR=/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3.5-nano-swe-step25/without_mtp
HF_DIR=/lustre/fsw/portfolios/llmservice/users/dmosallanezh/nemo-evaluator-rundirs/nano_v35/conversions/geshen_ultra_rl_v5_kd600_step30_fixedpath_20260520_1130/hf

cog submit --repo ~/RL --cluster-name oci-hsg --run-name ray-2node-nano35-math \
  --gpus 4 --nodes 2 --ntasks-per-node 1 --partition batch --time 01:30:00 --job-name ray2node-nano35 \
  --launcher ray \
  --setup-command 'cp -rf examples/. /opt/nemo-rl/examples/ 2>/dev/null || true; cp -rf nemo_rl/. /opt/nemo-rl/nemo_rl/ 2>/dev/null || true' \
  --command "export PYTHONPATH=/opt/nemo-rl; cd /opt/nemo-rl && uv run --no-sync python examples/run_grpo.py \
    --config examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation.yaml \
    policy.model_name=$HF_DIR policy.tokenizer.name=$HF_DIR \
    ++policy.pretrained_checkpoint.path=$MODEL_DIR ++policy.pretrained_checkpoint.format=megatron_lm \
    policy.generation.backend=megatron policy.max_total_sequence_length=2048 policy.train_global_batch_size=8 \
    ++loss_fn.reference_policy_kl_penalty=0 \
    ++policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    ++policy.generation.mcore_generation_config.expert_model_parallel_size=8 \
    ++policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
    ++policy.generation.mcore_generation_config.pipeline_model_parallel_size=1 \
    ++policy.generation.mcore_generation_config.context_parallel_size=1 \
    ++policy.generation.mcore_generation_config.sequence_parallel=false \
    ++policy.generation.mcore_generation_config.buffer_size_gb=10 \
    ++policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference=true \
    policy.megatron_cfg.tensor_model_parallel_size=1 policy.megatron_cfg.expert_model_parallel_size=8 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=false policy.megatron_cfg.moe_shared_expert_overlap=true \
    policy.megatron_cfg.activation_checkpointing=true \
    ++policy.megatron_cfg.moe_pad_experts_for_cuda_graph_inference=true \
    policy.generation.colocated.enabled=true \
    grpo.max_num_steps=1 grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 grpo.val_at_start=false grpo.val_period=0 \
    cluster.gpus_per_node=4 cluster.num_nodes=2 checkpointing.enabled=false \
    logger.wandb_enabled=false logger.tensorboard_enabled=false logger.monitor_gpus=false"
```

The `[cog-ray]` lines in the Slurm stdout trace the bootstrap: head IP, per-node
resources, `worker_units online: X/8`, "all workers connected; launching driver",
then the normal NeMo-RL driver output. Debug tips:
- Job hangs at `worker_units online: 4/8` → a worker never joined. Check the
  head node's stdout and confirm both nodes see the same head IP; a worker that
  can't reach `head_ip:9900` never registers.
- `--launcher ray requires --ntasks-per-node 1` → drop `--ntasks-per-node` back
  to 1 (parallelism scales via `cluster.num_nodes` + EP/TP/PP, not tasks).
- Worker imports stale `nemo_rl`/`megatron` → move the `cp -rf` overlay into
  `--setup-command` (runs on all nodes), not `--command` (head only).

## Non-colocated generation with `transformer_impl=inference_optimized` (validated)

The standalone mcore config uses `--transformer-impl inference_optimized`. As
explained above, that impl is **unreachable in a colocated run** (colocated
reuses the `transformer_engine` training model). To actually run generation with
`inference_optimized` you must run **non-colocated**: dedicated generation GPUs
separate from the training GPUs. On oci-hsg (4 GPUs/node) that means **2 nodes**
— one whole node for generation, one for training — so it is a `--launcher ray`
job.

This has been validated end-to-end on GB200 (loads the dist-ckpt, warms up the
`inference_optimized` CUDA graphs, generates, does a **cross-node weight refit**,
and completes 3 GRPO steps). Working run:
`https://wandb.ai/shanmugamr/nano35-mcore-noncolo/runs/6ynwwxwy`.

How the GPU pool splits (see `grpo.py` `setup()` non-colocated branch): with
`cluster.num_nodes>1` and `colocated.enabled=false`, inference takes
`colocated.resources.num_nodes` **whole** nodes (and `resources.gpus_per_node`
must equal `cluster.gpus_per_node`), training gets the rest. So for 2 nodes:
gen = 1 node (4 GPUs, EP=4), train = 1 node (4 GPUs, EP=4).

Key differences vs. the colocated command:
- `policy.generation.colocated.enabled=false` +
  `++policy.generation.colocated.resources.gpus_per_node=4`
  `++policy.generation.colocated.resources.num_nodes=1`.
- `++policy.generation.mcore_generation_config.transformer_impl=inference_optimized`
  now actually takes effect (non-colocated builds a dedicated generation model via
  `megatron_cfg.update(mcore_generation_config)`).
- **DROP all `moe_pad_experts_for_cuda_graph_inference` overrides.** With
  `inference_optimized` that padding flag is *forbidden* (trips the
  `text_generation_controller.py` assertion). It was only required on the
  colocated `transformer_engine` path.
- Cross-node **weight refit**: **use `refit_backend=nccl`** (recommended — see the
  "Refit backend" section below). `nvshmem` also works but requires
  `NVSHMEM_MAX_TEAMS` tuning *and* degrades badly step-over-step on this model
  (~95s → ~135s+ per refit). `gloo` is a slow TCP fallback.

```bash
source /Users/shanmugamr@nvidia.com/tokens  # exports WANDB_API_KEY
export COG_EXTRA_MOUNTS='/lustre/fsw/portfolios/llmservice/users/ksanthanam:/lustre/fsw/portfolios/llmservice/users/ksanthanam,/lustre/fsw/portfolios/llmservice/users/dmosallanezh:/lustre/fsw/portfolios/llmservice/users/dmosallanezh'
MODEL_DIR=/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3.5-nano-swe-step25/without_mtp
HF_DIR=/lustre/fsw/portfolios/llmservice/users/dmosallanezh/nemo-evaluator-rundirs/nano_v35/conversions/geshen_ultra_rl_v5_kd600_step30_fixedpath_20260520_1130/hf

cog submit --repo ~/RL --cluster-name oci-hsg --run-name nano35-mcore-noncolo \
  --gpus 4 --nodes 2 --ntasks-per-node 1 --partition batch --time 02:00:00 --job-name nano35-noncolo \
  --launcher ray \
  --setup-command 'cp -rf examples/. /opt/nemo-rl/examples/ 2>/dev/null || true; cp -rf nemo_rl/. /opt/nemo-rl/nemo_rl/ 2>/dev/null || true' \
  --command "export PYTHONPATH=/opt/nemo-rl; export WANDB_API_KEY=$WANDB_API_KEY; cd /opt/nemo-rl && uv run --no-sync python examples/run_grpo.py \
    --config examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation.yaml \
    policy.model_name=$HF_DIR policy.tokenizer.name=$HF_DIR \
    ++policy.pretrained_checkpoint.path=$MODEL_DIR ++policy.pretrained_checkpoint.format=megatron_lm \
    policy.generation.backend=megatron policy.max_total_sequence_length=2048 policy.train_global_batch_size=8 policy.train_micro_batch_size=1 \
    ++loss_fn.reference_policy_kl_penalty=0 \
    policy.generation.colocated.enabled=false \
    ++policy.generation.colocated.resources.gpus_per_node=4 \
    ++policy.generation.colocated.resources.num_nodes=1 \
    ++policy.generation.mcore_generation_config.transformer_impl=inference_optimized \
    ++policy.generation.mcore_generation_config.refit_backend=nccl \
    ++policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    ++policy.generation.mcore_generation_config.expert_model_parallel_size=4 \
    ++policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
    ++policy.generation.mcore_generation_config.pipeline_model_parallel_size=1 \
    ++policy.generation.mcore_generation_config.context_parallel_size=1 \
    ++policy.generation.mcore_generation_config.sequence_parallel=false \
    ++policy.generation.mcore_generation_config.buffer_size_gb=20 \
    ++policy.generation.mcore_generation_config.num_cuda_graphs=-1 \
    ++policy.generation.mcore_generation_config.cuda_graph_impl=local \
    ++policy.generation.mcore_generation_config.inference_cuda_graph_scope=block \
    ++policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=float32 \
    policy.megatron_cfg.tensor_model_parallel_size=1 policy.megatron_cfg.expert_model_parallel_size=4 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=false policy.megatron_cfg.activation_checkpointing=true \
    grpo.max_num_steps=3 grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 grpo.val_at_start=false grpo.val_period=0 \
    cluster.gpus_per_node=4 cluster.num_nodes=2 checkpointing.enabled=false \
    logger.wandb_enabled=true logger.wandb.project=nano35-mcore-noncolo logger.wandb.name=nano35-noncolo-infopt-nccl logger.tensorboard_enabled=false logger.monitor_gpus=true"
```

## Async non-colocated + MXFP8 training (validated)

Layer **async GRPO** + **MXFP8 training** on top of the non-colocated command
above. This runs generation on the dedicated gen node while training overlaps on
the train node (1-off async), and quantizes the training GEMMs to MXFP8. Validated
end-to-end on GB200 (2 nodes, 10 async steps, `Async GRPO training complete!`,
driver rc=0). Run: `https://wandb.ai/shanmugamr/nano35-mcore-async-mxfp8/runs/8mfdn8n7`.

Additions vs. the non-colocated command:
- **Async GRPO**: `grpo.async_grpo.enabled=true`
  `++grpo.async_grpo.max_trajectory_age_steps=1`
  `++grpo.async_grpo.in_flight_weight_updates=true`. Async requires the
  generation engine to be async — the base recipe already sets
  `mcore_generation_config.async_engine=true` (keep it). In-flight weight updates
  work with the `nccl` megatron refit; the AsyncTrajectoryCollector advances the
  weight version each step and resumes generation right after refit.
- **Importance sampling correction** is *required* for async and must be enabled
  explicitly: `++loss_fn.use_importance_sampling_correction=true` (async draws
  slightly off-policy trajectories from the replay buffer). Without it the async
  loss is biased.
- **MXFP8**: `policy.megatron_cfg.fp8_cfg.enabled=true`
  `policy.megatron_cfg.fp8_cfg.fp8=e4m3`
  `policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8`
  `policy.megatron_cfg.fp8_cfg.fp8_param=false`. `fp8_cfg` is predefined in the
  base config (grpo_math_1B.yaml) so plain overrides work (no `++`). Keep
  `fp8_param=false` — FP8 params can NaN the `token_mult_prob_error` (see the
  Fp8Config comment). Only the **training** model is MXFP8; generation stays
  `transformer_impl=inference_optimized` (bf16), and refit copies bf16 params, so
  MXFP8 does not complicate the cross-node weight transfer.
- **`policy.make_sequence_length_divisible_by=32`** for the MXFP8 32-element
  block: MXFP8 GEMMs need the token dim divisible by 32 (blockwise FP8 needs 128,
  other FP8 needs 16 — see `nemo_rl/models/megatron/data.py`). With TP=1/CP=1/SP
  off the individual-seq `minimum_pad_factor` is 1, so 32 is safe.

```bash
source /Users/shanmugamr@nvidia.com/tokens  # exports WANDB_API_KEY
export COG_EXTRA_MOUNTS='/lustre/fsw/portfolios/llmservice/users/ksanthanam:/lustre/fsw/portfolios/llmservice/users/ksanthanam,/lustre/fsw/portfolios/llmservice/users/dmosallanezh:/lustre/fsw/portfolios/llmservice/users/dmosallanezh'
MODEL_DIR=/lustre/fsw/portfolios/llmservice/users/ksanthanam/nemotron-3.5-nano-swe-step25/without_mtp
HF_DIR=/lustre/fsw/portfolios/llmservice/users/dmosallanezh/nemo-evaluator-rundirs/nano_v35/conversions/geshen_ultra_rl_v5_kd600_step30_fixedpath_20260520_1130/hf

cog submit --repo ~/RL --cluster-name oci-hsg --run-name nano35-async-mxfp8 \
  --gpus 4 --nodes 2 --ntasks-per-node 1 --partition batch --time 04:00:00 --job-name nano35-async-mxfp8 \
  --launcher ray \
  --setup-command 'cp -rf examples/. /opt/nemo-rl/examples/ 2>/dev/null || true; cp -rf nemo_rl/. /opt/nemo-rl/nemo_rl/ 2>/dev/null || true' \
  --command "export PYTHONPATH=/opt/nemo-rl; export WANDB_API_KEY=$WANDB_API_KEY; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd /opt/nemo-rl && uv run --no-sync python examples/run_grpo.py \
    --config examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation.yaml \
    policy.model_name=$HF_DIR policy.tokenizer.name=$HF_DIR \
    ++policy.pretrained_checkpoint.path=$MODEL_DIR ++policy.pretrained_checkpoint.format=megatron_lm \
    policy.generation.backend=megatron policy.max_total_sequence_length=2048 policy.train_global_batch_size=8 policy.train_micro_batch_size=1 \
    policy.make_sequence_length_divisible_by=32 \
    ++loss_fn.reference_policy_kl_penalty=0 \
    ++loss_fn.use_importance_sampling_correction=true \
    grpo.async_grpo.enabled=true ++grpo.async_grpo.max_trajectory_age_steps=1 ++grpo.async_grpo.in_flight_weight_updates=true \
    policy.megatron_cfg.fp8_cfg.enabled=true policy.megatron_cfg.fp8_cfg.fp8=e4m3 policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8 policy.megatron_cfg.fp8_cfg.fp8_param=false \
    policy.generation.colocated.enabled=false \
    ++policy.generation.colocated.resources.gpus_per_node=4 \
    ++policy.generation.colocated.resources.num_nodes=1 \
    ++policy.generation.mcore_generation_config.transformer_impl=inference_optimized \
    ++policy.generation.mcore_generation_config.refit_backend=nccl \
    ++policy.generation.mcore_generation_config.async_engine=true \
    ++policy.generation.mcore_generation_config.tensor_model_parallel_size=1 \
    ++policy.generation.mcore_generation_config.expert_model_parallel_size=4 \
    ++policy.generation.mcore_generation_config.expert_tensor_parallel_size=1 \
    ++policy.generation.mcore_generation_config.pipeline_model_parallel_size=1 \
    ++policy.generation.mcore_generation_config.context_parallel_size=1 \
    ++policy.generation.mcore_generation_config.sequence_parallel=false \
    ++policy.generation.mcore_generation_config.buffer_size_gb=20 \
    ++policy.generation.mcore_generation_config.num_cuda_graphs=-1 \
    ++policy.generation.mcore_generation_config.cuda_graph_impl=local \
    ++policy.generation.mcore_generation_config.inference_cuda_graph_scope=block \
    ++policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=float32 \
    policy.megatron_cfg.tensor_model_parallel_size=1 policy.megatron_cfg.expert_model_parallel_size=4 \
    policy.megatron_cfg.pipeline_model_parallel_size=1 policy.megatron_cfg.context_parallel_size=1 \
    policy.megatron_cfg.sequence_parallel=false policy.megatron_cfg.activation_checkpointing=true \
    grpo.max_num_steps=10 grpo.num_prompts_per_step=2 grpo.num_generations_per_prompt=4 grpo.val_at_start=false grpo.val_period=0 \
    cluster.gpus_per_node=4 cluster.num_nodes=2 checkpointing.enabled=false \
    logger.wandb_enabled=true logger.wandb.project=nano35-mcore-async-mxfp8 logger.wandb.name=nano35-async-noncolo-mxfp8 logger.tensorboard_enabled=false logger.monitor_gpus=true"
```

Notes:
- As with the sync smoke, the tiny batch (2 prompts × 4 gens) means most steps
  have no intra-group reward variance, so most losses read `0.0000` (a couple
  were nonzero, ~0.09). This is expected — raise `num_prompts_per_step` /
  `num_generations_per_prompt` for informative curves (see "Getting a meaningful
  loss curve"). The point of this run is to validate that async + non-colocated +
  MXFP8 compose and run clean, which they do.
- The `cog submit` for this used a blocking foreground `srun`, so the terminal
  stays attached for the whole job (no separate job id to poll); the run dir is
  `runs/nano35-async-mxfp8/slurm/<srun_jobid>.{out,err}`.

## NVLS inference MoE dispatcher (async non-colocated)

The generation model's MoE token dispatcher can use **NVLS** (NVLink SHARP
all-gather via symmetric memory) instead of the default `nccl` inference
dispatcher. It is selected with `inference_moe_token_dispatcher_type` (options:
`nvls` — requires Hopper+/NVLink — and `nccl` — the portable fallback). This key
is read in `nemo_rl/models/megatron/setup.py` from `megatron_cfg`; for
non-colocated generation the `mcore_generation_config` block is merged into the
gen model's `megatron_cfg` (`MegatronGeneration.__init__`), so set it in **both**
places to be safe:

```bash
    ++policy.generation.mcore_generation_config.inference_moe_token_dispatcher_type=nvls \
    ++policy.megatron_cfg.inference_moe_token_dispatcher_type=nvls \
```

Validated end-to-end on GB200 (2 nodes, async non-colocated, EP=4, MXFP8 training,
10 steps). Run: `https://wandb.ai/shanmugamr/nano35-mcore-async-noncolo/runs/puxastbg`
(W&B name `nvls`). It is the exact async non-colocated + MXFP8 command above with
**only** the dispatcher flipped `nccl`→`nvls`.

What to know:
- **Requires the topology/rank-ordering fixes from PR
  [#2902](https://github.com/NVIDIA-NeMo/RL/pull/2902)** (megatron branches in
  `grpo.py setup()`: compute `gpus_per_instance = TP*PP*CP` for the megatron
  backend and call `MegatronGeneration.init_cluster_placement_groups` for the
  non-colocated inference cluster instead of `VllmGeneration`). NVLS needs the EP
  ranks ordered within the NVLink domain. Apply #2902 to the local tree (it is
  overlaid onto `/opt/nemo-rl` by the `--setup-command` cp). For a **single**
  inference node (TP·PP·CP ≤ gpus_per_node → `use_unified_pg=False`) all 4 GPUs are
  on the GB200 NVL72 NVLink fabric anyway, so NVLS works; #2902's explicit
  ordering matters once an inference instance spans multiple nodes.
- **NVLS actually engages** when: the resolved config shows
  `inference_moe_token_dispatcher_type: 'nvls'`, Megatron runs
  `megatron/core/inference/symmetric_memory.py` (`enable_symm_mem_for_group` /
  `NVLSAllGatherVDispatcher.allocate_buffers` for EP>1), and the CUDA-graph warmup
  **captures prefill graphs** (`N P + M D`). The `nccl` dispatcher (and the
  `transformer_engine` training path) force-disable non-decode CUDA graphs
  (`dynamic_context.py`: `force_disable_non_decode_cuda_graphs`), so seeing prefill
  (`P`) graphs in warmup is a quick confirmation you're on the nvls path.
- NVLS (the token dispatcher) is independent of `refit_backend` (the weight-copy
  service) and of MXFP8 (training precision). Keep `refit_backend=nccl`.

## Refit backend: use `nccl` (not `nvshmem`) for non-colocated

`refit_backend` selects the copy service that streams training weights to the
generation engine each step. Megatron-Core ships three
(`megatron/core/resharding/copy_services/`): `gloo`, `nccl`, `nvshmem`. NeMo-RL
wires all three in
`nemo_rl/models/generation/megatron/megatron_worker.py::init_collective_mcore_generation`.
The `nccl` path registers an NCCL/CUDA backend on the cross-world refit
ProcessGroup (in addition to the GLOO/CPU backend used for the object
collectives in `prepare_swap_model_weights`) so `NCCLCopyService` can move weight
bytes with CUDA-tensor `batch_isend_irecv`.

**Measured on this model (30B-A3B nano-3.5, non-colocated 1+1 node, 10 GRPO
steps, `transformer_impl=inference_optimized`, `num_cuda_graphs=-1`):**

| refit backend | per-step refit (steps 2→10)     | trend             | step time (warm) |
| ------------- | ------------------------------- | ----------------- | ---------------- |
| `nccl`        | **~16–19s** (stable)            | flat              | **~37–40s**      |
| `nvshmem`     | ~95s → 100 → 133 → … → **135s+**| **grows steadily**| ~120–170s        |

So `nccl` is both faster *and* stable, while `nvshmem` refit is slow and
**degrades step-over-step** on this MoE (the "increasing refit" symptom). `nccl`
also needs **no** `NVSHMEM_MAX_TEAMS` tuning. Runs:
`nano35-mcore-nccl10` (wandb `70mgrdps`) vs `nano35-mcore-noncolo10` (nvshmem,
wandb `xenat5a8`).

### `nvshmem` fallback and its `NVSHMEM_MAX_TEAMS` gotcha

If you must use `nvshmem` (`++policy.generation.mcore_generation_config.refit_backend=nvshmem`),
the nano-3.5 MoE
cross-node refit through the `NVSHMEMCopyService` allocates many NVSHMEM teams and
overflows the default limit of **128**, killing the generation workers right after
`NVSHMEM v3.6.5` prints, during `refit_policy_generation` (grpo.py):

```
non-zero status: 2 No more teams available (max = 128), try setting NVSHMEM_MAX_TEAMS environment variable
... Unable to allocate enough duplicate teams ... Please increase NVSHMEM_MAX_TEAMS. Exiting
```

Fix: `export NVSHMEM_MAX_TEAMS=512` **in `--command`** (the driver). NeMo-RL
forwards the driver's `os.environ` into every worker's Ray `runtime_env`
(`worker_groups.py` `_create_workers_...`: `for k,v in os.environ.items(): ...`),
so a driver-side export reaches the generation workers. Do **not** rely on
`--setup-command` for this — cog runs the setup block in a subshell
(`bash /tmp/cog_ray_setup.sh`) so its exports never reach `ray start` / the
raylet / the workers.

`refit_backend=gloo` is a working alternative that sidesteps NVSHMEM teams
entirely (TCP-based copy service), if you'd rather not tune `NVSHMEM_MAX_TEAMS`.

### `nvshmem` refit works with **async** in-flight updates (validated)

`refit_backend=nvshmem` also composes cleanly with **async** non-colocated GRPO
(`grpo.async_grpo.enabled=true`, `in_flight_weight_updates=true`) — the
`AsyncTrajectoryCollector` advances the weight version and the `NVSHMEMCopyService`
streams weights to the gen node each step (you'll see rank-0 log
`Starting schedule: N send requests ... Packed: 128 batches across 4 destination PEs`).
Validated end-to-end on GB200 (2 nodes, async non-colocated, EP=4, MXFP8 training,
10 steps, `Async GRPO training complete!`, driver rc=0). Run:
`https://wandb.ai/shanmugamr/nano35-mcore-async-noncolo/runs/ibj64ric` (W&B name
`nvshmem_refit_with_env`). It is the async non-colocated + MXFP8 command with
`refit_backend=nccl`→`nvshmem` and these two env exports added **in `--command`**:

```bash
export NVSHMEM_MAX_TEAMS=512;   # required — avoids the 128-team overflow crash above
export NVSHMEM_MAX_CTAS=2;      # caps the CTAs each NVSHMEM copy kernel launches
```

`NVSHMEM_MAX_CTAS` bounds the number of CTAs (thread blocks) an NVSHMEM copy
kernel uses; like `NVSHMEM_MAX_TEAMS` it must be a **driver-side** export so
NeMo-RL forwards it into the worker `runtime_env`. Note the perf caveat above
still holds — nvshmem refit is slower and degrades step-over-step vs. `nccl`; use
`nvshmem` only when specifically testing that backend.

## Optimized mcore inference settings (match the tuned standalone config)

To make the megatron generation path match the tuned standalone mcore inference
config (fastest generation for nano-3.5), append the overrides below. Split by
where they must live, because **colocated** generation reuses the training model:

**Model/arch knobs — set on `policy.megatron_cfg.*`** (these drive the colocated
generation model; the `mcore_generation_config` copies only apply non-colocated):

```bash
    policy.megatron_cfg.moe_token_dispatcher_type=alltoall \
    policy.megatron_cfg.inference_moe_token_dispatcher_type=alltoall \
    policy.megatron_cfg.moe_grouped_gemm=true \
    policy.megatron_cfg.inference_grouped_gemm_backend=vllm \
    policy.megatron_cfg.moe_router_dtype=fp32 \
    policy.megatron_cfg.moe_shared_expert_overlap=true \
```

**Inference-engine knobs — set on `++policy.generation.mcore_generation_config.*`**
(read directly by the `DynamicInferenceEngine`, so they apply in either mode):

```bash
    ++policy.generation.mcore_generation_config.num_cuda_graphs=-1 \
    ++policy.generation.mcore_generation_config.cuda_graph_impl=local \
    ++policy.generation.mcore_generation_config.inference_cuda_graph_scope=block \
    ++policy.generation.mcore_generation_config.mamba_inference_ssm_states_dtype=float32 \
    ++policy.generation.mcore_generation_config.buffer_size_gb=20 \
```

Notes / caveats:
- **`num_cuda_graphs=-1`** is the single biggest generation-speed knob vs. the
  base recipe default of `4`: it captures the full set of decode graph sizes
  (matching `--inference-dynamic-batching-num-cuda-graphs -1`) so decode batches
  are padded much less. There is no NeMo-RL key for
  `--inference-dynamic-batching-cuda-graph-sizing-distribution linear`; the engine
  always uses Megatron's default `EXPONENTIAL` distribution.
- **`inference_cuda_graph_scope=block`** is the correct value for the standalone
  `--cuda-graph-scope full_iteration_inference`, which Megatron migrates to
  `block` internally.
- **colocated vs `inference_optimized`:** the standalone config uses
  `--transformer-impl inference_optimized`. That impl is **not reachable in a
  colocated run** — colocated generation reuses the training model, and training
  workers are forbidden from `inference_optimized`
  (`megatron_policy_worker.py`), so the model runs as `transformer_engine`. On the
  `transformer_engine` + `cuda_graph_impl=local` path you MUST keep
  `moe_pad_experts_for_cuda_graph_inference=true` (on both `megatron_cfg` and
  `mcore_generation_config`). To actually use `transformer_impl=inference_optimized`
  you must run **non-colocated** generation (`policy.generation.colocated.enabled=false`
  with dedicated generation GPUs); then set
  `++policy.generation.mcore_generation_config.transformer_impl=inference_optimized`
  and **drop** `moe_pad_experts_for_cuda_graph_inference` (with `inference_optimized`
  that padding flag trips an assertion — see gotchas).

## Weights & Biases logging (and returning the dashboard link)

Always enable W&B for multi-step runs so results are browsable, and **return the
dashboard URL to the user**. Add these to the command:

```bash
# in the local shell before cog submit:
source /Users/shanmugamr@nvidia.com/tokens
# inside --command, forward the sourced variable before the python call:
export WANDB_API_KEY=$WANDB_API_KEY;
# ... and pass these overrides to run_grpo.py:
    logger.wandb_enabled=true \
    logger.wandb.project=nano35-mcore-megatron \
    logger.wandb.name=nano35-mcore-<N>step \
    logger.monitor_gpus=true
```

W&B prints the run URL to stdout/stderr on startup. After the job is running,
extract and return it:

```bash
RUN=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/runs/<run-name>
ssh oci "grep -rhoE 'https://wandb.ai/\S+' $RUN/slurm/*.out $RUN/slurm/*.err | head -1"
# -> e.g. https://wandb.ai/<entity>/nano35-mcore-megatron/runs/<id>
```

Give that `https://wandb.ai/<entity>/<project>/runs/<id>` link back to the user.
For more than one step, drop the smoke's step count and set
`grpo.max_num_steps=<N>` (e.g. 10). The CUDA-graph warmup cost is paid only on
the first generation; later steps are much faster.

### Getting a *meaningful* (nonzero) loss curve

GRPO loss can read `0.0000` when there is no **intra-group reward variance** —
i.e. with a tiny smoke batch (2 prompts × 4 gens) every rollout in a prompt
group gets the same reward, so the group-relative advantage is 0 and the policy
gradient vanishes. This is expected for the smoke, not a bug. To see a real
loss/reward trend, raise the per-step batch so prompt groups contain a mix of
correct/incorrect rollouts:

- `grpo.num_prompts_per_step=16 grpo.num_generations_per_prompt=8` (128
  rollouts/step) with `policy.train_global_batch_size=128` produces nonzero,
  informative curves on a single 4×GB200 node (EP=4, `reference_policy_kl_penalty=0`,
  `activation_checkpointing=true`).
- Keep ≥8 generations per prompt — that's what creates the reward variance.
- **Memory footprint for the bigger batch (single 4×GB200 node):** this ~34B
  recipe squeezed onto 4 GPUs is very tight. `max_total_sequence_length=4096`
  with the bigger batch **OOMs in the training backward** (MoE grad alloc; the
  first policy step dies with `torch.OutOfMemoryError`, generation itself is
  fine). Use the proven footprint: `max_total_sequence_length=2048`,
  `policy.train_micro_batch_size=1`, `buffer_size_gb=10`, and export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (the failing step showed
  ~11 GB reserved-but-unallocated fragmentation). Backward memory is set by
  `train_micro_batch_size` × seq len, not by the total rollout count (rollouts
  are microbatched), so a big batch is fine as long as seq len stays at 2048.
  Raise seq len only with more GPUs.

### Time limit (oci-hsg `batch` partition)

`batch` MaxTime is **04:00:00** (default 31 min). `--time 06:00:00` fails at
submit with `srun: error: ... Requested time limit is invalid (missing or
exceeds some limit)`. Use `--time 04:00:00` for a 10-step bigger-batch run, or
the `batch_long` partition (7-day MaxTime) for longer jobs.

### Multi-node caveat

`cog` multi-node only populates torch-dist env vars (`MASTER_ADDR`, `WORLD_SIZE`,
etc.); it does **not** bootstrap a Ray head/worker cluster. NeMo-RL orchestrates
workers via Ray (see the repo's `ray.sub` sbatch flow), so a real 2+ node run
needs that Ray bootstrap, not a plain `cog submit --nodes N`. For demonstrations,
prefer a single-node bigger batch (above) over hand-rolling Ray under cog.

## Running with the vLLM generation backend (instead of megatron)

The same nano-3.5 GRPO run can use **vLLM** for generation instead of the
megatron `DynamicInferenceEngine`. Training still runs on megatron
(`policy.megatron_cfg.*`); only the rollout/generation engine changes. vLLM
instantiates the model from the HF architecture at `policy.model_name`
(`NemotronHForCausalLM`, `model_type=nemotron_h`, 128 experts — this **is**
supported by the image's vLLM) and gets its weights refit from the megatron
policy each step. Use it to cross-check that mcore and vLLM generation give
consistent GRPO results.

Change only these overrides vs. the bigger-batch megatron command above:

```bash
# swap the backend
policy.generation.backend=vllm
# vLLM shards the ~34B model across the 4 GPUs with tensor parallel
# (the megatron path used expert_model_parallel_size=4 instead)
policy.generation.vllm_cfg.tensor_parallel_size=4
policy.generation.vllm_cfg.max_model_len=2048
policy.generation.vllm_cfg.gpu_memory_utilization=0.6
# drop ALL the ++policy.generation.mcore_generation_config.* overrides — they are
# ignored by the vLLM backend.
```

Keep everything else identical (`policy.megatron_cfg.*` training parallelism =
EP4/TP1, seq 2048, `train_micro_batch_size=1`, `train_global_batch_size=128`,
`reference_policy_kl_penalty=0`, `grpo.num_prompts_per_step=16`,
`num_generations_per_prompt=8`, colocated enabled, W&B).

**CRITICAL — do NOT set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for a
vLLM run.** vLLM's colocated allocator uses a CUDA memory pool
(`device_allocator/cumem.py`) that is incompatible with expandable segments; the
worker dies at model load with `AssertionError: Expandable segments are not
compatible with memory pool`. Expandable segments were only needed to defrag the
*megatron-generation* path — vLLM manages its own KV cache via
`gpu_memory_utilization`, so just omit the env var.

Verified on GB200 (4×): mcore and vLLM produce closely-matching reward curves
(within sampling noise) over 10 GRPO steps on OpenMathInstruct-2. One expected
systematic difference: vLLM's **generation KL error is ~10× lower** (~0.0015 vs
~0.013) because its post-refit sampler numerics align more tightly with the
megatron training-model logprobs than the mcore CUDA-graph/grouped-GEMM path.

### vLLM **non-colocated** (head-to-head vs mcore non-colocated)

vLLM also runs in the non-colocated 2-node layout — just take the mcore
non-colocated command and swap the backend block (as above): set
`policy.generation.backend=vllm`, `vllm_cfg.tensor_parallel_size=4`,
`vllm_cfg.max_model_len=2048`, `vllm_cfg.gpu_memory_utilization=0.8` (non-colocated
owns the whole gen node, so it can use more), and **drop** all the
`mcore_generation_config.*` overrides *and* the `NVSHMEM_MAX_TEAMS` export (vLLM
uses its own collective/IPC weight-update path — `VllmInternalWorkerExtension` —
not the megatron `NVSHMEMCopyService`, so no team-limit issue). Keep training
`megatron_cfg.*` EP4/TP1, same GRPO settings.

Validated runs (2 nodes, 1 gen + 1 train, 3 GRPO steps, seq 2048, 8 rollouts/step,
OpenMathInstruct-2):
- mcore (`inference_optimized`, nvshmem): `runs/6ynwwxwy`, wall clock **20.5 min**
- vLLM (TP=4): `runs/ox6jispo`, wall clock **10.3 min**

Per-step breakdown (from the driver `timing_metrics`, steady-state step 3):

| phase | mcore non-colo | vLLM non-colo |
|---|---|---|
| total step time (step 2 / step 3) | 144.3s / 132.9s | 62.1s / 36.8s |
| **weight refit** (`transfer_and_update_weights`, step 3) | **107.9s (81%)** | **22.7s (62%)** |
| generation/decode (step 3) | 10.4s | 10.8s |
| step 1 (one-time warmup) | 495s (mcore CUDA-graph warmup) | 85s |
| generation KL error | 0.0010–0.0013 | 0.0011–0.0019 |

Key takeaways:
- **The dominant non-colocated cost is the cross-node weight refit, not
  generation.** Pure decode is ~equal (~10s). vLLM's collective/IPC refit
  (~23–46s) is ~3–5× faster than mcore's nvshmem refit into the inference_optimized
  engine (~108s), which is why vLLM wins ~2× on wall clock and ~2.8× per
  steady-state step.
- mcore pays a large one-time CUDA-graph warmup (step 1 ≈ 495s with
  `num_cuda_graphs=-1`); amortized over a long run this matters less.
- **Generation quality is comparable here** (both KL error ~0.001–0.002). Note this
  is *better* mcore alignment than the colocated `transformer_engine` path
  (~0.013 above) — running mcore with `transformer_impl=inference_optimized`
  non-colocated closes most of the KL-error gap vs vLLM.

## Watch / debug

```bash
ssh oci 'squeue -u $USER -o "%.10i %.16j %.8T %.10M %R"'
cog logs slurm --run-name nano35-mcore-math --job-id <JOBID> --stream both
ssh oci 'tail -f /lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/runs/nano35-mcore-math/slurm/<JOBID>.err'
```

## Iterating

For many quick iterations use a persistent allocation instead of re-queuing.
`session start` does **not** honor `COG_EXTRA_MOUNTS`, so register the
llmservice path with the cluster (or start the session from a scratch-local copy
of the checkpoint) if you go the session route:

```bash
cog session start --repo ~/RL --session-handle nano35 --gpus 4 --time 04:00:00 --partition batch
cog session exec --session-handle nano35 --repo ~/RL \
  --command 'cp -rf examples/. /opt/nemo-rl/examples/; cd /opt/nemo-rl && uv run --no-sync python examples/nemo_gym/run_grpo_nemo_gym.py --config examples/nemo_gym/grpo_nanov3.yaml ...same overrides...' \
  --wait-timeout 3600
cog session stop --session-handle nano35
```

## Gotchas checklist

- `FileNotFoundError: /path/to/train.jsonl` → grpo_nanov3.yaml ships placeholder
  data paths and the Gym data is NOT committed to the image; run `ng_prepare_data`
  first (source `/Users/shanmugamr@nvidia.com/tokens` to obtain `HF_TOKEN`) or
  point `data.train/validation.data_path` at a real prepared nemo-gym dataset
  (see step 3).
- `QOSMinGRES` → request `--gpus 4` (whole node).
- `pretrained_checkpoint.path=... does not contain metadata.json` → point at the
  torch_dist iter dir (or root with `latest_checkpointed_iteration.txt`); the
  `without_mtp` dir must contain `metadata.json` (or `iter_*` subdirs).
- Checkpoint/HF-config architecture mismatch (unexpected/missing weights, shape
  errors) → the HF `config.json` at `policy.model_name` must describe the
  nano-3.5 architecture (hidden 2688, 128 experts, mamba dims, hybrid pattern,
  squared-relu, sigmoid router, topk 6, etc.). Patch specific fields with
  `++policy.hf_config_overrides.<field>=<value>` rather than editing the ckpt.
- `No such file or directory` for the model/HF dir inside the container →
  `COG_EXTRA_MOUNTS` not exported before `cog submit`, or a `session` (which
  ignores it) is being used.
- `expert_model_parallel_size` not dividing `num_experts` (128), or the product
  of parallel sizes ≠ allocated GPUs → adjust EP / GPU count together.
- `nemo-gym references a workspace ... but is not a workspace member` → you ran
  from the synced workspace; run from `/opt/nemo-rl` (full submodules + venv).
- `AssertionError: --moe-pad-experts-for-cuda-graph-inference must be set when
  using CUDA graphs with expert parallelism` → this model uses EP + local CUDA
  graphs; add
  `++policy.generation.mcore_generation_config.moe_pad_experts_for_cuda_graph_inference=true`
  (and the same under `policy.megatron_cfg`). **Required for nano-3.5** on the
  colocated (`transformer_engine`) path.
- `AssertionError: moe_pad_experts_for_cuda_graph_inference cannot be True when
  transformer_impl is 'inference_optimized'` (text_generation_controller.py) →
  you enabled `transformer_impl=inference_optimized` (only possible non-colocated)
  **and** left `moe_pad_experts_for_cuda_graph_inference=true`. With
  `inference_optimized` the expert padding is unnecessary and forbidden — drop the
  `moe_pad_experts_for_cuda_graph_inference` overrides. (Conversely, on the
  colocated `transformer_engine` path the flag is *required*, per the assertion
  above — the two are mutually exclusive.)
- `No more teams available (max = 128), try setting NVSHMEM_MAX_TEAMS` /
  `Unable to allocate enough duplicate teams` (generation worker dies during
  `refit_policy_generation`, right after `NVSHMEM v3.6.5`, on a **non-colocated**
  run) → the cross-node NVSHMEM weight-refit overflows the default 128-team pool.
  Best fix: switch to `++policy.generation.mcore_generation_config.refit_backend=nccl`
  (faster, stable, and needs no NVSHMEM tuning — see the "Refit backend" section).
  If you must stay on nvshmem: `export NVSHMEM_MAX_TEAMS=512` in the driver
  `--command` (NeMo-RL forwards the driver env to workers). Must be in `--command`,
  not `--setup-command` (cog runs the latter in a subshell that never reaches
  `ray start`). `refit_backend=gloo` also works (slow TCP fallback).
- **nvshmem refit slow and growing step-over-step** (`transfer_and_update_weights`
  climbs ~95s → 135s+ across steps and dominates step time, non-colocated) → known
  behavior of the `NVSHMEMCopyService` on this MoE. Fix:
  `++policy.generation.mcore_generation_config.refit_backend=nccl` (stays flat at
  ~16–19s). See the "Refit backend" section.
- `AssertionError: Expandable segments are not compatible with memory pool`
  (vLLM worker dies at model load) → you set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on a **vLLM** run. That env
  var is only for the megatron-generation path; remove it for vLLM (see the vLLM
  backend section).
- `ModuleNotFoundError: No module named 'megatron'` during Ray object
  deserialization (job hangs right after "Generating responses...") → the driver
  imported `nemo_rl` from cog's synced workspace (whose `3rdparty` submodules are
  empty) so it can't import `megatron` to unpickle worker results. Fix: overlay
  the synced `nemo_rl` onto the image and force the image path on the driver —
  prepend `export PYTHONPATH=/opt/nemo-rl; ... cp -rf nemo_rl/. /opt/nemo-rl/nemo_rl/;`
  to `--command` (run entirely from `/opt/nemo-rl`).
- `expert_tensor_parallel_size` / `pipeline_model_parallel_size` /
  `context_parallel_size` / `buffer_size_gb` `... is not in struct` → the recipe's
  `mcore_generation_config` doesn't predefine these; use `++` to append them.
- With `tensor_model_parallel_size=1` you must set `sequence_parallel=false`
  (sequence parallel requires TP > 1) on both `megatron_cfg` and
  `mcore_generation_config`.
- `ng_prepare_data: No such file or directory` / `No module named nemo_gym` →
  the nightly image's `/opt/nemo_rl_venv` does NOT include the `nemo_gym` extra.
  Gym runs (`grpo_nanov3.yaml`) need `nemo_gym` installed plus prepared data.
- OOM during **generation** / KV-cache too large → lower
  `mcore_generation_config.buffer_size_gb`, `max_total_sequence_length`, or
  `max_tokens`.
- `torch.OutOfMemoryError` during the **training forward** (MoE experts) on a
  4-GPU colocated smoke → nano-3.5 (~34B) + reference model (~36 GB) + a large
  persisted KV buffer leaves little room for training activations. Mitigations
  (any/all): `++loss_fn.reference_policy_kl_penalty=0` (skips loading the
  reference model, frees ~36 GB), lower
  `++policy.generation.mcore_generation_config.buffer_size_gb` (e.g. 10),
  `policy.megatron_cfg.activation_checkpointing=true`, and reduce
  `policy.max_total_sequence_length` (e.g. 2048). For real runs, use more
  GPUs/nodes instead of squeezing onto 4.
