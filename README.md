# Nemo RL: A Scalable and Efficient Post-Training Library

<!-- markdown all in one -->
- [Nemo RL: A Scalable and Efficient Post-Training Library](#nemo-rl-a-scalable-and-efficient-post-training-library)
  - [📣 News](#-news)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Training Backends](#training-backends)
  - [GRPO](#grpo)
    - [GRPO Single Node](#grpo-single-node)
    - [GRPO Multi-node](#grpo-multi-node)
      - [GRPO Qwen2.5-32B](#grpo-qwen25-32b)
      - [GRPO Multi-Turn](#grpo-multi-turn)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
    - [SFT Single Node](#sft-single-node)
    - [SFT Multi-node](#sft-multi-node)
  - [DPO](#dpo)
    - [DPO Single Node](#dpo-single-node)
    - [DPO Multi-node](#dpo-multi-node)
  - [Evaluation](#evaluation)
    - [Convert Model Format (Optional)](#convert-model-format-optional)
    - [Run Evaluation](#run-evaluation)
  - [Set Up Clusters](#set-up-clusters)
  - [Tips and Tricks](#tips-and-tricks)
  - [Citation](#citation)
  - [Contributing](#contributing)
  - [Licenses](#licenses)

**Nemo RL** is a scalable and efficient post-training library designed for models ranging from 1 GPU to thousands, and from tiny to over 100 billion parameters.

What you can expect:

- **Seamless integration with Hugging Face** for ease of use, allowing users to leverage a wide range of pre-trained models and tools.
- **High-performance implementation with Megatron Core**, supporting various parallelism techniques for large models (>100B) and large context lengths.
- **Efficient resource management using Ray**, enabling scalable and flexible deployment across different hardware configurations.
- **Flexibility** with a modular design that allows easy integration and customization.
- **Comprehensive documentation** that is both detailed and user-friendly, with practical examples.

## 📣 News
* [7/25/2025] [Release v0.3.0!](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.3.0)
    * 📊 View the release run metrics on [Google Colab](https://colab.research.google.com/drive/15kpesCV1m_C5UQFStssTEjaN2RsBMeZ0?usp=sharing) to get a head start on your experimentation.
* [5/14/2025] [Reproduce DeepscaleR with NeMo RL!](docs/guides/grpo-deepscaler.md)
* [5/14/2025] [Release v0.2.1!](https://github.com/NVIDIA-NeMo/RL/releases/tag/v0.2.1)
    * 📊 View the release run metrics on [Google Colab](https://colab.research.google.com/drive/1o14sO0gj_Tl_ZXGsoYip3C0r5ofkU1Ey?usp=sharing) to get a head start on your experimentation.

## Features

✅ _Available now_ | 🔜 _Coming in v0.4_

- ✅ **Fast Generation** - vLLM backend for optimized inference.
- ✅ **HuggingFace Integration** - Works with 1-70B models (Qwen, Llama).
- ✅ **Distributed Training** - Fully Sharded Data Parallel (FSDP2) support and Ray-based infrastructure.
- ✅ **Environment Support** - Support for multi-environment training.
- ✅ **Learning Algorithms** - GRPO (Group Relative Policy Optimization), SFT (Supervised Fine-Tuning), and DPO (Direct Preference Optimization).
- ✅ **Multi-Turn RL** - Multi-turn generation and training for RL with tool use, games, etc.
- ✅ **Large Model Support** - Native PyTorch support for models up to 70B parameters.
- ✅ **Advanced Parallelism** - PyTorch native FSDP2, TP, CP, and SP for efficient training.
- ✅ **(even) Larger Model Support with Long(er) Sequences** - Advanced parallelisms with Megatron Core (TP/PP/CP/SP/EP).
- ✅ **Worker Isolation** - Process isolation between RL Actors (no worries about global state).
- ✅ **Environment Isolation** - Dependency isolation between components.
- ✅ **Megatron Inference** - (static) Megatron Inference for day-0 support for new megatron models.
- ✅ **MoE Models** - Support for DeepseekV3 and Qwen-3 MoE models
- ✅ **Sequence Packing** - Sequence packing in both DTensor and MCore for huge training perf gains


- 🔜 **Improved Native Performance** - Improve training time for Native Pytorch Models.
- 🔜 **Megatron Inference** - (dynamic) Megatron Inference for fast day-0 support for new megatron models.

## Prerequisites

Clone **NeMo RL**.
```sh
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl
cd nemo-rl

# If you are using the Megatron backend, download the pinned versions of Megatron-LM and NeMo submodules 
# by running (This is not necessary if you are using the pure Pytorch/DTensor path):
git submodule update --init --recursive

# Different branches of the repo can have different pinned versions of these third-party submodules. Ensure
# submodules are automatically updated after switching branches or pulling updates by configuring git with:
# git config submodule.recurse true

# **NOTE**: this setting will not download **new** or remove **old** submodules with the branch's changes.
# You will have to run the full `git submodule update --init --recursive` command in these situations.
```

If you are using the Megatron backend on bare-metal (outside of a container), you may
need to install the cudnn headers as well. Here is how you can check as well as install them:
```sh
# Check if you have libcudnn installed
dpkg -l | grep cudnn.*cuda

# Find the version you need here: https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network
# As an example, these are the "Linux Ubuntu 20.04 x86_64" instructions
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cudnn-cuda-12
```

Install `uv`.
```sh
# For faster setup and environment isolation, we use `uv`
pip install uv

# Initialize NeMo RL project virtual environment
# NOTE: Please do not use -p/--python and instead allow uv venv to read it from .python-version
#       This ensures that the version of python used is always what we prescribe.
uv venv

# If working outside a container, it can help to build flash-attn and warm the
# uv cache before your first run. The NeMo RL Dockerfile will warm the uv cache
# with flash-attn. See https://docs.nvidia.com/nemo/rl/latest/docker.html for
# instructions if you are looking for the NeMo RL container.
bash tools/build-flash-attn-in-uv-cache.sh
# If sucessful, you should see "✅ flash-attn successfully added to uv cache"

# If you cannot install at the system level, you can install for your user with
# pip install --user uv

# Use `uv run` to launch all commands. It handles pip installing implicitly and
# ensures your environment is up to date with our lock file.

# Note that it is not recommended to activate the venv and instead use `uv run` since
# it ensures consistent environment usage across different shells and sessions.
# Example: uv run python examples/run_grpo_math.py
```

**Important Notes:**

- Use the `uv run <command>` to execute scripts within the managed environment. This helps maintain consistency across different shells and sessions.
- Ensure you have the necessary CUDA drivers and PyTorch installed compatible with your hardware.
- On the first install, `flash-attn` can take a while to install (~45min with 48 CPU hyperthreads). After it is built once, it is cached in your `uv`'s cache dir making subsequent installs much quicker.
- **Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.

## Training Backends

NeMo RL supports multiple training backends to accommodate different model sizes and hardware configurations:

- **DTensor (FSDP2)** - PyTorch's next-generation distributed training with improved memory efficiency
- **Megatron** - NVIDIA's high-performance training framework for scaling to large models (>100B parameters)

The training backend is automatically determined based on your YAML configuration settings. For detailed information on backend selection, configuration, and examples, see the [Training Backends documentation](docs/design-docs/training-backends.md).

## GRPO

We have a reference GRPO experiment config set up trained for math benchmarks using the [OpenInstructMath2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) dataset.

You can read about the details of the GRPO implementation [here](docs/guides/grpo.md)

### GRPO Single Node

To run GRPO on a single GPU for `Qwen/Qwen2.5-1.5B`:

```sh
# Run the GRPO math example using a 1B parameter model
uv run python examples/run_grpo_math.py
```

By default, this uses the configuration in `examples/configs/grpo_math_1B.yaml`. You can customize parameters with command-line overrides. For example, to run on 8 GPUs,

```sh
# Run the GRPO math example using a 1B parameter model using 8 GPUs
uv run python examples/run_grpo_math.py \
  cluster.gpus_per_node=8
```

You can override any of the parameters listed in the yaml configuration file. For example,

```sh
uv run python examples/run_grpo_math.py \
  policy.model_name="meta-llama/Llama-3.2-1B-Instruct" \
  checkpointing.checkpoint_dir="results/llama1b_math" \
  logger.wandb_enabled=True \
  logger.wandb.name="grpo-llama1b_math" \
  logger.num_val_samples_to_print=10
```

The default configuration uses the DTensor training backend. We also provide a config `examples/configs/grpo_math_1B_megatron.yaml` which is set up to use the Megatron backend out of the box.

To train using this config on a single GPU:

```sh
# Run a GRPO math example on 1 GPU using the Megatron backend
uv run python examples/run_grpo_math.py \
  --config examples/configs/grpo_math_1B_megatron.yaml
```

For additional details on supported backends and how to configure the training backend to suit your setup, refer to the [Training Backends documentation](docs/design-docs/training-backends.md).

### GRPO Multi-node

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

# grpo_math_8b uses Llama-3.1-8B-Instruct model
COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml cluster.num_nodes=2 checkpointing.checkpoint_dir='results/llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='grpo-llama8b_math'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```
The required `CONTAINER` can be built by following the instructions in the [Docker documentation](docs/docker.md).

#### GRPO Qwen2.5-32B

This section outlines how to run GRPO for Qwen2.5-32B with a 16k sequence length.
```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=32

# Download Qwen before the job starts to avoid spending time downloading during the training loop
HF_HOME=/path/to/hf_home huggingface-cli download Qwen/Qwen2.5-32B

# Ensure HF_HOME is included in your MOUNTS
HF_HOME=/path/to/hf_home \
COMMAND="uv run ./examples/run_grpo_math.py --config examples/configs/grpo_math_8B.yaml policy.model_name='Qwen/Qwen2.5-32B' policy.generation.vllm_cfg.tensor_parallel_size=4 policy.max_total_sequence_length=16384 cluster.num_nodes=${NUM_ACTOR_NODES} policy.dtensor_cfg.enabled=True policy.dtensor_cfg.tensor_parallel_size=8 policy.dtensor_cfg.sequence_parallel=True policy.dtensor_cfg.activation_checkpointing=True checkpointing.checkpoint_dir='results/qwen2.5-32b' logger.wandb_enabled=True logger.wandb.name='qwen2.5-32b'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

#### GRPO Multi-Turn

We also support multi-turn generation and training (tool use, games, etc.).
Reference example for training to play a Sliding Puzzle Game:
```sh
uv run python examples/run_grpo_sliding_puzzle.py
```

## Supervised Fine-Tuning (SFT)

We provide an example SFT experiment using the [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/).

### SFT Single Node

The default SFT configuration is set to run on a single GPU. To start the experiment:

```sh
uv run python examples/run_sft.py
```

This fine-tunes the `Llama3.2-1B` model on the SQuAD dataset using a 1 GPU.

To use multiple GPUs on a single node, you can modify the cluster configuration. This adjustment will also let you potentially increase the model and batch size:

```sh
uv run python examples/run_sft.py \
  policy.model_name="meta-llama/Meta-Llama-3-8B" \
  policy.train_global_batch_size=128 \
  sft.val_global_batch_size=128 \
  cluster.gpus_per_node=8
```

Refer to `examples/configs/sft.yaml` for a full list of parameters that can be overridden.

### SFT Multi-node

```sh
# Run from the root of NeMo RL repo
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_sft.py --config examples/configs/sft.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 checkpointing.checkpoint_dir='results/sft_llama8b_2nodes' logger.wandb_enabled=True logger.wandb.name='sft-llama8b'" \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

## DPO

We provide a sample DPO experiment that uses the [HelpSteer3 dataset](https://huggingface.co/datasets/nvidia/HelpSteer3) for preference-based training.

### DPO Single Node

The default DPO experiment is configured to run on a single GPU. To launch the experiment:

```sh
uv run python examples/run_dpo.py
```

This trains `Llama3.2-1B-Instruct` on one GPU.

If you have access to more GPUs, you can update the experiment accordingly. To run on 8 GPUs, we update the cluster configuration and switch to an 8B Llama3.1 Instruct model:

```sh
uv run python examples/run_dpo.py \
  policy.model_name="meta-llama/Llama-3.1-8B-Instruct" \
  policy.train_global_batch_size=256 \
  cluster.gpus_per_node=8
```

Any of the DPO parameters can be customized from the command line. For example:

```sh
uv run python examples/run_dpo.py \
  dpo.sft_loss_weight=0.1 \
  dpo.preference_average_log_probs=True \
  checkpointing.checkpoint_dir="results/llama_dpo_sft" \
  logger.wandb_enabled=True \
  logger.wandb.name="llama-dpo-sft"
```

Refer to `examples/configs/dpo.yaml` for a full list of parameters that can be overridden. For an in-depth explanation of how to add your own DPO dataset, refer to the [DPO documentation](docs/guides/dpo.md).

### DPO Multi-node

For distributed DPO training across multiple nodes, modify the following script for your use case:

```sh
# Run from the root of NeMo RL repo
## number of nodes to use for your job
NUM_ACTOR_NODES=2

COMMAND="uv run ./examples/run_dpo.py --config examples/configs/dpo.yaml cluster.num_nodes=2 cluster.gpus_per_node=8 dpo.val_global_batch_size=32 checkpointing.checkpoint_dir='results/dpo_llama81_2nodes' logger.wandb_enabled=True logger.wandb.name='dpo-llama1b'" \
RAY_DEDUP_LOGS=0 \
CONTAINER=YOUR_CONTAINER \
MOUNTS="$PWD:$PWD" \
sbatch \
    --nodes=${NUM_ACTOR_NODES} \
    --account=YOUR_ACCOUNT \
    --job-name=YOUR_JOBNAME \
    --partition=YOUR_PARTITION \
    --time=4:0:0 \
    --gres=gpu:8 \
    ray.sub
```

## Evaluation

We provide evaluation tools to assess model capabilities.

### Convert Model Format (Optional)

If you have trained a model and saved the checkpoint in the Pytorch DCP format, you first need to convert it to the Hugging Face format before running evaluation:

```sh
# Example for a GRPO checkpoint at step 170
uv run python examples/converters/convert_dcp_to_hf.py \
    --config results/grpo/step_170/config.yaml \
    --dcp-ckpt-path results/grpo/step_170/policy/weights/ \
    --hf-ckpt-path results/grpo/hf
```
> **Note:** Adjust the paths according to your training output directory structure.

For an in-depth explanation of checkpointing, refer to the [Checkpointing documentation](docs/design-docs/checkpointing.md).

### Run Evaluation

Run evaluation script with converted model:

```sh
uv run python examples/run_eval.py generation.model_name=$PWD/results/grpo/hf
```

Run evaluation script with custom settings:

```sh
# Example: Evaluation of DeepScaleR-1.5B-Preview on MATH-500 using 8 GPUs
#          Pass@1 accuracy averaged over 16 samples for each problem
uv run python examples/run_eval.py \
    --config examples/configs/evals/math_eval.yaml \
    generation.model_name=agentica-org/DeepScaleR-1.5B-Preview \
    generation.temperature=0.6 \
    generation.top_p=0.95 \
    generation.vllm_cfg.max_model_len=32768 \
    data.dataset_name=math500 \
    eval.num_tests_per_prompt=16 \
    cluster.gpus_per_node=8
```
> **Note:** Evaluation results may vary slightly due to various factors, such as sampling parameters, random seed, inference engine version, and inference engine settings.

Refer to `examples/configs/evals/eval.yaml` for a full list of parameters that can be overridden. For an in-depth explanation of evaluation, refer to the [Evaluation documentation](docs/guides/eval.md).

## Set Up Clusters

For detailed instructions on how to set up and launch NeMo RL on Slurm or Kubernetes clusters, please refer to the dedicated [Cluster Start](docs/cluster.md) documentation.

## Tips and Tricks
- If you forget to initialize the NeMo and Megatron submodules when cloning the NeMo-RL repository, you may run into an error like this:
  
  ```sh
  ModuleNotFoundError: No module named 'megatron'
  ```
  
  If you see this error, there is likely an issue with your virtual environments. To fix this, first intialize the submodules:

  ```sh
  git submodule update --init --recursive
  ```

  and then force a rebuild of the virutal environments by setting `NRL_FORCE_REBUILD_VENVS=true` next time you launch a run:

  ```sh
  NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py ...
  ```

## Citation

If you use NeMo RL in your research, please cite it using the following BibTeX entry:

```bibtex
@misc{nemo-rl,
title = {NeMo RL: A Scalable and Efficient Post-Training Library},
howpublished = {\url{https://github.com/NVIDIA-NeMo/RL}},
year = {2025},
note = {GitHub repository},
}
```

## Contributing

We welcome contributions to NeMo RL\! Please see our [Contributing Guidelines](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md) for more information on how to get involved.

## Licenses

NVIDIA NeMo RL is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/RL/blob/main/LICENSE).
