# Installation and Prerequisites

## Clone the Repository

Clone **NeMo RL** with submodules:

```sh
git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
cd nemo-rl

# If you are already cloned without the recursive option, you can initialize the submodules recursively
git submodule update --init --recursive

# Different branches of the repo can have different pinned versions of these third-party submodules. Ensure
# submodules are automatically updated after switching branches or pulling updates by configuring git with:
# git config submodule.recurse true

# **NOTE**: this setting will not download **new** or remove **old** submodules with the branch's changes.
# You will have to run the full `git submodule update --init --recursive` command in these situations.
```

## Install System Dependencies

### cuDNN (For Megatron Backend)

If you are using the Megatron backend on bare metal (outside of a container), you may need to install the cuDNN headers. Here is how you check and install them:

```sh
# Check if you have libcudnn installed
dpkg -l | grep cudnn.*cuda

# Find the version you need here: https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network
# As an example, these are the "Linux Ubuntu 20.04 x86_64" instructions
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cudnn  # Will install cuDNN meta packages which points to the latest versions
# sudo apt install cudnn9-cuda-12  # Will install cuDNN version 9.x.x compiled for cuda 12.x
# sudo apt install cudnn9-cuda-12-8  # Will install cuDNN version 9.x.x compiled for cuda 12.8
```

### libibverbs (For vLLM Dependencies)

If you encounter problems when installing vllm's dependency `deepspeed` on bare-metal (outside of a container), you may need to install `libibverbs-dev`:

```sh
sudo apt-get update
sudo apt-get install libibverbs-dev
```

## Install UV Package Manager

For faster setup and environment isolation, we use [uv](https://docs.astral.sh/uv/).

Follow [these instructions](https://docs.astral.sh/uv/getting-started/installation/) to install uv.

Quick install:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Create Virtual Environment

Initialize the NeMo RL project virtual environment:

```sh
uv venv
```

```{note}
Please do not use `-p/--python` and instead allow `uv venv` to read it from `.python-version`.
This ensures that the version of python used is always what we prescribe.
```

## Build Flash Attention (Optional but Recommended)

If working outside a container, it can help to build [flash-attn](https://github.com/Dao-AILab/flash-attention) and warm the uv cache before your first run:

```sh
bash tools/build-flash-attn-in-uv-cache.sh
```

```{note}
On the first install, `flash-attn` can take a while to install (approximately 45 minutes with 48 CPU hyperthreads). After it is built once, it is cached in your uv's cache directory, making subsequent installs much quicker.
```

```{tip}
The NeMo RL Dockerfile will warm the uv cache with flash-attn.
See [Docker documentation](../docker.md) for instructions if you are looking for the NeMo RL container.
```

If successful, you should see `flash-attn successfully added to uv cache`.

## Using UV to Run Commands

Use `uv run` to launch all commands. It handles pip installing implicitly and ensures your environment is up to date with our lock file.

```sh
# Example: Run GRPO with DTensor backend
uv run python examples/run_grpo_math.py

# Example: Run GRPO with Megatron backend
uv run python examples/run_grpo_math.py --config examples/configs/grpo_math_1B_megatron.yaml
```

```{note}
- It is not recommended to activate the `venv`, and you should use `uv run <command>` instead to execute scripts within the managed environment.
  This ensures consistent environment usage across different shells and sessions.
- Ensure your system has the appropriate CUDA drivers installed, and that your PyTorch version is compatible with both your CUDA setup and hardware.
- If you update your environment in `pyproject.toml`, it is necessary to force a rebuild of the virtual environments by setting `NRL_FORCE_REBUILD_VENVS=true` next time you launch a run.
- **Reminder**: Don't forget to set your `HF_HOME`, `WANDB_API_KEY`, and `HF_DATASETS_CACHE` (if needed). You'll need to do a `huggingface-cli login` as well for Llama models.
```

