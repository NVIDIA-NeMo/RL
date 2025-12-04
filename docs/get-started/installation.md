---
description: "Detailed step-by-step guide for installing NeMo RL, including bare-metal dependencies and troubleshooting"
categories: ["getting-started", "setup"]
tags: ["installation", "dependencies", "setup", "system-requirements"]
personas: ["data-scientist-focused", "mle-focused", "cluster-administrator-focused"]
difficulty: "beginner"
content_type: "tutorial"
---

# Installation and Prerequisites

Use this guide if you are setting up on a bare-metal system, need specific backend dependencies (like Megatron or vLLM), or are troubleshooting your environment.

:::{card}
**Goal**: Fully configure your system, install NeMo RL dependencies, and prepare the virtual environment.

^^^

**Steps**:

1. **System Dependencies**: Install backend-specific libraries (cuDNN, libibverbs).
2. **Clone**: Get the source code with submodules.
3. **Package Manager**: Install `uv`.
4. **Virtual Env**: Create and verify your Python environment.
:::

:::{button-ref} index
:color: secondary
:outline:
:ref-type: doc

← Back to Quickstart
:::

---

## 1. Install System Dependencies

Before installing the Python package, ensure your operating system has the required libraries for your chosen backend.

:::{note}
If you are using a pre-built NVIDIA container (e.g., from NGC), most of these dependencies are likely pre-installed. These steps are critical for **bare-metal** installations (e.g., a fresh Ubuntu server).
:::

::::{tab-set}

:::{tab-item} Megatron Backend (Bare Metal)
If you plan to use the **Megatron Core** backend, you must have the cuDNN headers installed.

**Check for existing installation:**
```sh
dpkg -l | grep cudnn.*cuda
```

**Install cuDNN (Ubuntu 20.04/22.04 example):**

Find the correct version for your system at the [NVIDIA cuDNN Downloads page](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network).

1. Add the NVIDIA repo key:
   ```sh
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt update
   ```

2. Install the libraries:
   ```sh
   # Installs meta-packages pointing to the latest versions
   sudo apt install cudnn

   # OR install specific versions (adjust version numbers as needed)
   # sudo apt install cudnn9-cuda-12
   ```
:::

:::{tab-item} vLLM / DeepSpeed
For the **vLLM** inference backend (often used with DeepSpeed), `libibverbs-dev` is required on bare metal to avoid build errors.

**Install libibverbs:**
```sh
sudo apt-get update
sudo apt-get install libibverbs-dev
```
:::

::::

---

## 2. Clone the Repository

NeMo RL relies on several third-party libraries included as git submodules. You must clone the repository recursively.

1. **Clone with recursion**:
   ```sh
   git clone git@github.com:NVIDIA-NeMo/RL.git nemo-rl --recursive
   cd nemo-rl
   ```

2. **(Optional) Initialize existing clone**:
   If you already cloned without the `--recursive` flag, fix it by running:
   ```sh
   git submodule update --init --recursive
   ```

:::{tip} Keep Submodules in Sync
Different branches may pin different versions of submodules. To ensure they update automatically when you switch branches or pull, configure git:

```sh
git config submodule.recurse true
```
*Note: This will not remove old submodules or download new ones if the directory structure changes significantly; in those cases, run the full update command above.*
:::

---

## 3. Install UV Package Manager

We use `uv` for fast, reliable, and isolated Python package management.

1. **Install `uv`**:
   Follow the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) or use the quick script:
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Verify installation**:
   ```sh
   uv --version
   ```

---

## 4. Create Virtual Environment

Initialize the project-specific virtual environment. NeMo RL uses a `.python-version` file to pin the supported Python version automatically.

1. **Create the venv**:
   ```sh
   uv venv
   ```

   :::{important}
   Do **not** specify a python version manually (e.g., `-p python3.10`). Let `uv` read the correct version from the configuration file to ensure compatibility.
   :::

2. **(Optional) Rebuilding Environments**:
   If you change branches or modify `pyproject.toml` significantly, you may need to force a rebuild of the environment variables and dependencies:
   ```sh
   NRL_FORCE_REBUILD_VENVS=true uv sync
   ```

---

## 5. Using UV to Run Commands

In NeMo RL, we recommend using `uv run` to execute scripts rather than manually activating the virtual environment. This ensures you are always using the locked dependencies and correct environment variables.

**Examples**:

*   **Run a Python script**:
    ```sh
    uv run python examples/run_grpo_math.py
    ```

*   **Run with arguments**:
    ```sh
    uv run python examples/run_grpo_math.py --config examples/configs/grpo_math_1B_megatron.yaml
    ```

---

## 6. Configure Access Tokens

Some models and datasets require authentication.

**Hugging Face Token** (for gated models like Llama):

```sh
# Set the token (avoids huggingface-cli login prompts)
export HF_TOKEN=your_token_here

# OR login interactively (required for Llama and other gated models)
huggingface-cli login
```

**Weights & Biases** (for experiment tracking):

```sh
export WANDB_API_KEY=your_key_here
```

:::{tip}
Add these exports to your `~/.bashrc` or `~/.zshrc` so they persist across sessions.
:::

---

## Troubleshooting

:::{dropdown} CUDA Compatibility
Ensure your system has the appropriate CUDA drivers installed and that your PyTorch version is compatible with both your CUDA setup and hardware. Run `nvidia-smi` to check your driver version.
:::

:::{dropdown} Gated Model Access Errors
If you see authentication errors when downloading models like Llama:
1. Accept the model license on Hugging Face (visit the model page)
2. Run `huggingface-cli login` or set `HF_TOKEN`
:::

:::{seealso}
Ready to run your first job? Go back to the [Quickstart Guide](index.md) or jump to [Supervised Fine-Tuning (SFT)](sft.md).
:::
