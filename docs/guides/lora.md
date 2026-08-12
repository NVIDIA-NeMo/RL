# LoRA (Low-Rank Adaptation)

NeMo RL supports LoRA for parameter-efficient fine-tuning. LoRA reduces the number of
trainable parameters by learning low-rank update matrices for selected linear layers while
keeping the base model frozen. This lowers memory usage and checkpoint size, and enables
serving many task-specific adapters on top of a single shared base model.

This page is the single reference for LoRA in NeMo RL. It covers backend support, a
side-by-side schema comparison, backend-specific config examples, and links to the
algorithm guides and checkpoint-export tooling.

For the theory behind the method, see
[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

## Backend Support

LoRA is implemented on two training backends, each with its own config schema:

| Backend | Config path | Notes |
| --- | --- | --- |
| **DTensor (Automodel)** | `policy.dtensor_cfg.lora_cfg` | Requires DTensor v2 (`policy.dtensor_cfg._v2=true`). DTensor v1 does **not** support LoRA. This is the default backend. |
| **Megatron Core** | `policy.megatron_cfg.peft` | Requires `policy.megatron_cfg.enabled=true` (and `policy.dtensor_cfg.enabled=false`). |

LoRA is supported across the SFT, GRPO, and DPO algorithms on both backends.

```{note}
On the DTensor backend, Triton-optimized LoRA kernels are only used in the DTensor v2 path.
Automodel does not support Triton kernels when `tensor_parallel_size > 1`, so set
`use_triton: false` in that case (see the parameter details below).
```

## Schema Comparison

The two backends share most of their fields but differ in a few backend-specific options.
The table below maps equivalent fields and highlights the differences.

| Concept | DTensor (`lora_cfg`) | Megatron (`peft`) |
| --- | --- | --- |
| Enable LoRA | `enabled` | `enabled` |
| Modules to adapt | `target_modules` | `target_modules` |
| Modules to skip | `exclude_modules` | `exclude_modules` |
| Adapt all linear layers | `match_all_linear` | _(implicit: empty `target_modules` adapts all linear layers)_ |
| Rank (r) | `dim` | `dim` |
| Scaling factor | `alpha` | `alpha` |
| Dropout probability | `dropout` | `dropout` |
| Dropout position | `dropout_position` (`"pre"`/`"post"`) | `dropout_position` (`"pre"`/`"post"`) |
| A-matrix init | `lora_A_init` | `lora_A_init_method` |
| B-matrix init | _(always zero)_ | `lora_B_init_method` |
| Optimized kernels | `use_triton` | — |
| Adapter dtype | _(follows base layer)_ | `lora_dtype` |
| Experimental A2A comm | — | `a2a_experimental` |

The effective learning-rate multiplier for the adapter is `alpha / dim` on both backends.

## DTensor Configuration

LoRA settings live under `policy.dtensor_cfg.lora_cfg`:

```yaml
policy:
  dtensor_cfg:
    _v2: true                   # LoRA requires DTensor v2
    lora_cfg:
      enabled: False            # Set to True to enable LoRA fine-tuning
      target_modules: []        # List of module names to apply LoRA
      exclude_modules: []       # List of module names to exclude from LoRA
      match_all_linear: true    # Apply LoRA to all linear layers
      dim: 8                    # LoRA rank (r): controls adaptation capacity
      alpha: 32                 # LoRA scaling factor (effective lr = alpha/dim)
      dropout: 0.0              # Dropout probability for LoRA layers
      dropout_position: "post"  # Dropout position: "pre" or "post"
      lora_A_init: "xavier"     # Initialization method: "xavier" or "uniform"
      use_triton: true          # Use Triton-optimized kernels (DTensor v2 path)
```

### DTensor Parameter Details

- **`enabled`** (bool): Whether to enable LoRA training.
- **`target_modules`** (list): Specific module names to apply LoRA. Empty with `match_all_linear=true` applies to all linear layers.
- **`exclude_modules`** (list): Module names to exclude from LoRA.
- **`match_all_linear`** (bool): When `true`, applies LoRA to all linear layers (overrides `target_modules`).
- **`dim`** (int): LoRA rank (r). Lower values = fewer parameters but less capacity. Typical: 4, 8, 16, 32, 64.
- **`alpha`** (int): LoRA scaling factor. Effective learning rate multiplier = `alpha/dim`. Typical: 16, 32, 64.
- **`dropout`** (float): Dropout probability for regularization.
- **`dropout_position`** (str): Apply dropout before (`"pre"`) or after (`"post"`) LoRA.
- **`lora_A_init`** (str): Initialization method for the LoRA A matrix (`"xavier"` or `"uniform"`). The B matrix is always initialized to zero.
- **`use_triton`** (bool): Use Triton-optimized kernels for better performance. Used for DTensor v2 only. **Note**: [Automodel does not support Triton for TP > 1](https://github.com/NVIDIA-NeMo/Automodel/blob/b2db55eee98dfe81a8bfe5e23ac4e57afd8ab261/nemo_automodel/recipes/llm/train_ft.py#L199). Set to `false` when `tensor_parallel_size > 1` to avoid compatibility issues.

## Megatron Configuration

LoRA settings live under `policy.megatron_cfg.peft`:

```yaml
policy:
  megatron_cfg:
    peft:
      enabled: false                # Set to True to enable LoRA fine-tuning
      target_modules: []            # List of module names to apply LoRA, defaults to all linear layers
      exclude_modules: []           # List of module names not to apply LoRA
      dim: 32                       # LoRA rank (r): controls adaptation capacity
      alpha: 32                     # LoRA scaling factor (effective lr = alpha/dim)
      dropout: 0.0                  # Dropout probability for LoRA layers
      dropout_position: "pre"       # Dropout position: "pre" or "post"
      lora_A_init_method: "xavier"  # Initialization method for lora A: "xavier" or "uniform"
      lora_B_init_method: "zero"    # Initialization method for lora B: "zero"
      a2a_experimental: false       # Enables the experimental All-to-All (A2A) communication strategy
      lora_dtype: None              # Adapter weights dtype
```

### Megatron Parameter Details

- **`enabled`** (bool): Whether to enable LoRA training.
- **`target_modules`** (list): Specific module names to apply LoRA. Defaults to all linear layers if the list is left empty. Example: `['linear_qkv', 'linear_proj', 'linear_fc1', 'linear_fc2']`.
  - `linear_qkv`: Apply LoRA to the fused linear layer used for query, key, and value projections in self-attention.
  - `linear_proj`: Apply LoRA to the linear layer used for projecting the output of self-attention.
  - `linear_fc1`: Apply LoRA to the first fully-connected layer in MLP.
  - `linear_fc2`: Apply LoRA to the second fully-connected layer in MLP.

  Target modules can also contain wildcards. For example, you can specify `target_modules=['*.layers.0.*.linear_qkv', '*.layers.1.*.linear_qkv']` to add LoRA to only `linear_qkv` on the first two layers.
- **`exclude_modules`** (list, optional): A list of module names not to apply LoRA. It will match all `nn.Linear` & `nn.Linear`-adjacent modules whose name does not match any string in `exclude_modules`. If used, it requires `target_modules` to be an empty list or `None`.
- **`dim`** (int): LoRA rank (r). Lower values = fewer parameters but less capacity. Typical: 4, 8, 16, 32, 64.
- **`alpha`** (int): LoRA scaling factor. Effective learning rate multiplier = `alpha/dim`. Typical: 16, 32, 64.
- **`dropout`** (float): Dropout probability for regularization, defaults to 0.0.
- **`dropout_position`** (str): Apply dropout before (`"pre"`) or after (`"post"`) LoRA.
- **`lora_A_init_method`** (str): Initialization method for `lora_A` (choices: `['xavier', 'uniform']`), defaults to `xavier`.
- **`lora_B_init_method`** (str): Initialization method for the low-rank matrix B. Defaults to `"zero"`.
- **`a2a_experimental`** (bool): Enables the experimental All-to-All (A2A) communication strategy. Defaults to `False`.
- **`lora_dtype`** (torch.dtype): Adapter weights dtype. By default it follows `orig_linear`'s dtype, but for quantized weights (e.g. 4-bit) it must be specified explicitly.

## Usage by Algorithm

### SFT

The config uses the DTensor backend by default, so DTensor LoRA only requires enabling the flag:

```bash
uv run examples/run_sft.py policy.dtensor_cfg.lora_cfg.enabled=true
```

To use the Megatron backend, switch backends and enable the PEFT block:

```sh
uv run examples/run_sft.py \
  --config examples/configs/sft.yaml \
  policy.dtensor_cfg.enabled=false \
  policy.megatron_cfg.enabled=true \
  policy.megatron_cfg.peft.enabled=true
```

See the [SFT guide](sft.md) for the full SFT workflow.

### GRPO

GRPO supports LoRA on both backends. Enable the DTensor adapter with:

```bash
uv run examples/run_grpo.py policy.dtensor_cfg.lora_cfg.enabled=true
```

The DTensor GRPO LoRA path uses a **merge-weight** approach: during generation, LoRA adapter
weights are merged into the base linear weights. This improves performance at the cost of a
small train/inference mismatch that we consider acceptable. If you require strict
train/inference parity, use the
[split-weight variant branch](https://github.com/NVIDIA-NeMo/RL/tree/ruit/lora_grpo_async),
which may trade off some performance. For a comparison between merge-weight and split-weight,
see [PR 1797: Support lora in dtensor grpo workflow by merging weight](https://github.com/NVIDIA-NeMo/RL/pull/1797).

See the [GRPO guide](grpo.md) for the full GRPO workflow.

### DPO

DPO fully supports LoRA on **both** the DTensor and Megatron backends, using the same
`lora_cfg` / `peft` config blocks as SFT and GRPO. There is no dedicated DPO LoRA recipe;
enable it on an existing DPO config via an override. For the DTensor backend:

```bash
uv run examples/run_dpo.py policy.dtensor_cfg.lora_cfg.enabled=true
```

For the Megatron backend:

```bash
uv run examples/run_dpo.py \
  --config examples/configs/dpo.yaml \
  policy.dtensor_cfg.enabled=false \
  policy.megatron_cfg.enabled=true \
  policy.megatron_cfg.peft.enabled=true
```

See the [DPO guide](dpo.md) for the full DPO workflow.

## Example Recipes

Ready-to-run LoRA recipes live under `examples/configs/recipes/llm/`:

| Recipe | Algorithm | Backend |
| --- | --- | --- |
| [`sft-llama3.1-8b-1n8g-fsdp2tp1-lora.yaml`](../../examples/configs/recipes/llm/sft-llama3.1-8b-1n8g-fsdp2tp1-lora.yaml) | SFT | DTensor |
| [`sft-llama3.1-8b-1n8g-megatron-lora.yaml`](../../examples/configs/recipes/llm/sft-llama3.1-8b-1n8g-megatron-lora.yaml) | SFT | Megatron |
| [`sft-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`](../../examples/configs/recipes/llm/sft-nanov3-30BA3B-2n8g-fsdp2-lora.yaml) | SFT | DTensor |
| [`grpo-qwen3-8B-base-1n8g-fsdp2-lora.yaml`](../../examples/configs/recipes/llm/grpo-qwen3-8B-base-1n8g-fsdp2-lora.yaml) | GRPO | DTensor |
| [`grpo-qwen3-8b-base-1n8g-megatron-lora.yaml`](../../examples/configs/recipes/llm/grpo-qwen3-8b-base-1n8g-megatron-lora.yaml) | GRPO | Megatron |
| [`grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`](../../examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml) | GRPO | DTensor |
| [`grpo-nanov3-30BA3B-2n8g-megatron-lora.yaml`](../../examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron-lora.yaml) | GRPO | Megatron |

DPO LoRA is supported on both backends but ships no dedicated recipe — enable it on an existing DPO config with the overrides shown in [Usage by Algorithm → DPO](#dpo).

## Exporting a LoRA Checkpoint to Hugging Face Format

After training with LoRA on the Megatron backend, the `convert_lora_to_hf.py` script supports
two export modes:

- **Merged**: fold the adapter into the base model and export a single standalone Hugging Face checkpoint for inference or evaluation.
- **Adapter-only**: export only the adapter weights in Hugging Face PEFT format, keeping the base model separate (e.g. for use with vLLM's LoRA support).

See the [Checkpointing documentation](../design-docs/checkpointing.md#converting-megatron-lora-adapter-checkpoints-to-hugging-face-format) for full usage details and examples.
