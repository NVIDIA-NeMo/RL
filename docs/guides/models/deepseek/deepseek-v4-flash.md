# DeepSeek V4 Flash

This guide describes GRPO training of DeepSeek V4 Flash and Flash Base with the
AutoModel training backend and vLLM generation.

> [!IMPORTANT]
> **Status: Functionally Ready.** The reference recipe has been validated with
> an end-to-end colocated run on 16 H100 nodes. Its five-step smoke test is
> registered in the recurring nightly suite. This is not a long-run convergence
> claim; see [Known Limitations](#known-limitations) for precision alignment and
> hardware coverage.

## Support Status

| Model | Training backend | Validated training parallelism | Generation backend | Status |
| --- | --- | --- | --- | --- |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | AutoModel | TP1 + CP8 + EP128 | vLLM with TP8 + EP1 | Functionally Ready |
| `deepseek-ai/DeepSeek-V4-Flash-Base` | AutoModel | TP1 + CP8 + EP128 | vLLM with TP8 + EP1 | Functionally Ready |

## Validated Scope

- **Models**: `deepseek-ai/DeepSeek-V4-Flash-0731` and
  `deepseek-ai/DeepSeek-V4-Flash-Base`.
- **Algorithm**: GRPO with `DAPOMath17K` for training and
  `DAPOMathAIME2024` for validation.
- **Training backend**: AutoModel with BF16 training, activation checkpointing,
  TileLang attention, and the DeepEP expert dispatcher.
- **Training parallelism**: TP1, CP8, and EP128 on 128 GPUs. CP and EP are
  model-owned parallel dimensions that coexist on the same device mesh; they
  are not multiplicative.
- **Generation backend**: vLLM with TP8, EP1, blockwise FP8 weights, and the
  `fp8_ds_mla` KV cache.
- **Sequence length**: 1,024 prompt tokens plus up to 2,048 response tokens,
  for a maximum total sequence length of 3,072.
- **Reference allocation**: 16 nodes with 8 H100 GPUs per node.
- **Deployment**: Colocated training and generation on the same GPUs. Only
  this deployment on H100 has been tested.
- **MTP**: Disabled by setting `num_nextn_predict_layers: 0`.

Recipe YAML files under `examples/configs/recipes/` are the source of truth for
resource, parallelism, dataset, and checkpointing settings.

## How to Run

### 1. Prepare the Environment

Use the dependency lock and AutoModel submodule recorded by the NeMo RL
revision that contains this guide. From the repository root, run:

```bash
git submodule update --init --recursive
uv sync --locked
```

For a large multi-node run, use a container built from the same checkout so its
backend-specific worker environments match the lockfile. See the
[installation guide](../../../about/installation.md) and
[Dependency Management](../../../design-docs/dependency-management.md) for
container and development setup details.

The recipe downloads the following model and datasets from Hugging Face:

- `deepseek-ai/DeepSeek-V4-Flash-0731` or
  `deepseek-ai/DeepSeek-V4-Flash-Base`
- `BytedTsinghua-SIA/DAPO-Math-17k`
- `BytedTsinghua-SIA/AIME-2024`

Set `HF_HOME` to a cache visible from every node:

```bash
export HF_HOME=<path-to-shared-huggingface-cache>
export WANDB_API_KEY=<your-wandb-api-key>
```

The reference recipe enables W&B logging. If W&B is not configured, pass
`logger.wandb_enabled=false` when launching.

### 2. Choose the Reference Recipe

| Model | Algorithm | Backend | Scale | Recipe |
| --- | --- | --- | --- | --- |
| DeepSeek-V4-Flash-0731 | GRPO | AutoModel | 16n8g | [`grpo-deepseek-v4-flash-0731-16n8g-automodel-cp8ep128.yaml`](../../../../examples/configs/recipes/llm/grpo-deepseek-v4-flash-0731-16n8g-automodel-cp8ep128.yaml) |

The associated
[`grpo-deepseek-v4-flash-0731-16n8g-automodel-cp8ep128.sh`](../../../../tests/test_suites/llm/grpo-deepseek-v4-flash-0731-16n8g-automodel-cp8ep128.sh)
test runs five training steps and is registered as a recurring nightly
functional test in [`nightly.txt`](../../../../tests/test_suites/nightly.txt).

### 3. Launch

From a 16-node allocation with 8 H100 GPUs per node, launch the standard GRPO
entry point:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-deepseek-v4-flash-0731-16n8g-automodel-cp8ep128.yaml
```

The recipe defaults to `DeepSeek-V4-Flash-0731`. To run Flash Base with the
same configuration, override the model and tokenizer:

```bash
uv run examples/run_grpo.py \
  --config examples/configs/recipes/llm/grpo-deepseek-v4-flash-0731-16n8g-automodel-cp8ep128.yaml \
  policy.model_name=deepseek-ai/DeepSeek-V4-Flash-Base \
  policy.tokenizer.name=deepseek-ai/DeepSeek-V4-Flash-Base
```

See the [GRPO guide](../../grpo.md) for algorithm and common configuration
details and [Cluster Setup](../../../cluster.md) for multi-node launch setup.
Before changing the node count, review the training and generation parallel
dimensions instead of changing `cluster.num_nodes` alone.

## Important Recipe Settings

- `policy.dequantize_base_checkpoint: true` loads either checkpoint as BF16
  weights for training. Flash-0731 stores its expert weights in FP4, while
  Flash Base stores them in FP8. Training-side FP8 fake quantization is not
  enabled.
- `policy.hf_config_overrides.expert_dtype: fp8` selects the FP8 expert layout
  used for generation. Keep the checkpoint's own `quantization_config` intact
  so AutoModel can dequantize the training weights correctly.
- Generation uses DeepGEMM with UE8M0 power-of-two scales. Keep
  `VLLM_USE_DEEP_GEMM_E8M0=1`, `use_deep_gemm: true`, and
  `pow2_weight_scaling_factors: true` aligned.
- `policy.generation.vllm_kwargs.attention_config.backend: FLASHMLA_SPARSE_DSV4`
  pins the validated FlashMLA sparse attention backend.
- The policy tokenizer uses `chat_template: deepseek_v4`, while vLLM uses
  `tokenizer_mode: deepseek_v4`. The reference recipe disables thinking mode.
- Eager execution (`enforce_eager: true`) remains part of the validated
  configuration.

## Reference Training Curves

The following curves were produced with `DeepSeek-V4-Flash-0731` on the
16-node H100 configuration described above. They show the raw, unsmoothed
training and validation metrics through training step 50.

![DeepSeek-V4-Flash-0731 DAPO-GRPO training reward, validation accuracy, response length, entropy, generation KL error, and gradient norm through step 50](../../../assets/deepseek/deepseek-v4-flash-0731-grpo-50steps.png)

## Known Limitations

- The checked-in support is limited to `DeepSeek-V4-Flash-0731` and
  `DeepSeek-V4-Flash-Base` with the AutoModel training backend and vLLM
  generation. DeepSeek V4 Pro, Megatron training, and SGLang generation are not
  covered by this guide.
- **Hardware and deployment coverage**: Only colocated training and generation
  on H100 GPUs have been tested. Non-colocated deployments and Blackwell GPUs
  have not been validated. On Blackwell, DeepGEMM with E8M0 scales has a
  potential refit issue: changes to the FP8 scale layout can leave the MoE
  kernel referencing stale scales.
- Training uses dequantized BF16 weights, while generation uses refitted FP8
  weights. This numerical precision mismatch causes significant train/inference
  inconsistency. The current recipe does not include training-side
  quantization-aware training (QAT) to reduce this mismatch.
- MTP is disabled in the reference configuration.
- Long-run convergence has not been documented for this recipe.

The separate
[`deepseek-v4-support` development branch](https://github.com/NVIDIA-NeMo/RL/tree/deepseek-v4-support)
includes QAT through training-side FP8 fake quantization. It reduces the
train/inference mismatch and has enabled stable training beyond 100 steps.
See the [development branch guide](https://github.com/NVIDIA-NeMo/RL/blob/deepseek-v4-support/docs/guides/deepseek-v4.md)
for its recipes and validation results.
These results apply to that development branch, not the current recipe. The
branch uses an older NeMo-RL codebase and a customized AutoModel dependency,
so its changes cannot be merged directly into `main`. We plan to upstream the
relevant changes incrementally into NeMo-RL `main` and the upstream dependency
frameworks, including AutoModel.
