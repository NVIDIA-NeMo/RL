# Add New Models

This guide outlines how to integrate and validate a new model within NeMo RL. Each new model must pass a standard set of compatibility tests before being considered ready to be used in RL pipelines. The guide also details diagnostic scripts to help identify and resolve common issues during model integration.

## Importance of Log Probability Consistency in Training and Inference

In on-policy RL, we sample tokens (actions) from the latest version of the policy. This means the sampling distribution of token probabilities produced by the inference framework must closely match those from the training framework. If the inference framework produces significantly different probabilities, we effectively sample from a different distribution, leading to errors in the loss estimation.

As an example, we would see errors in naive KL estimation:

$$\text{KL} = E_{x \sim \pi}[\pi(x) - \pi_{\text{ref}}(x)]$$

When summed/integrated, replacing the $x \sim \pi$ with $x \sim \pi_{\text{wrong}}$ leads to an error of:

$$\sum_{x} \left( \pi(x) - \pi_{\text{ref}}(x) \right) \left( \pi_{\text{wrong}}(x) - \pi(x) \right)$$

So, to verify correctness, we calculate:

$$
\frac{1}{n}\sum_{i=1}^{n\text{(tokens)}}\exp\left(\left\|\text{logprobs-train-fwk}_i - \text{logprobs-inference-fwk}_i\right\|\right)
$$

as a measure of multiplicative probability error for sampled tokens, where samples are drawn as $x \sim \pi_{\text{inference-framework}}$.

Note that this is not exhaustive (the inference framework could lack distribution support and we wouldn't catch it here, as $x \sim \pi_{\text{inference-framework}}$). To get a much stricter guarantee on correctness, you should run this metric twice and average the results, where in the second run, you sample $x \sim \pi_{\text{training-framework}}$. In practice, we use just the former in our tests and find it sufficient.

## Understand Discrepancies Between Backends

When validating models across different backends, you may encounter discrepancies in log probabilities. These differences can stem from various sources with effects ranging from negligible to significant:

- **Numerical precision differences**: Training and inference backends may differ in precision formats (FP32, FP16, BF16, FP8).
  - Training may use mixed precision, while the inference backend may not.
  - High-precision training with FP8 inference may not be numerically stable for certain models.
  - Differences can occur at the layer level, with some layers in FP32, while others use lower precision.

- **Implementation variations**: Subtle differences in how layer implementations like softmax, layer normalization, or attention mechanisms are implemented.
  - Attention/Norm layers (which could be fused) in TransformerEngine may not be bit-wise identical to implementations in inference backends.
  - Inference backends may re-implement kernels (e.g., for SSM layers) leading to differences.
  - Softmax in training frameworks may be calculated differently than in inference backends for numerical stability.

- **KV/Prefill cache handling**: Differences in how key-value/prefill caches are managed during autoregressive generation.
  - In some cases, disabling the inference backend cache can resolve discrepancies.

- **Parallelism effects**: Parallelisms like Tensor parallelism may introduce small variations.

- **Inherent non-determinism**: Some neural network operations are inherently non-deterministic (e.g., `torch.cumsum`).

- **Prefill/Decoding kernel mismatch**: Different kernels for prefill and decoding phases may produce different log probabilities.
  - Training frameworks typically use prefill kernels, while inference backends may use both prefill kernels and specialized decoding kernels.

- **Imperfect Refit**: Weight conversion from the training framework to the inference backend may be incomplete or data formats may be incorrect.
  - If weights are reshaped or reordered incorrectly, generations tend to be very wrong.
  - In some cases, if some weights in the inference backend are not refit after each training step, the error between training and inference log probabilities can diverge as training progresses.

- **Batch size**: In some cases, `batch_size>1` may produce larger errors than `batch_size=1`

When investigating discrepancies beyond the acceptable threshold, focus on these areas and determine whether the differences appear systematically or only in specific contexts.


---

## 1. Hugging Face–Based Models

### Validation Workflow

When validating Hugging Face-based models, perform the following checks:

- **Compare log probabilities**
  Ensure the generation log probabilities from inference backends like **vLLM** match those computed by Hugging Face. This comparison helps diagnose potential mismatches.

- **Test parallelism**
  Verify consistency with other parallelism settings.

- **Variance**
  Repeat tests multiple times (e.g., 10 runs) to confirm that behavior is deterministic or within acceptable variance.

- **Check sequence lengths**
  Perform inference on sequence lengths of 100, 1,000, and 10,000 tokens.
  Ensure the model behaves consistently at each length.

- **Use real and dummy data**
  - **Real data:** Tokenize and generate from actual text samples.
  - **Dummy data:** Simple numeric sequences to test basic generation.

- **Vary sampling parameters**
  Test both greedy and sampling generation modes.
  Adjust temperature and top-p to confirm output consistency across backends.

- **Test different batch sizes**
  Try with batch sizes of 1, 8, and 32 to ensure consistent behavior across different batch configurations.

---

## 2. Megatron Models

### Additional Validation

- **Compare Megatron outputs**
  Ensure the Megatron forward pass aligns with Hugging Face and the generation log probabilities from inference backends like **vLLM**.

- **Parallel settings**
  Match the same parallelism configurations used for the HuggingFace-based tests.
  Confirm outputs remain consistent across repeated runs.

---

## 3. Expected Error Threshold

When comparing log probabilities between training and inference backends, we use an error threshold of `1.05` to determine acceptable variance (for equal precision). An error of `1.0` indicates a perfect match, and values exceeding `1.05` require further investigation.

When validating your model, you should analyze the results across different configurations. Your analysis should include:

| Sequence Length | Data Type  | Generation Method | Batch Size | HF vs VLLM | Megatron vs VLLM |
|-----------------|------------|-------------------|------------|------------|------------------|
| 100             | Real       | Greedy            | 1          | 1.02       | 1.01             |
| 100             | Real       | Sampling          | 8          | 1.03       | 1.02             |
| 100             | Synthetic  | Greedy            | 1          | 1.01       | 1.02             |
| 1,000           | Real       | Greedy            | 32         | 1.04       | 1.03             |
| ...             | ...        | ...               | ...        | ...        | ...              |

---

By following these validation steps and ensuring your model's outputs remain consistent across backends, you can confirm that your new model meets the requirements of NeMo RL.


# Model Diagnostics

We also maintain a set of standalone scripts that can be used to diagnose issues related to correctness that
we have encountered before.

## [1.max_model_len_respected.py](https://github.com/NVIDIA-NeMo/RL/blob/main/tools/model_diagnostics/1.max_model_len_respected.py)

Test if a new model respects the `max_model_len` passed to vllm:

```sh
# Run that is expected to pass
uv run --extra vllm tools/model_diagnostics/1.max_model_len_respected.py Qwen/Qwen2.5-1.5B
# ...
# Prompt tokens: 8
# Generated tokens: 12
# Total tokens: 20
# [Qwen/Qwen2.5-1.5B] ALL GOOD!
```

## [2.long_generation_decode_vs_prefill](https://github.com/NVIDIA-NeMo/RL/blob/main/tools/model_diagnostics/2.long_generation_decode_vs_prefill.py)

Test that vLLM yields near-identical token log-probabilities when comparing decoding with a single prefill pass across multiple prompts.

```sh
# Run that is expected to pass
uv run --extra vllm tools/model_diagnostics/2.long_generation_decode_vs_prefill.py Qwen/Qwen2.5-1.5B
# ...
# [Qwen/Qwen2.5-1.5B] ALL GOOD!
```

## [3.check_hf_model_embeddings_untrained.py](https://github.com/NVIDIA-NeMo/RL/blob/main/tools/model_diagnostics/3.check_hf_model_embeddings_untrained.py)

Detects untrained or improperly initialized Hugging Face model embeddings by scanning for near-zero rows and rows with near-identical values in both input and output embeddings. The script also reports whether word embeddings are tied and summarizes basic statistics.

```sh
# Example run
uv run --extra mcore tools/model_diagnostics/3.check_hf_model_embeddings_untrained.py --model nvidia/Nemotron-H-8B-Base-8K

# ....
#================================================================================
#EMBEDDING SUMMARIES
#================================================================================
#
#--- Input Embeddings Summary ---
#Shape: torch.Size([131072, 4096]), Dtype: torch.bfloat16
#Near-zero embeddings (abs < 1.00e-10): 1039/131072 (0.8%)
#  Indices: 0-1,3-999,1192-1193,1245-1255,55014,77579,81772,81819,82312,82500,82725,82737,82977,84020,84121,84521,84794,85015,86409,87411,89412,90320,91368,94485,96385,104097,108262,112147,112327,112497,114755
#Identical embeddings (std < 1.00e-08): 1041/131072 (0.8%)
#  Indices: 0-1,3-999,1192-1193,1245-1255,55014,77579,81772,81819,82312,82500,82725,82737,82977,83855,84020,84121,84521,84794,85015,86409,87411,89412,90320,91368,94485,96385,101707,104097,108262,112147,112327,112497,114755
#Statistics: mean_abs=0.007874, max_abs=0.196289, std_range=[0.000000, 0.015442]
#⚠️  POTENTIAL ISSUES: 1039 near-zero embeddings, 1041 identical embeddings
#
#--- Output Embeddings Summary (Tied: False) ---
#Shape: torch.Size([131072, 4096]), Dtype: torch.bfloat16
#Near-zero embeddings (abs < 1.00e-10): 0/131072 (0.0%)
#Identical embeddings (std < 1.00e-08): 0/131072 (0.0%)
#Statistics: mean_abs=0.006775, max_abs=0.200195, std_range=[0.004089, 0.021240]
#✅ No obvious untrained patterns detected
#
#=== Final Summary ===
#Model: nvidia/Nemotron-H-8B-Base-8K
#Analysis complete.
```

- Thresholds can be adjusted via flags:
  - `--near-zero-threshold` (default: `1e-10`)
  - `--identical-threshold` (default: `1e-8`)
- If any near-zero or identical rows are reported, the model may have issues of numerical instability (e.g., inf grad norms) during post-training if any of these problematic tokens are encountered. We have observed this happening when special tokens are reserved in the tokenizer and embedding, but none are encountered during pre-training. It may help to initialize these embeddings similar to how they were initialize during pre-training.