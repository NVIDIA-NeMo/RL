# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test vLLM logprobs_mode functionality to verify processed_logprobs matches expectations."""

import pytest
import torch


@pytest.mark.vllm
def test_processed_logprobs_matches_manual_computation():
    """Test that processed_logprobs mode matches manual computation from HF ground truth.

    Mathematical Framework:
    =======================
    For a full vocabulary V with logits x_i, after temperature τ and top_k/top_p filtering:

        processed_logprob(x_i) = log_softmax(filter(x_i/τ))

    Where filter applies top_k and top_p masking to create filtered set F ⊆ V:
        log_softmax_filtered(x_i) = x_i/τ - log(Σ_{j∈F} exp(x_j/τ))

    Test Strategy:
    ==============
    1. Generate 3 tokens with vLLM (processed_logprobs mode, float32)
       → Get sampled token IDs and vLLM's processed logprobs

    2. Load HuggingFace model (float32) and run single forward pass
       → Get ground truth logits for all 3 tokens from full vocabulary

    3. Manually compute for each token:
       - Apply temperature scaling: x_i/τ
       - Apply top_k/top_p filtering: apply_top_k_top_p(x_i/τ, k, p)
       - Compute log_softmax over filtered tokens

    4. Validate: assert vLLM logprobs ≈ manual logprobs using torch.testing.assert_close

    Notes:
    ===============
    - HF model provides FULL vocabulary logits (no missing probability mass)
        - Tried using raw_logits, but get hangs if SamplingParams.logprobs is too large (or -1)
    - Both vLLM and HF use float32 for consistent numerical precision
    - Validates our apply_top_k_top_p implementation matches vLLM exactly

    Note: Run with: uv run --extra vllm --group test pytest ... --vllm-only
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams

    from nemo_rl.models.policy.utils import apply_top_k_top_p

    # Use a small model for fast testing
    model_name = "facebook/opt-125m"

    # Sampling parameters (mathematical notation in docstring):
    # τ (tau) = temperature, k = top_k, p = top_p
    temperature = 1.5  # τ: temperature scaling factor
    num_logprobs = 500  # N: get top-500 logprobs from vLLM (for validation)
    top_k = 500  # k: top_k sampling parameter
    top_p = 0.9  # p: nucleus sampling threshold
    seed = 42  # Deterministic seed for reproducibility
    num_tokens = 3  # Generate 3 tokens for testing

    # New approach: Use HuggingFace to get ground truth logits, then compare against
    # vLLM's processed_logprobs. This avoids:
    # - O(n) hang with large logprobs
    # - Needing 2 vLLM instances
    # - Token set mismatch issues

    # Common parameters for both LLMs
    llm_kwargs = {
        "model": model_name,
        "seed": seed,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.3,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "dtype": "float32",  # Use float32 for maximum precision
    }

    prompt = "The quick brown fox jumps over the"

    # Step 1: Use vLLM to generate tokens and get which ones were sampled
    print(
        f"Step 1: Generating {num_tokens} tokens with vLLM (processed_logprobs mode)..."
    )
    llm_vllm = LLM(
        **llm_kwargs, logprobs_mode="processed_logprobs", max_logprobs=num_logprobs
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_tokens=num_tokens,
        logprobs=num_logprobs,
        seed=seed,
    )

    outputs_vllm = llm_vllm.generate([prompt], sampling_params=sampling_params)
    del llm_vllm  # Free GPU memory

    # Extract vLLM's processed logprobs and sampled token IDs
    vllm_token_ids = []
    vllm_logprobs = []

    for output in outputs_vllm[0].outputs:
        for sampled_token_id, logprob_dict in zip(output.token_ids, output.logprobs):
            vllm_token_ids.append(sampled_token_id)
            vllm_logprobs.append(logprob_dict[sampled_token_id].logprob)

    print(f"vLLM sampled tokens: {vllm_token_ids}")
    print(f"vLLM processed logprobs: {[f'{lp:.4f}' for lp in vllm_logprobs]}")

    # Step 2: Use HuggingFace model to get ground truth logits and manually compute
    print("\nStep 2: Loading HuggingFace model to get ground truth logits...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use float32 for maximum precision
    ).cuda()
    hf_model.eval()

    # Tokenize the prompt and append all generated tokens for a single forward pass
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    # Construct full sequence: prompt + all 3 generated tokens
    generated_ids = torch.tensor(
        [vllm_token_ids], dtype=torch.long, device=input_ids.device
    )
    full_sequence = torch.cat([input_ids, generated_ids], dim=1)

    print(f"Single forward pass with sequence length: {full_sequence.shape[1]}")

    # Single forward pass to get all logits
    with torch.no_grad():
        outputs_hf = hf_model(full_sequence)
        all_logits = outputs_hf.logits[0]  # Shape: [seq_len, vocab_size]

    # Extract logits at positions where we need to predict each generated token
    # For token i, we need logits at position (prompt_len + i - 1)
    prompt_len = input_ids.shape[1]
    expected_logprobs = []

    for i, sampled_token_id in enumerate(vllm_token_ids):
        # Get logits right before this token was generated
        logits = all_logits[prompt_len + i - 1, :]  # Shape: [vocab_size]

        # Manually compute processed logprobs following vLLM's processing pipeline:
        # Step 1: Apply temperature scaling → x_i/τ
        # All computations in float32 for maximum precision
        scaled_logits = logits / temperature

        # Step 2 & 3: Apply top_k and top_p filtering using vLLM's implementation
        scaled_logits_batched = scaled_logits.unsqueeze(0).unsqueeze(
            0
        )  # [1, 1, vocab_size]
        filtered_logits_batched = apply_top_k_top_p(
            scaled_logits_batched, top_k=top_k, top_p=top_p
        )
        filtered_logits = filtered_logits_batched.squeeze(0).squeeze(0)  # [vocab_size]

        # Step 4: Compute log_softmax over filtered tokens
        manual_logprobs = torch.nn.functional.log_softmax(filtered_logits, dim=0)

        # Get the logprob for the sampled token
        expected_logprobs.append(manual_logprobs[sampled_token_id].item())

    print(
        f"HF model computed logprobs (float32): {[f'{lp:.4f}' for lp in expected_logprobs]}"
    )

    # Step 3: Compare vLLM's processed_logprobs against our HF-based manual computation
    print("\nStep 3: Comparing logprobs...")
    expected_logprobs_tensor = torch.tensor(expected_logprobs)
    vllm_logprobs_tensor = torch.tensor(vllm_logprobs)

    # Print individual comparisons
    print("\nPer-token comparison:")
    for i in range(num_tokens):
        diff = abs(vllm_logprobs[i] - expected_logprobs[i])
        print(
            f"Token {i} (ID={vllm_token_ids[i]}): "
            f"manual={expected_logprobs[i]:.6f}, "
            f"vllm={vllm_logprobs[i]:.6f}, "
            f"diff={diff:.6f}"
        )

    # Use torch.testing.assert_close to validate the match
    print("\nValidating match with torch.testing.assert_close...")
    torch.testing.assert_close(
        vllm_logprobs_tensor,
        expected_logprobs_tensor,
        rtol=1e-3,
        atol=1e-2,
    )

    print("✓ Test passed: processed_logprobs match manual computation from HF model!")
    print(f"  Tokens: {vllm_token_ids}")
    print("  Validated with rtol=1e-3, atol=1e-2")


@pytest.mark.vllm
@pytest.mark.parametrize(
    "top_k,top_p,test_name",
    [
        (100, 0.9, "top_k + top_p"),
        (None, 0.9, "top_p only"),
        (100, None, "top_k only"),
        (None, None, "passthrough (no filtering)"),
    ],
)
def test_apply_top_k_top_p_matches_vllm_upstream(top_k, top_p, test_name):
    """Test that our apply_top_k_top_p implementation matches vLLM's upstream version.

    This test directly compares our simplified scalar-parameter implementation in
    nemo_rl.models.policy.utils against vLLM's batched tensor-parameter implementation.

    Key differences in interfaces:
    - Our version: scalar top_k/top_p, expects shape [batch, seq, vocab]
    - vLLM version: tensor top_k/top_p (one per batch), expects shape [batch, vocab]

    This test validates that for the same logits and parameters, both produce identical results.

    Args:
        top_k: Top-k value to test (or None)
        top_p: Top-p value to test (or None)
        test_name: Description of the test case
    """
    from vllm.v1.sample.ops.topk_topp_sampler import (
        apply_top_k_top_p as vllm_apply_top_k_top_p,
    )

    from nemo_rl.models.policy.utils import apply_top_k_top_p

    # Test configuration
    batch_size = 4
    seq_len = 2
    vocab_size = 1000

    # Generate synthetic logits (deterministic for reproducibility)
    torch.manual_seed(42)
    logits_3d = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32)

    print(f"Testing: {test_name}")

    # Our implementation: expects [batch, seq, vocab], takes scalar k/p
    our_result = apply_top_k_top_p(logits_3d.clone(), top_k=top_k, top_p=top_p)

    # vLLM upstream: expects [batch, vocab], takes tensor k/p with shape [batch]
    # Process each sequence position separately (vLLM doesn't batch over seq_len)
    vllm_results = []
    for seq_idx in range(seq_len):
        logits_2d = logits_3d[:, seq_idx, :].clone()  # [batch, vocab]

        # Convert scalar parameters to tensors for vLLM
        k_tensor = (
            None
            if top_k is None
            else torch.full((batch_size,), top_k, dtype=torch.long)
        )
        p_tensor = (
            None
            if top_p is None
            else torch.full((batch_size,), top_p, dtype=torch.float32)
        )

        vllm_result = vllm_apply_top_k_top_p(logits_2d, k=k_tensor, p=p_tensor)
        vllm_results.append(vllm_result)

    vllm_result_3d = torch.stack(vllm_results, dim=1)  # [batch, seq, vocab]

    # Compare results
    torch.testing.assert_close(
        our_result,
        vllm_result_3d,
        msg=f"Our apply_top_k_top_p doesn't match vLLM upstream ({test_name})",
    )
    print(f"✓ Results match for {test_name}")
