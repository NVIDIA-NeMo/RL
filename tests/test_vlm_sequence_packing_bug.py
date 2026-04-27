"""
Pytest for VLM sequence packing bug.

Tests that sequence packing with VLMs handles token expansion correctly.
Uses real production functions — only the model forward pass is mocked.

Run in the full nemo-rl environment (same as training jobs):

    HF_HOME=/lustre/fs1/portfolios/coreai/users/aroshanghias/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    NRL_IGNORE_VERSION_MISMATCH=1 \
    uv run python -m pytest tests/test_vlm_sequence_packing_bug.py -v -s
"""

import os
import sys
import pytest
import torch
import torch.distributed as dist
from unittest.mock import MagicMock

# The base venv (/opt/nemo_rl_venv) is missing transformer_engine (CUDA-compiled)
# and may have Megatron version mismatches (Bridge expects modules that LM doesn't
# have).  Two meta-path finders handle this:
#
#   1. _BlockedPkgFinder (inserted FIRST): intercepts ALL imports for packages
#      that are completely absent or broken (transformer_engine, megatron.bridge).
#      megatron.bridge is blocked because the installed Bridge and Megatron-LM
#      are version-mismatched (Bridge wants CudaGraphScope which LM lacks).
#      Returns MagicMock.
#
#   2. _FallbackFinder (appended LAST): catches any megatron.* submodule that
#      the real finders can't resolve (version mismatches).  Real modules load
#      normally; only genuinely missing ones get a MagicMock.
#
# Both are no-ops when running in a complete environment (uv run).

# Instructions for running this test:
# cd /lustre/fs1/portfolios/coreai/users/aroshanghias/nemo-rl && \
# HF_HOME=/lustre/fs1/portfolios/coreai/users/aroshanghias/.cache/huggingface \
# HF_HUB_OFFLINE=1 \
# NRL_IGNORE_VERSION_MISMATCH=1 \
# python -m pytest tests/test_vlm_sequence_packing_bug.py -v -s 2>&1

import importlib.abc
import importlib.machinery

class _MockLoader(importlib.abc.Loader):
    """Loader that creates a MagicMock module."""
    def create_module(self, spec):
        mod = MagicMock()
        mod.__name__ = spec.name
        mod.__path__ = []          # pretend it's a package so sub-imports work
        mod.__version__ = "1.0.0"
        mod.__spec__ = spec
        return mod
    def exec_module(self, module):
        pass

class _BlockedPkgFinder(importlib.abc.MetaPathFinder):
    """Intercept all imports for wholly-missing packages."""
    _BLOCKED = ("transformer_engine", "megatron.bridge")
    def find_spec(self, fullname, path, target=None):
        if any(fullname == p or fullname.startswith(p + ".") for p in self._BLOCKED):
            return importlib.machinery.ModuleSpec(fullname, _MockLoader())
        return None

class _FallbackFinder(importlib.abc.MetaPathFinder):
    """Last-resort finder: mock megatron.* modules missing due to version skew."""
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("megatron."):
            return importlib.machinery.ModuleSpec(fullname, _MockLoader())
        return None

if "transformer_engine" not in sys.modules or "megatron.bridge" not in sys.modules:
    sys.meta_path.insert(0, _BlockedPkgFinder())    # TE + Bridge: block everything
sys.meta_path.append(_FallbackFinder())              # megatron: fallback only

# Real production functions
from nemo_rl.models.megatron.multimodal import collapse_multimodal_tokens
from nemo_rl.models.megatron.common import (
    _pack_sequences_for_megatron,
    _vlm_sp_repad_collapsed,
)
from nemo_rl.distributed.model_utils import (
    from_parallel_logits_to_logprobs_packed_sequences,
    from_parallel_logits_to_logprobs,
    repack_original_tokens_for_vlm_logprobs,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module", autouse=True)
def init_dist():
    """Initialize single-process distributed environment (needed for logprob functions)."""
    if not dist.is_available():
        pytest.skip("PyTorch distributed not available")

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    yield


@pytest.fixture
def mock_model():
    """
    Mock model that replicates LLaVA's token expansion behavior.

    This is the ONLY mock in the test. Everything else uses real production code.

    Replicates two critical LLaVA behaviors:
    1. Token expansion: each collapsed <image> token → img_seq_len embeddings in output
    2. In-place mutation of packed_seq_params.cu_seqlens_*_padded[-1]
       (see llava_model.py _preprocess_data, lines ~999-1006)
    """
    class MockLLaVAModel:
        def __init__(self, img_seq_len=576, vocab_size=32000):
            # Token IDs — needed by the real collapse_multimodal_tokens
            self.img_start_token_id = 21
            self.img_end_token_id = 22
            self.image_token_id = 20

            self.img_seq_len = img_seq_len
            self.vocab_size = vocab_size

        def __call__(self, input_ids, packed_seq_params=None, **kwargs):
            batch_size, seq_len = input_ids.shape

            # Count collapsed <image> tokens PER SAMPLE.  For batched (non-packed)
            # input each sample expands independently; we use the max to get a
            # uniform output tensor.  For packed input (batch_size=1) max == only
            # element, so total expansion is computed correctly either way.
            per_sample_img_count = (input_ids == self.image_token_id).sum(dim=1)
            per_sample_expansion = per_sample_img_count * (self.img_seq_len - 1)
            expanded_len = (seq_len + per_sample_expansion).max().item()

            # CRITICAL: replicate LLaVA's in-place mutation of packed_seq_params.
            # In _pack_sequences_for_megatron, all four cu_seqlens fields point to
            # the SAME tensor, so mutating one mutates them all — including the
            # cu_seqlens_padded variable captured by the collection_fn closure.
            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    packed_seq_params.cu_seqlens_q_padded[-1] = expanded_len
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    packed_seq_params.cu_seqlens_kv_padded[-1] = expanded_len

            return torch.randn(batch_size, expanded_len, self.vocab_size)

    return MockLLaVAModel()


@pytest.fixture
def mock_data():
    """Create mock VLM data with expanded image tokens."""
    batch_size = 4
    seq_len_expanded = 1000
    num_image_tokens = 576

    input_ids_list = []
    input_lengths_list = []

    for i in range(batch_size):
        actual_len = seq_len_expanded - (i * 50)  # 1000, 950, 900, 850

        # Build: [text] <img> <image>×576 </img> [text] [padding]
        text_before = 100
        text_after = actual_len - text_before - num_image_tokens - 2

        if text_after < 0:
            text_before = actual_len - num_image_tokens - 2 - 50
            text_after = actual_len - text_before - num_image_tokens - 2

        seq = []
        seq.extend(torch.randint(1, 20, (text_before,)).tolist())
        seq.append(21)  # <img>
        seq.extend([20] * num_image_tokens)  # <image> repeated
        seq.append(22)  # </img>
        if text_after > 0:
            seq.extend(torch.randint(1, 20, (text_after,)).tolist())

        seq = seq + [0] * (seq_len_expanded - len(seq))
        input_ids_list.append(seq)
        input_lengths_list.append(actual_len)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths_list, dtype=torch.long),
        "pixel_values": torch.randn(batch_size, 3, 384, 384),
        "imgs_sizes": torch.tensor([[384, 384]] * batch_size, dtype=torch.int32),
    }


# ============================================================================
# Helper: run the OLD (buggy) SP pipeline — collapse → pack → forward → unpack
# ============================================================================

def _run_buggy_sp_pipeline(mock_data, mock_model):
    """Run the SP pipeline the OLD way (collapsed target + mixed cu_seqlens).

    Returns (logprobs [batch, seq], zero_fraction).
    """
    batch_size = mock_data["input_ids"].shape[0]
    input_seq_dim_size = mock_data["input_ids"].shape[1]

    data_collapsed = collapse_multimodal_tokens(mock_data.copy(), mock_model)
    collapsed_seq_len = data_collapsed["input_ids"].shape[1]

    (input_ids_packed, _, packed_seq_params,
     cu_seqlens, cu_seqlens_padded) = _pack_sequences_for_megatron(
        data_collapsed["input_ids"].clone(),
        data_collapsed["input_lengths"],
        pad_individual_seqs_to_multiple_of=1,
        pad_packed_seq_to_multiple_of=1,
        pad_packed_seq_to=None,
        cp_rank=0,
        cp_size=1,
    )

    output = mock_model(input_ids_packed, packed_seq_params=packed_seq_params)

    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
        output,
        target=input_ids_packed,
        cu_seqlens_padded=cu_seqlens_padded,
        unpacked_seqlen=collapsed_seq_len,
        vocab_start_index=0,
        vocab_end_index=output.shape[-1],
        group=dist.group.WORLD,
        inference_only=True,
    )

    token_logprobs = torch.cat(
        [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
    )
    padding_needed = input_seq_dim_size - token_logprobs.shape[1]
    if padding_needed > 0:
        token_logprobs = torch.nn.functional.pad(
            token_logprobs, (0, padding_needed), mode="constant", value=0.0
        )

    zero_frac = (token_logprobs == 0).float().mean().item()
    return token_logprobs, zero_frac


# ============================================================================
# Helper: run the FIXED SP pipeline — expand target + expanded cu_seqlens
# ============================================================================

def _run_fixed_sp_pipeline(mock_data, mock_model):
    """Run the SP pipeline with the Option A fix (expanded target + expanded cu_seqlens).

    Returns (logprobs [batch, seq], zero_fraction).
    """
    batch_size = mock_data["input_ids"].shape[0]
    input_seq_dim_size = mock_data["input_ids"].shape[1]

    original_input_ids = mock_data["input_ids"].clone()
    original_input_lengths = mock_data["input_lengths"].clone()

    data_collapsed = collapse_multimodal_tokens(mock_data.copy(), mock_model)
    tokens_removed_per_sample = data_collapsed.pop("tokens_removed_per_sample", None)
    assert tokens_removed_per_sample is not None, \
        "collapse_multimodal_tokens should return tokens_removed_per_sample"

    (input_ids_packed, _, packed_seq_params,
     cu_seqlens, cu_seqlens_padded) = _pack_sequences_for_megatron(
        data_collapsed["input_ids"].clone(),
        data_collapsed["input_lengths"],
        pad_individual_seqs_to_multiple_of=1,
        pad_packed_seq_to_multiple_of=1,
        pad_packed_seq_to=None,
        cp_rank=0,
        cp_size=1,
    )

    # Compute expanded cu_seqlens from collapsed + cumulative removed tokens
    n_seqs = cu_seqlens_padded.shape[0] - 1
    cumulative_removed = torch.zeros(
        n_seqs + 1, dtype=torch.int32, device=cu_seqlens_padded.device
    )
    cumulative_removed[1:] = tokens_removed_per_sample[:n_seqs].to(torch.int32).cumsum(0)
    cu_seqlens_padded_expanded = cu_seqlens_padded.clone() + cumulative_removed
    cu_seqlens_expanded = cu_seqlens.clone() + cumulative_removed

    # --- Phase 2: Pre-set expanded cu_seqlens on packed_seq_params ---
    # Single clone, all four fields alias it (matches original packing pattern).
    cu_seqlens_for_attn = cu_seqlens_padded_expanded.clone()
    packed_seq_params.cu_seqlens_q = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_kv = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_q_padded = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_kv_padded = cu_seqlens_for_attn

    expanded_slot_lengths = cu_seqlens_padded_expanded[1:] - cu_seqlens_padded_expanded[:-1]
    packed_seq_params.max_seqlen_q = expanded_slot_lengths.max().item()
    packed_seq_params.max_seqlen_kv = expanded_slot_lengths.max().item()

    # Model forward (LLaVA mutates cu_seqlens_*_padded[-1] in-place, but with
    # expanded values pre-set this is a no-op — overwriting with the same value)
    output = mock_model(input_ids_packed, packed_seq_params=packed_seq_params)

    # Repack original (uncollapsed) tokens into expanded packed format
    target_expanded = repack_original_tokens_for_vlm_logprobs(
        original_input_ids,
        original_input_lengths,
        cu_seqlens_padded_expanded,
        device=output.device,
    )

    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
        output,
        target=target_expanded,
        cu_seqlens_padded=cu_seqlens_padded_expanded,
        unpacked_seqlen=original_input_ids.shape[1],
        vocab_start_index=0,
        vocab_end_index=output.shape[-1],
        group=dist.group.WORLD,
        inference_only=True,
        cu_seqlens=cu_seqlens_expanded,
    )

    token_logprobs = torch.cat(
        [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
    )
    padding_needed = input_seq_dim_size - token_logprobs.shape[1]
    if padding_needed > 0:
        token_logprobs = torch.nn.functional.pad(
            token_logprobs, (0, padding_needed), mode="constant", value=0.0
        )

    zero_frac = (token_logprobs == 0).float().mean().item()
    return token_logprobs, zero_frac


# ============================================================================
# Helper: run non-SP pipeline (for reuse and to avoid test return-value warnings)
# ============================================================================

def _run_non_sp_pipeline(mock_data, mock_model):
    """Run VLM logprob extraction WITHOUT sequence packing. Returns (logprobs, zero_frac)."""
    batch_size = mock_data["input_ids"].shape[0]
    input_seq_dim_size = mock_data["input_ids"].shape[1]
    original_input_ids = mock_data["input_ids"].clone()

    data_collapsed = collapse_multimodal_tokens(mock_data.copy(), mock_model)
    output = mock_model(data_collapsed["input_ids"])
    token_logprobs = from_parallel_logits_to_logprobs(
        output,
        target=original_input_ids,
        vocab_start_index=0,
        vocab_end_index=output.shape[-1],
        tp_group=dist.group.WORLD,
        inference_only=True,
    )
    token_logprobs = torch.cat(
        [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
    )
    padding_needed = input_seq_dim_size - token_logprobs.shape[1]
    if padding_needed > 0:
        token_logprobs = torch.nn.functional.pad(
            token_logprobs, (0, padding_needed), mode="constant", value=0.0
        )
    zero_frac = (token_logprobs == 0).float().mean().item()
    return token_logprobs, zero_frac


# ============================================================================
# Bug-demonstration tests (old pipeline — these document the bug)
# ============================================================================

def test_vlm_without_sequence_packing(mock_data, mock_model):
    """Test VLM logprob extraction WITHOUT sequence packing (baseline)."""
    batch_size = mock_data["input_ids"].shape[0]
    input_seq_dim_size = mock_data["input_ids"].shape[1]

    data_collapsed = collapse_multimodal_tokens(mock_data.copy(), mock_model)
    assert data_collapsed["input_ids"].shape[1] < input_seq_dim_size, \
        "Collapse should reduce sequence length"

    token_logprobs, zero_frac = _run_non_sp_pipeline(mock_data, mock_model)
    assert token_logprobs.shape == (batch_size, input_seq_dim_size), \
        f"Final shape mismatch: {token_logprobs.shape}"
    assert zero_frac < 0.15, f"Without SP should have <15% zeros, got {zero_frac:.1%}"


def test_vlm_with_sequence_packing_buggy(mock_data, mock_model):
    """
    Test VLM logprob extraction WITH sequence packing using the OLD (buggy) pipeline.

    This test demonstrates the bug:
    1. Tokens are collapsed, then packed with collapsed cu_seqlens boundaries
    2. Model forward expands tokens AND mutates cu_seqlens_padded[-1] in-place
    3. Unpacker receives expanded output + mutated boundaries + collapsed target
    4. torch.gather operates on misaligned dimensions → garbled logprobs
    """
    batch_size = mock_data["input_ids"].shape[0]
    input_seq_dim_size = mock_data["input_ids"].shape[1]

    # Step 1: Collapse using REAL function
    data_collapsed = collapse_multimodal_tokens(mock_data.copy(), mock_model)
    collapsed_seq_len = data_collapsed["input_ids"].shape[1]

    # Step 2: Pack sequences using REAL function
    (input_ids_packed, input_ids_cp_sharded, packed_seq_params,
     cu_seqlens, cu_seqlens_padded) = _pack_sequences_for_megatron(
        data_collapsed["input_ids"].clone(),
        data_collapsed["input_lengths"],
        pad_individual_seqs_to_multiple_of=1,
        pad_packed_seq_to_multiple_of=1,
        pad_packed_seq_to=None,
        cp_rank=0,
        cp_size=1,
    )

    # Verify all cu_seqlens in PackedSeqParams point to the same tensor.
    # This is what makes LLaVA's mutation of cu_seqlens_q_padded[-1] propagate
    # to the cu_seqlens_padded variable used by the unpacker.
    assert packed_seq_params is not None
    assert packed_seq_params.cu_seqlens_q is cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_kv is cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_q_padded is cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_kv_padded is cu_seqlens_padded

    cu_seqlens_before = cu_seqlens_padded.clone()

    # Step 3: Model forward — expands tokens AND mutates cu_seqlens_padded[-1]
    output = mock_model(input_ids_packed, packed_seq_params=packed_seq_params)

    assert cu_seqlens_padded[-1] > cu_seqlens_before[-1], \
        "Model should mutate cu_seqlens_padded[-1] to reflect expansion"

    # Step 4: Unpack using REAL function — THIS IS WHERE THE BUG MANIFESTS
    # cu_seqlens_padded has been mutated: last element is expanded, intermediates are collapsed
    # target (input_ids_packed) is still in collapsed space
    # output has expanded sequence length
    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
        output,
        target=input_ids_packed,
        cu_seqlens_padded=cu_seqlens_padded,
        unpacked_seqlen=collapsed_seq_len,
        vocab_start_index=0,
        vocab_end_index=output.shape[-1],
        group=dist.group.WORLD,
        inference_only=True,
    )

    # Step 5: Post-process
    token_logprobs = torch.cat(
        [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
    )
    padding_needed = input_seq_dim_size - token_logprobs.shape[1]
    if padding_needed > 0:
        token_logprobs = torch.nn.functional.pad(
            token_logprobs, (0, padding_needed), mode="constant", value=0.0
        )

    assert token_logprobs.shape == (batch_size, input_seq_dim_size)

    zero_frac = (token_logprobs == 0).float().mean().item()
    assert zero_frac > 0.5, f"Buggy pipeline should produce >50%% zeros, got {zero_frac:.1%}"


@pytest.mark.xfail(
    reason="Bug demonstration: old SP pipeline produces garbled logprobs "
           "(expanded output + collapsed target + mixed cu_seqlens)",
    strict=True,
)
def test_vlm_sequence_packing_bug_detection(mock_data, mock_model):
    """
    Compare buggy SP vs non-SP paths to detect the bug.

    Without SP: model output aligns with expanded target → correct logprobs, minimal zeros.
    With SP (buggy): expanded output + collapsed target + mutated boundaries → garbled logprobs,
    excessive zeros from misaligned slicing.

    This test is expected to FAIL (xfail) because the old pipeline is still buggy.
    """
    logprobs_no_sp, zero_frac_no_sp = _run_non_sp_pipeline(mock_data, mock_model)
    logprobs_with_sp, zero_frac_sp = _run_buggy_sp_pipeline(mock_data, mock_model)

    assert logprobs_no_sp.shape == logprobs_with_sp.shape

    print(f"\nZero fraction comparison (buggy pipeline):")
    print(f"  Without SP: {zero_frac_no_sp:.1%}")
    print(f"  With SP:    {zero_frac_sp:.1%}")
    print(f"  Difference: {zero_frac_sp - zero_frac_no_sp:.1%}")

    # The bug causes the SP path to have far more zeros than the non-SP path.
    # This assertion SHOULD FAIL, demonstrating the bug exists.
    assert zero_frac_sp - zero_frac_no_sp < 0.1, (
        f"SP path has {(zero_frac_sp - zero_frac_no_sp):.1%} more zeros than non-SP path — "
        f"indicates wrong slicing due to expanded cu_seqlens with collapsed target"
    )


@pytest.mark.parametrize("num_images_per_sample", [1, 2, 3])
def test_vlm_sp_buggy_severity_scales_with_images(mock_model, num_images_per_sample):
    """Bug severity scales with number of images per sample (old pipeline)."""
    batch_size = 2
    num_image_tokens = 576

    input_ids_list = []
    input_lengths_list = []

    for b in range(batch_size):
        seq = [5]  # Start token
        for _ in range(num_images_per_sample):
            seq.extend([10] * 50)  # Text between images
            seq.append(21)  # <img>
            seq.extend([20] * num_image_tokens)  # <image> repeated
            seq.append(22)  # </img>
        seq.extend([10] * (50 + b * 30))  # Trailing text (vary length per sample)
        input_ids_list.append(seq)
        input_lengths_list.append(len(seq))

    max_content_len = max(input_lengths_list)
    seq_len = max_content_len + 20

    for i in range(batch_size):
        input_ids_list[i] = input_ids_list[i] + [0] * (seq_len - len(input_ids_list[i]))

    data_dict = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths_list, dtype=torch.long),
        "pixel_values": torch.randn(batch_size, 3, 384, 384),
        "imgs_sizes": torch.tensor([[384, 384]] * batch_size, dtype=torch.int32),
    }

    _, zero_frac = _run_buggy_sp_pipeline(data_dict, mock_model)
    print(f"[BUGGY] Images per sample: {num_images_per_sample}, Zero fraction: {zero_frac:.1%}")

    # More images → more expansion → worse misalignment → more zeros
    assert zero_frac > 0.3, (
        f"Expected bug to cause >30% zeros "
        f"with {num_images_per_sample} images, got {zero_frac:.1%}"
    )


# ============================================================================
# Fix-verification tests (new pipeline — Option A)
# ============================================================================

def test_vlm_sp_fix_matches_non_sp(mock_data, mock_model):
    """
    With the fix, the SP path should produce similar zero fractions to the non-SP path.

    The fix aligns all three inputs to expanded space before calling the unpacker:
    1. Repack original (uncollapsed) tokens as target
    2. Compute expanded cu_seqlens from collapsed cu_seqlens + tokens_removed_per_sample
    3. Model output is already in expanded space
    """
    logprobs_no_sp, zero_frac_no_sp = _run_non_sp_pipeline(mock_data, mock_model)
    logprobs_fixed_sp, zero_frac_sp = _run_fixed_sp_pipeline(mock_data, mock_model)

    assert logprobs_no_sp.shape == logprobs_fixed_sp.shape

    print(f"\nZero fraction comparison (fixed pipeline):")
    print(f"  Without SP: {zero_frac_no_sp:.1%}")
    print(f"  With SP:    {zero_frac_sp:.1%}")
    print(f"  Difference: {zero_frac_sp - zero_frac_no_sp:.1%}")

    assert zero_frac_sp - zero_frac_no_sp < 0.1, (
        f"SP path has {(zero_frac_sp - zero_frac_no_sp):.1%} more zeros than non-SP path — "
        f"fix may not be working correctly"
    )


def test_vlm_sp_fix_alignment(mock_data, mock_model):
    """Verify that the fix produces correctly aligned tensors."""
    original_input_ids = mock_data["input_ids"].clone()
    original_input_lengths = mock_data["input_lengths"].clone()

    data_collapsed = collapse_multimodal_tokens(mock_data.copy(), mock_model)
    tokens_removed_per_sample = data_collapsed.pop("tokens_removed_per_sample", None)
    assert tokens_removed_per_sample is not None

    (input_ids_packed, _, packed_seq_params,
     cu_seqlens, cu_seqlens_padded) = _pack_sequences_for_megatron(
        data_collapsed["input_ids"].clone(),
        data_collapsed["input_lengths"],
        pad_individual_seqs_to_multiple_of=1,
        pad_packed_seq_to_multiple_of=1,
        pad_packed_seq_to=None,
        cp_rank=0,
        cp_size=1,
    )

    # Compute expanded cu_seqlens
    n_seqs = cu_seqlens_padded.shape[0] - 1
    cumulative_removed = torch.zeros(
        n_seqs + 1, dtype=torch.int32, device=cu_seqlens_padded.device
    )
    cumulative_removed[1:] = tokens_removed_per_sample[:n_seqs].to(torch.int32).cumsum(0)
    cu_seqlens_padded_expanded = cu_seqlens_padded.clone() + cumulative_removed
    cu_seqlens_expanded = cu_seqlens.clone() + cumulative_removed

    # Phase 2: Pre-set expanded cu_seqlens on packed_seq_params
    cu_seqlens_for_attn = cu_seqlens_padded_expanded.clone()
    packed_seq_params.cu_seqlens_q = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_kv = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_q_padded = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_kv_padded = cu_seqlens_for_attn
    expanded_slot_lengths = cu_seqlens_padded_expanded[1:] - cu_seqlens_padded_expanded[:-1]
    packed_seq_params.max_seqlen_q = expanded_slot_lengths.max().item()
    packed_seq_params.max_seqlen_kv = expanded_slot_lengths.max().item()

    # Model forward
    output = mock_model(input_ids_packed, packed_seq_params=packed_seq_params)

    # Repack target
    target_expanded = repack_original_tokens_for_vlm_logprobs(
        original_input_ids,
        original_input_lengths,
        cu_seqlens_padded_expanded,
        device=output.device,
    )

    # All three should now be in the same coordinate system
    assert output.shape[1] == target_expanded.shape[1], \
        f"Logits and target must match: {output.shape[1]} vs {target_expanded.shape[1]}"
    assert cu_seqlens_padded_expanded[-1].item() == output.shape[1], \
        f"cu_seqlens_padded_expanded[-1] must match output: {cu_seqlens_padded_expanded[-1]} vs {output.shape[1]}"

    # Verify each sample: expanded_slot = collapsed_slot + tokens_removed (same coordinate formula)
    # cu_seqlens_padded is unchanged here because we replaced packed_seq_params before model forward
    for i in range(n_seqs):
        expanded_slot_len = (
            cu_seqlens_padded_expanded[i + 1] - cu_seqlens_padded_expanded[i]
        ).item()
        collapsed_slot_len = (cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]).item()
        removed = tokens_removed_per_sample[i].item()
        assert expanded_slot_len - removed == collapsed_slot_len, (
            f"Sample {i}: expanded_slot - tokens_removed ({expanded_slot_len} - {removed}) "
            f"must equal collapsed_slot ({collapsed_slot_len})"
        )


def test_vlm_sp_phase2_attention_masking(mock_data, mock_model):
    """
    Verify Phase 2: expanded cu_seqlens are set on packed_seq_params BEFORE
    model forward, preserving aliasing so attention would see correct boundaries.

    Note: The mock model does not use packed attention (it only mutates [-1]).
    This test validates the parameter setup; real Layer 1 behavior is seen on cluster.
    """
    data_collapsed = collapse_multimodal_tokens(mock_data.copy(), mock_model)
    tokens_removed_per_sample = data_collapsed.pop("tokens_removed_per_sample", None)
    assert tokens_removed_per_sample is not None

    (input_ids_packed, _, packed_seq_params,
     cu_seqlens, cu_seqlens_padded) = _pack_sequences_for_megatron(
        data_collapsed["input_ids"].clone(),
        data_collapsed["input_lengths"],
        pad_individual_seqs_to_multiple_of=1,
        pad_packed_seq_to_multiple_of=1,
        pad_packed_seq_to=None,
        cp_rank=0,
        cp_size=1,
    )

    # Before Phase 2: all four cu_seqlens fields are aliased to same tensor
    assert packed_seq_params.cu_seqlens_q is cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_kv is cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_q_padded is cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_kv_padded is cu_seqlens_padded

    # Compute expanded cu_seqlens
    n_seqs = cu_seqlens_padded.shape[0] - 1
    cumulative_removed = torch.zeros(
        n_seqs + 1, dtype=torch.int32, device=cu_seqlens_padded.device
    )
    cumulative_removed[1:] = tokens_removed_per_sample[:n_seqs].to(torch.int32).cumsum(0)
    cu_seqlens_padded_expanded = cu_seqlens_padded.clone() + cumulative_removed
    cu_seqlens_expanded = cu_seqlens.clone() + cumulative_removed

    # Apply Phase 2: single clone, all four fields alias it (matches TE expectations)
    cu_seqlens_for_attn = cu_seqlens_padded_expanded.clone()
    packed_seq_params.cu_seqlens_q = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_kv = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_q_padded = cu_seqlens_for_attn
    packed_seq_params.cu_seqlens_kv_padded = cu_seqlens_for_attn

    expanded_slot_lengths = cu_seqlens_padded_expanded[1:] - cu_seqlens_padded_expanded[:-1]
    packed_seq_params.max_seqlen_q = expanded_slot_lengths.max().item()
    packed_seq_params.max_seqlen_kv = expanded_slot_lengths.max().item()

    # After Phase 2: all four fields still alias the SAME tensor (required by TE)
    assert packed_seq_params.cu_seqlens_q is packed_seq_params.cu_seqlens_kv
    assert packed_seq_params.cu_seqlens_q is packed_seq_params.cu_seqlens_q_padded
    assert packed_seq_params.cu_seqlens_q_padded is packed_seq_params.cu_seqlens_kv_padded

    # But they are a DIFFERENT tensor from the original cu_seqlens_padded
    assert packed_seq_params.cu_seqlens_q is not cu_seqlens_padded

    # All four must have expanded values
    assert torch.equal(packed_seq_params.cu_seqlens_q_padded, cu_seqlens_padded_expanded)

    # max_seqlen should reflect the largest expanded individual sequence
    assert packed_seq_params.max_seqlen_q == expanded_slot_lengths.max().item()
    assert packed_seq_params.max_seqlen_kv == expanded_slot_lengths.max().item()

    # Snapshot before model forward
    cu_seqlens_before = packed_seq_params.cu_seqlens_q_padded.clone()

    # Model forward: LLaVA mutates cu_seqlens_*_padded[-1]
    output = mock_model(input_ids_packed, packed_seq_params=packed_seq_params)

    # With expanded boundaries, the mutation should be a no-op:
    # LLaVA sets [-1] = actual_seq_len = T_exp, which equals our expanded [-1]
    assert packed_seq_params.cu_seqlens_q_padded[-1].item() == cu_seqlens_padded_expanded[-1].item(), \
        "LLaVA's [-1] mutation should be no-op with expanded cu_seqlens pre-set"

    # Closure-captured cu_seqlens_padded_expanded is a SEPARATE tensor,
    # safe from any in-place mutation
    assert torch.equal(cu_seqlens_padded_expanded, cu_seqlens_before), \
        "cu_seqlens_padded_expanded must be unaffected by model's in-place mutation"

    # Verify the expanded total matches the model output size
    assert output.shape[1] == cu_seqlens_padded_expanded[-1].item(), \
        f"Model output {output.shape[1]} must match expanded total {cu_seqlens_padded_expanded[-1].item()}"

    print(f"\nPhase 2 verification:")
    print(f"  Sequences packed: {n_seqs}")
    print(f"  Collapsed total: {cu_seqlens_padded[-1].item()}")
    print(f"  Expanded total:  {cu_seqlens_padded_expanded[-1].item()}")
    print(f"  Max expanded seqlen: {packed_seq_params.max_seqlen_q}")
    print(f"  Model output seq dim: {output.shape[1]}")
    print(f"  Aliasing preserved (all four same tensor): True")
    print(f"  [-1] mutation was no-op: True")


@pytest.mark.parametrize("num_images_per_sample", [1, 2, 3])
def test_vlm_sp_fix_with_multiple_images(mock_model, num_images_per_sample):
    """With the fix, zero fraction should be low regardless of image count."""
    batch_size = 2
    num_image_tokens = 576

    # Build sequences first to determine actual content length, then pad tightly
    input_ids_list = []
    input_lengths_list = []

    for b in range(batch_size):
        seq = [5]  # Start token
        for _ in range(num_images_per_sample):
            seq.extend([10] * 50)  # Text between images
            seq.append(21)  # <img>
            seq.extend([20] * num_image_tokens)  # <image> repeated
            seq.append(22)  # </img>
        seq.extend([10] * (50 + b * 30))  # Trailing text (vary length per sample)
        input_ids_list.append(seq)
        input_lengths_list.append(len(seq))

    # Pad to max content length + small buffer (not a huge fixed size)
    max_content_len = max(input_lengths_list)
    seq_len = max_content_len + 20

    for i in range(batch_size):
        input_ids_list[i] = input_ids_list[i] + [0] * (seq_len - len(input_ids_list[i]))

    data_dict = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths_list, dtype=torch.long),
        "pixel_values": torch.randn(batch_size, 3, 384, 384),
        "imgs_sizes": torch.tensor([[384, 384]] * batch_size, dtype=torch.int32),
    }

    _, zero_frac = _run_fixed_sp_pipeline(data_dict, mock_model)
    print(f"[FIXED] Images per sample: {num_images_per_sample}, Zero fraction: {zero_frac:.1%}")

    assert zero_frac < 0.15, (
        f"Expected <15% zeros with fix applied, got {zero_frac:.1%} "
        f"with {num_images_per_sample} images per sample"
    )


# ============================================================================
# CP > 1 + VLM packing tests (expanded-aligned padding)
# ============================================================================

class TestVLMPackingCPAlignment:
    """Tests for _pack_sequences_for_megatron with VLM + CP > 1.

    Verifies that expanded slot lengths (collapsed_padded + tokens_removed) are
    multiples of cp_size * 2, so that CP sharding in the logprob path does not
    produce tensor-size mismatches.
    """

    @staticmethod
    def _make_batch(seq_lens, tokens_removed_list):
        """Create (input_ids, seq_lengths, tokens_removed_per_sample) tensors."""
        batch_size = len(seq_lens)
        max_seq_len = max(seq_lens)
        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        for b in range(batch_size):
            input_ids[b, :seq_lens[b]] = torch.randint(1, 100, (seq_lens[b],))
        seq_lengths = torch.tensor(seq_lens, dtype=torch.long)
        tokens_removed = torch.tensor(tokens_removed_list, dtype=torch.long)
        return input_ids, seq_lengths, tokens_removed

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_expanded_slots_are_cp_aligned(self, cp_size):
        """Every expanded slot (collapsed_padded + removed) must be a multiple of cp_size * 2."""
        pad_factor = cp_size * 2
        seq_lens = [100, 200, 150, 300]
        tokens_removed = [279, 0, 575, 123]  # Mix of odd, zero, large
        input_ids, seq_lengths_t, tokens_removed_t = self._make_batch(seq_lens, tokens_removed)

        (all_ids, cp_ids, packed_params, cu_seqlens, cu_seqlens_padded) = (
            _pack_sequences_for_megatron(
                input_ids,
                seq_lengths_t,
                pad_individual_seqs_to_multiple_of=pad_factor,
                pad_packed_seq_to_multiple_of=1,
                pad_packed_seq_to=None,
                cp_rank=0,
                cp_size=cp_size,
                tokens_removed_per_sample=tokens_removed_t,
            )
        )

        for b in range(len(seq_lens)):
            collapsed_padded = (cu_seqlens_padded[b + 1] - cu_seqlens_padded[b]).item()
            expanded_slot = collapsed_padded + tokens_removed[b]
            assert expanded_slot % pad_factor == 0, (
                f"Seq {b}: expanded_slot={expanded_slot} not aligned to {pad_factor}"
            )

    def test_vlm_skips_cp_sharding(self):
        """With VLM + CP > 1, packed_input_ids should equal all_input_ids (no CP sharding)."""
        cp_size = 2
        pad_factor = cp_size * 2
        seq_lens = [100, 200]
        tokens_removed = [279, 575]
        input_ids, seq_lengths_t, tokens_removed_t = self._make_batch(seq_lens, tokens_removed)

        (all_ids, cp_ids, _, _, _) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths_t,
            pad_individual_seqs_to_multiple_of=pad_factor,
            cp_rank=0,
            cp_size=cp_size,
            tokens_removed_per_sample=tokens_removed_t,
        )

        # For VLM, packed_input_ids (cp_ids) should be same as all_input_ids (no sharding)
        assert torch.equal(all_ids, cp_ids), (
            f"VLM + CP: packed_input_ids should equal all_input_ids (no CP sharding). "
            f"Shapes: all={all_ids.shape}, cp={cp_ids.shape}"
        )

    def test_non_vlm_still_cp_shards(self):
        """Without tokens_removed_per_sample, CP sharding should still happen."""
        cp_size = 2
        pad_factor = cp_size * 2
        seq_lens = [100, 200]
        input_ids, seq_lengths_t, _ = self._make_batch(seq_lens, [0, 0])

        (all_ids, cp_ids, _, _, _) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths_t,
            pad_individual_seqs_to_multiple_of=pad_factor,
            cp_rank=0,
            cp_size=cp_size,
            tokens_removed_per_sample=None,  # Non-VLM
        )

        # Non-VLM: packed_input_ids should be CP-sharded (shorter)
        assert cp_ids.shape[1] < all_ids.shape[1], (
            f"Non-VLM: cp_ids should be smaller than all_ids. "
            f"Shapes: all={all_ids.shape}, cp={cp_ids.shape}"
        )

    def test_llava_text_only_can_skip_local_cp_sharding(self):
        """LLaVA text-only microbatches should keep full packed ids for model-side CP."""
        cp_size = 2
        pad_factor = cp_size * 2
        seq_lens = [100, 200]
        input_ids, seq_lengths_t, _ = self._make_batch(seq_lens, [0, 0])

        (all_ids, cp_ids, _, _, _) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths_t,
            pad_individual_seqs_to_multiple_of=pad_factor,
            cp_rank=0,
            cp_size=cp_size,
            tokens_removed_per_sample=None,
            skip_local_cp_sharding=True,
        )

        assert torch.equal(all_ids, cp_ids), (
            "LLaVA text-only batches should preserve full packed ids when the "
            "model owns CP sharding."
        )

    def test_mixed_vlm_text_samples(self):
        """Samples with tokens_removed=0 should get standard padding (expanded==collapsed)."""
        cp_size = 2
        pad_factor = cp_size * 2
        seq_lens = [100, 200, 150]
        tokens_removed = [279, 0, 575]  # Sample 1 is text-only
        input_ids, seq_lengths_t, tokens_removed_t = self._make_batch(seq_lens, tokens_removed)

        (_, _, _, cu_seqlens, cu_seqlens_padded) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths_t,
            pad_individual_seqs_to_multiple_of=pad_factor,
            cp_rank=0,
            cp_size=cp_size,
            tokens_removed_per_sample=tokens_removed_t,
        )

        for b in range(len(seq_lens)):
            collapsed_padded = (cu_seqlens_padded[b + 1] - cu_seqlens_padded[b]).item()
            expanded_slot = collapsed_padded + tokens_removed[b]
            assert expanded_slot % pad_factor == 0, (
                f"Seq {b}: expanded_slot={expanded_slot} not aligned to {pad_factor}"
            )
            # Text-only sample (removed=0): collapsed_padded should also be aligned
            if tokens_removed[b] == 0:
                assert collapsed_padded % pad_factor == 0, (
                    f"Seq {b} (text-only): collapsed_padded={collapsed_padded} should be aligned"
                )

    def test_pp_vlm_cp_raises(self):
        """PP > 1 + VLM + CP > 1 should raise NotImplementedError."""
        cp_size = 2
        pad_factor = cp_size * 2
        seq_lens = [100, 200]
        tokens_removed = [279, 575]
        input_ids, seq_lengths_t, tokens_removed_t = self._make_batch(seq_lens, tokens_removed)

        with pytest.raises(NotImplementedError, match="PP > 1 with VLM"):
            _pack_sequences_for_megatron(
                input_ids,
                seq_lengths_t,
                pad_individual_seqs_to_multiple_of=pad_factor,
                pad_packed_seq_to=1000,  # PP > 1 sets this
                cp_rank=0,
                cp_size=cp_size,
                tokens_removed_per_sample=tokens_removed_t,
            )

    def test_fp8_total_alignment_vlm(self):
        """With FP8 + VLM, the expanded total should be aligned to pad_packed_seq_to_multiple_of."""
        cp_size = 2
        pad_factor = cp_size * 2
        fp8_divisor = 16  # Typical non-blockwise FP8
        pad_packed_seq_to_multiple_of = fp8_divisor * pad_factor  # 16 * 4 = 64

        seq_lens = [100, 200]
        tokens_removed = [279, 575]
        input_ids, seq_lengths_t, tokens_removed_t = self._make_batch(seq_lens, tokens_removed)

        (_, _, _, _, cu_seqlens_padded) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths_t,
            pad_individual_seqs_to_multiple_of=pad_factor,
            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
            cp_rank=0,
            cp_size=cp_size,
            tokens_removed_per_sample=tokens_removed_t,
        )

        # Compute expanded total
        total_collapsed = cu_seqlens_padded[-1].item()
        total_removed = sum(tokens_removed)
        total_expanded = total_collapsed + total_removed

        assert total_expanded % pad_packed_seq_to_multiple_of == 0, (
            f"Expanded total={total_expanded} not aligned to {pad_packed_seq_to_multiple_of}"
        )

    def test_cp1_is_noop(self):
        """With CP=1 (pad_factor=1), VLM packing should behave identically to non-VLM."""
        seq_lens = [100, 200]
        tokens_removed = [279, 575]
        input_ids, seq_lengths_t, tokens_removed_t = self._make_batch(seq_lens, tokens_removed)

        # With VLM
        (_, _, _, _, cu_padded_vlm) = _pack_sequences_for_megatron(
            input_ids.clone(),
            seq_lengths_t,
            pad_individual_seqs_to_multiple_of=1,
            cp_rank=0,
            cp_size=1,
            tokens_removed_per_sample=tokens_removed_t,
        )

        # Without VLM
        (_, _, _, _, cu_padded_no_vlm) = _pack_sequences_for_megatron(
            input_ids.clone(),
            seq_lengths_t,
            pad_individual_seqs_to_multiple_of=1,
            cp_rank=0,
            cp_size=1,
            tokens_removed_per_sample=None,
        )

        assert torch.equal(cu_padded_vlm, cu_padded_no_vlm), (
            f"CP=1: VLM and non-VLM packing should be identical. "
            f"VLM cu_seqlens_padded={cu_padded_vlm.tolist()}, "
            f"non-VLM cu_seqlens_padded={cu_padded_no_vlm.tolist()}"
        )

    def test_input_validation_batch_size(self):
        """tokens_removed_per_sample must have at least batch_size entries."""
        seq_lens = [100, 200, 150]
        input_ids, seq_lengths_t, _ = self._make_batch(seq_lens, [0, 0, 0])
        tokens_removed_short = torch.tensor([279, 575], dtype=torch.long)  # Only 2 for 3 seqs

        with pytest.raises(AssertionError, match="tokens_removed_per_sample has 2"):
            _pack_sequences_for_megatron(
                input_ids,
                seq_lengths_t,
                pad_individual_seqs_to_multiple_of=4,
                cp_rank=0,
                cp_size=2,
                tokens_removed_per_sample=tokens_removed_short,
            )

    @pytest.mark.parametrize(
        "seq_lens,tokens_removed",
        [
            ([1], [1]),  # Minimal: single token, single removed
            ([7], [3]),  # Prime seq_len + odd removed
            ([1, 1], [0, 0]),  # All-text minimal batch
            ([500, 300, 700], [575, 575, 575]),  # Uniform large removal
        ],
    )
    def test_edge_cases_cp2(self, seq_lens, tokens_removed):
        """Various edge cases should still produce CP-aligned expanded slots."""
        cp_size = 2
        pad_factor = cp_size * 2
        input_ids, seq_lengths_t, tokens_removed_t = self._make_batch(seq_lens, tokens_removed)

        (_, _, _, _, cu_seqlens_padded) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths_t,
            pad_individual_seqs_to_multiple_of=pad_factor,
            cp_rank=0,
            cp_size=cp_size,
            tokens_removed_per_sample=tokens_removed_t,
        )

        for b in range(len(seq_lens)):
            collapsed_padded = (cu_seqlens_padded[b + 1] - cu_seqlens_padded[b]).item()
            expanded_slot = collapsed_padded + tokens_removed[b]
            assert expanded_slot % pad_factor == 0, (
                f"Seq {b}: expanded_slot={expanded_slot} (collapsed={collapsed_padded}, "
                f"removed={tokens_removed[b]}) not aligned to {pad_factor}"
            )
            # Collapsed padded must be >= actual seq len (no negative padding)
            assert collapsed_padded >= seq_lens[b], (
                f"Seq {b}: collapsed_padded={collapsed_padded} < seq_len={seq_lens[b]}"
            )


class _StaticResolutionMismatchLLaVA:
    """Minimal static-resolution model stub for real collapse/packing tests."""

    def __init__(self, img_seq_len=256):
        self._dynamic_resolution = False
        self.img_seq_len = img_seq_len
        self.img_start_token_id = 21
        self.img_end_token_id = 22


class _StaticResolutionMismatchModel:
    """Wrap LLaVA-style attrs under llava_model like the real runtime."""

    def __init__(self, img_seq_len=256):
        self.llava_model = _StaticResolutionMismatchLLaVA(img_seq_len=img_seq_len)


class _DynamicResolutionMismatchLLaVA:
    """Minimal dynamic-resolution model stub for non-regression coverage."""

    def __init__(self):
        self._dynamic_resolution = True
        self.img_start_token_id = 21
        self.img_end_token_id = 22


class _DynamicResolutionMismatchModel:
    """Wrap dynamic-resolution attrs under llava_model like the real runtime."""

    def __init__(self):
        self.llava_model = _DynamicResolutionMismatchLLaVA()


def _make_static_resolution_mismatch_batch(
    num_images_per_sample=None,
    image_hw=(448, 640),
):
    """Create a batch where physical removal != static img_seq_len expansion."""
    if num_images_per_sample is None:
        num_images_per_sample = [1, 2]

    img_start_id = 21
    image_token_id = 20
    img_end_id = 22
    num_image_tokens = 280
    text_before = 10
    text_after = 8

    sequences = []
    input_lengths = []

    for sample_idx, num_images in enumerate(num_images_per_sample):
        seq = list(range(100 + sample_idx * 10, 100 + sample_idx * 10 + text_before))
        for _ in range(num_images):
            seq.append(img_start_id)
            seq.extend([image_token_id] * num_image_tokens)
            seq.append(img_end_id)
        seq.extend(range(200 + sample_idx * 10, 200 + sample_idx * 10 + text_after))
        sequences.append(seq)
        input_lengths.append(len(seq))

    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [0] * (max_len - len(seq)) for seq in sequences]
    total_images = sum(num_images_per_sample)

    return {
        "input_ids": torch.tensor(padded, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "pixel_values": torch.randn(total_images, 3, image_hw[0], image_hw[1]),
        "imgs_sizes": torch.tensor([list(image_hw)] * total_images, dtype=torch.int32),
    }


def _expanded_boundaries_from(cu_seqlens_padded: torch.Tensor, expansion: torch.Tensor) -> torch.Tensor:
    cumulative = torch.zeros(
        expansion.shape[0] + 1, dtype=torch.int32, device=cu_seqlens_padded.device
    )
    cumulative[1:] = expansion.to(torch.int32).cumsum(0)
    return cu_seqlens_padded.clone() + cumulative


def test_static_resolution_mismatch_propagates_into_packed_boundaries():
    """Current code leaks physical removal count into expansion-space metadata for static-res VLMs."""
    model = _StaticResolutionMismatchModel(img_seq_len=256)
    data_dict = _make_static_resolution_mismatch_batch()
    collapsed = collapse_multimodal_tokens(data_dict, model)

    reported = collapsed["tokens_removed_per_sample"]
    expected_static_expansion = torch.tensor([255, 510], dtype=torch.long)

    # This is the core MR9 regression: current branch stores physical removals
    # [279, 558] instead of static img_seq_len-based expansion [255, 510].
    assert torch.equal(reported.cpu(), expected_static_expansion), (
        "Static-resolution VLM should report actual expansion-space deltas, "
        f"but collapse_multimodal_tokens returned {reported.tolist()} instead of "
        f"{expected_static_expansion.tolist()}."
    )

    (_, _, _, _, cu_seqlens_padded) = _pack_sequences_for_megatron(
        collapsed["input_ids"],
        collapsed["input_lengths"],
        pad_individual_seqs_to_multiple_of=4,
        cp_rank=0,
        cp_size=2,
        tokens_removed_per_sample=reported,
    )

    expanded_from_reported = _expanded_boundaries_from(cu_seqlens_padded, reported)
    expanded_from_static = _expanded_boundaries_from(
        cu_seqlens_padded, expected_static_expansion.to(device=reported.device)
    )

    assert torch.equal(expanded_from_reported, expanded_from_static), (
        "Static-resolution packed boundaries should be computed from fixed img_seq_len "
        f"expansion. Got reported boundaries {expanded_from_reported.tolist()} vs "
        f"expected {expanded_from_static.tolist()}."
    )


def test_dynamic_resolution_keeps_physical_removal_count():
    """Dynamic-resolution models should keep reporting physical removals."""
    model = _DynamicResolutionMismatchModel()
    data_dict = _make_static_resolution_mismatch_batch()
    collapsed = collapse_multimodal_tokens(data_dict, model)

    reported = collapsed["tokens_removed_per_sample"]
    expected_physical_removal = torch.tensor([279, 558], dtype=torch.long)

    assert torch.equal(reported.cpu(), expected_physical_removal), (
        "Dynamic-resolution VLM should keep the physical removal count in "
        f"tokens_removed_per_sample, but got {reported.tolist()} instead of "
        f"{expected_physical_removal.tolist()}."
    )


def test_static_resolution_sp_repad_uses_corrected_expansion():
    """SP repad width should follow the corrected static-resolution expansion delta."""
    model = _StaticResolutionMismatchModel(img_seq_len=256)
    data_dict = _make_static_resolution_mismatch_batch()
    collapsed = collapse_multimodal_tokens(data_dict, model)

    reported = collapsed["tokens_removed_per_sample"]
    repadded = _vlm_sp_repad_collapsed(
        collapsed["input_ids"],
        reported,
        tp_size=32,
    )

    collapsed_width = collapsed["input_ids"].shape[1]
    expected_width = ((collapsed_width + 510 + 31) // 32) * 32 - 510

    assert repadded.shape[1] == expected_width, (
        "Static-resolution SP repad should use the corrected expansion delta. "
        f"Got width {repadded.shape[1]} instead of {expected_width}."
    )
