import numpy as np
import torch, torch._dynamo as dynamo
import torch.nn.functional as F
import math
import time
import sys
import os

# Add parent directory (model root) to path for utils imports
_model_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _model_root not in sys.path:
    sys.path.insert(0, _model_root)

from utils.eos_detect import detect_eos, detect_eos_min_cpu, detect_eos_optim_gpu, detect_eos_in_block
from transformers.cache_utils import DynamicCache, StaticCache
from utils.sampler import get_transfer_index_optimized, get_transfer_index


def _extract_base_model(model):
    base = getattr(model, "model", None)
    if base is not None:
        return base
    return model


def _ensure_commit_mask_capacity(model, batch_size, max_block_len, max_total_len):
    base_model = _extract_base_model(model)
    buffer = getattr(base_model, "_commit_mask_buffer", None)
    if buffer is None or buffer.numel() == 0:
        return

    current_shape = buffer.shape
    target_shape = (
        max(current_shape[0], batch_size),
        1,
        max(current_shape[2], max_block_len),
        max(current_shape[3], max_total_len),
    )

    if (
        current_shape[0] >= target_shape[0]
        and current_shape[2] >= target_shape[2]
        and current_shape[3] >= target_shape[3]
    ):
        return

    new_buffer = torch.zeros(target_shape, dtype=buffer.dtype, device=buffer.device)
    base_model._commit_mask_buffer = new_buffer

    if hasattr(base_model, "config"):
        base_model.config.commit_mask_max_batch = target_shape[0]
        base_model.config.commit_mask_max_block = target_shape[2]
        base_model.config.commit_mask_max_total_len = target_shape[3]

@torch.inference_mode()
def generate_continous_indexing(
    model,
    prompt,
    *,
    steps=128,
    step_size=4,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    sampling_strategy="fixed",
    unmasking="low_confidence",
    threshold=0.1,
    factor=1.0,
    mask_id=126336,
    shift_logits=False,
    neg_entropy=False,
    dbg_print=False,
    eos_token_id=None,
):
    """
    Mode 1 (fused): Sliding window algorithm with overlap for autoregressive generation.
    
    Key concepts:
    - Each block starts with a context token (last token from previous content)
    - When sliding window triggers, we expand to double-block and commit prefix to KV cache
    - The last uncommitted token becomes the first token of next block (overlap)
    - This ensures continuous autoregressive prediction across blocks
    
    No saved_context_logit needed:
    - Each block includes its context token at position 0
    - Model naturally produces logits where logits[i] predicts position[i+1]
    - The overlap design eliminates the need for saving/concatenating logits
    """
    B, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = prompt.device
    assert gen_length % block_length == 0, "gen_length must be a multiple of block_length"

    # Sequence buffer: [prompt | generation area filled with masks]
    x_accum = torch.empty((B, total_len), dtype=prompt.dtype, device=device)
    x_accum[:, :prompt_len] = prompt
    x_accum[:, prompt_len:] = mask_id

    max_block_len = max(prompt_len, 2 * block_length)
    max_total_len = prompt_len + gen_length + max_block_len
    _ensure_commit_mask_capacity(model, B, max_block_len, max_total_len)

    # ---- Prefill KV cache ----
    # CRITICAL: Only commit prompt up to position prompt_len-1
    # The last prompt token will be included in the first block
    # This avoids having the same token in KV cache twice
    current_block_start = prompt_len - 1
    t0 = time.perf_counter()
    prefill_cache = DynamicCache()
    prefill_out = model(
        prompt,
        use_cache=True,
        committed_prefix_len=current_block_start,
        past_key_values=prefill_cache,
    )
    prefill_time = time.perf_counter() - t0
    past_key_values = prefill_cache

    # Pre-allocate mask buffer for max possible window size (double-block)
    mask_block_idx = torch.empty((B, 2*block_length), device=device, dtype=torch.bool)
    # Initialize entire mask_block_idx to True 
    mask_block_idx.fill_(True)
    # Performance counters
    accepted_tokens_t = torch.zeros((), device=device, dtype=torch.int32)
    iter_count_t = torch.zeros((), device=device, dtype=torch.int32)

    prefix_len = 0                 # 0 = single-block mode, >0 = double-block mode
    active_len = block_length

    # ---- Initialize first block ----
    # CRITICAL: First block starts at prompt_len-1 to include the last prompt token as context
    # This ensures we have a predictor for position prompt_len
    # Block structure: [prompt[-1], mask, mask, ..., mask]
    #                  ^             ^
    #                  context       positions to fill
    # Create view for active block: [prompt[-1], mask, mask, ..., mask]
    in_x_block_view = x_accum.narrow(1, current_block_start, active_len)
    out_x_block_view = x_accum.narrow(1, current_block_start+1, active_len)
    
    # Initialize mask for this block (True where tokens are masked)
    # Position 0 will be False (it's the context token from prompt)
    # Positions 1 to active_len-1 will be True (they're masked)
    torch.eq(out_x_block_view, mask_id, out=mask_block_idx[:, :active_len])

    # Sliding window parameters
    sliding_window_threshold = block_length  # Trigger double-block when this many positions are unmasked
    sliding_window_size = block_length       # Size of prefix to commit when triggered
    
    #print(f"MASK_ID:{mask_id}")
    
    # ---- Decode loop ----
    d0 = time.perf_counter()
    while True:      
        # Run model on current window (single-block or double-block)
        # committed_prefix_len tells model how many tokens to commit to KV cache
        logits_block = model(
            in_x_block_view,
            past_key_values=past_key_values,
            committed_prefix_len=prefix_len,
        ).logits  # [B, L or 2L, V]
        
        #print(f"MASK_BLOCK_IDX_START: {mask_block_idx}`")
        
        active_mask = mask_block_idx[:, prefix_len:active_len]
        transfer_idx = get_transfer_index_optimized(
            logits_block[:, prefix_len:active_len, :],  # Already sized for right block
            active_mask, 
            out_x_block_view[:, prefix_len:active_len],  # Updated in-place
            sampling_scaling_factor=factor
        )        
        
        active_mask.logical_and_(~transfer_idx)
        
        if prefix_len > 0:
            # Advance the current block sliding window
            # Update the current block view to the next unmasked token
            current_block_start += prefix_len
            prefix_len = 0
            active_len = block_length
            # Update the block views
            in_x_block_view = x_accum.narrow(1, current_block_start, active_len)
            out_x_block_view = x_accum.narrow(1, current_block_start+1, active_len) 
            mask_block_idx[:, :active_len].fill_(True) 
            torch.eq(out_x_block_view, mask_id, out=mask_block_idx[:, :active_len])
        
        
        # Device-side counters
        accepted_tokens_t += transfer_idx.sum().to(torch.int32)
        iter_count_t += 1  # Every iteration increments by 1        
        
        if detect_eos_in_block(out_x_block_view, eos_token_id=eos_token_id, mask_block_idx=mask_block_idx[:, :active_len]):
            break
        
        # Check if first N positions in current block are all unmasked
        # mask_block_idx is True where masked, False where unmasked
        positions_to_check = mask_block_idx[:, :sliding_window_threshold]
        all_unmasked = not positions_to_check.any().item()
        
        if all_unmasked:
            # Determine how many tokens to commit to the left of the block
            prefix_len = sliding_window_size
            active_len = prefix_len+block_length
            if (current_block_start + active_len) >= (prompt_len + gen_length):
                break
            # Grow the window by sliding_window_size
            in_x_block_view = x_accum.narrow(1, current_block_start, active_len)
            out_x_block_view = x_accum.narrow(1, current_block_start+1, active_len)
            # Initialize entire mask_block_idx to True 
            mask_block_idx[:, :active_len].fill_(True) 
            # Update the mask for the new block
            torch.eq(out_x_block_view, mask_id, out=mask_block_idx[:, :active_len])


    # ===== END OF DECODE LOOP =====
    
    # ---- Collect Statistics ----
    accepted_tokens = int(accepted_tokens_t.item())
    iter_count = int(iter_count_t.item())
    nfe = iter_count  # Number of forward evaluations
    decode_time = time.perf_counter() - d0

    if dbg_print:
        print(f"Total iterations: {iter_count}")
        print(f"Accepted tokens: {accepted_tokens}")
        if iter_count > 0:
            print(f"Acceptance rate: {accepted_tokens / iter_count:.4f}")

    return x_accum, nfe, iter_count, accepted_tokens, prefill_time, decode_time

@torch.inference_mode()
def generate_dynamic_block_size(
    model,
    prompt,
    *,
    steps=128,
    step_size=4,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    sampling_strategy="fixed",
    unmasking="low_confidence",
    threshold=0.1,
    factor=1.0,
    mask_id=151662,
    shift_logits=False,
    neg_entropy=False,
    dbg_print=False,
    eos_token_id=None,
    use_compile=True,
    prefix_bidir=False,  # If True, committed prefix can attend to uncommitted tokens
    distance_bidir=None,  # If set, limits attention to only distance_bidir FUTURE tokens
):
    """
    Mode 1 (fused): when a block finishes, the *next iteration* runs ONE double-block forward with
    committed_prefix_len=block_length and use_cache=True — this both (a) commits the left block to KV
    and (b) provides logits for the whole window so we start denoising the right block immediately.
    """
    model = model.eval()
    
    B, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = prompt.device

    # Sequence buffer
    x_accum = torch.empty((B, total_len), dtype=prompt.dtype, device=device)
    x_accum[:, :prompt_len] = prompt
    x_accum[:, prompt_len:] = mask_id

    max_block_len = max(prompt_len, 2 * block_length)
    max_total_len = prompt_len + gen_length + max_block_len
    _ensure_commit_mask_capacity(model, B, max_block_len, max_total_len)
    
    position_buffer = torch.arange(
        max_total_len,
        device=device,
        dtype=torch.long,
    )

    past_cache_length = 0

    # ---- Prefill KV cache (last logit only) ----
    t0 = time.perf_counter()
    #prefill_cache = DynamicCache()
    prefill_cache = StaticCache(config=model.config,
                   max_batch_size=1,
                   max_cache_len=total_len+block_length,
                   device=model.device,
                   dtype=model.dtype)

    prefill_out = model(
        prompt,
        use_cache=True,
        past_cache_length=past_cache_length,
        committed_prefix_len=prompt_len,
        past_key_values=prefill_cache,
        cache_position=position_buffer[:prompt_len],
        block_length=prompt_len,
        prefix_bidir=prefix_bidir,
        distance_bidir=distance_bidir,
    )
    prefill_time = time.perf_counter() - t0
    past_key_values = prefill_cache

    # Make sure context predictor has shape [B, 1, V]
    next_logits_context = prefill_out.logits
    if next_logits_context.dim() != 3 or next_logits_context.size(1) != 1:
        next_logits_context = next_logits_context[:, -1:, :]

    # ---- Reusable device buffers ----
    vocab_size = int(next_logits_context.size(-1))
    # Pre-allocate for max size (2*block_length for fused double-block in Mode 1)
    logits_use = torch.empty((B, 2 * block_length, vocab_size), device=device, dtype=next_logits_context.dtype)
    mask_block_idx = torch.empty((B, 2 * block_length), device=device, dtype=torch.bool)
    iter_count_t = torch.zeros((), device=device, dtype=torch.int32)
    average_window_size_t = torch.zeros((), device=device, dtype=torch.int32)
    
    # Track max block size and max masked tokens across all iterations
    max_block_size_recorded = 0
    max_masked_tokens_per_block = 0

    # Active block as a view
    current_block_start = prompt_len
    logits_use[:, 0, :].copy_(next_logits_context[:, 0, :])
    x_block = x_accum.narrow(1, current_block_start, block_length)
    # Initial mask for this block (only first block_length positions used initially)
    mask_block_idx[:, :block_length].fill_(True)

    #sliding_window_threshold = block_length
    #sliding_window_size = block_length
    prefix_len = 0
    active_len = block_length
    
    # ---- Decode loop ----
    d0 = time.perf_counter()
    while True:
        # Denoise forward (do NOT mutate cache during single-block; commit during double-block)
        cache_position_block = position_buffer[
            current_block_start : current_block_start + active_len
        ]
        # Record the average window size
        average_window_size_t += active_len
        
        # Track largest block size seen
        if active_len > max_block_size_recorded:
            max_block_size_recorded = active_len
        
        # Track max masked tokens in the full block (what the model actually sees)
        # Count directly from x_block for all positions
        is_masked_pre = (x_block == mask_id)
        per_batch_masked = is_masked_pre.sum(dim=1)  # [B]
        current_max_masked = int(per_batch_masked.max().item())
        if current_max_masked > max_masked_tokens_per_block:
            max_masked_tokens_per_block = current_max_masked

        logits_block = model(
            x_block,
            past_key_values=past_key_values,
            past_cache_length=current_block_start,
            committed_prefix_len=prefix_len,
            cache_position=cache_position_block,
            block_length=active_len,
            prefix_bidir=prefix_bidir,
            distance_bidir=distance_bidir,
        ).logits  # [B, L or 2L, V]

        if prefix_len > 0:
            # Extract logit at boundary between blocks for right-block position 0
            logits_use[:, 0:block_length, :].copy_(logits_block[:, (prefix_len-1):-1, :])
            # Fused double-block: advance to right block and extract its predictors
            current_block_start += prefix_len
            prefix_len = 0
            active_len = block_length
            # equivalent to x_block = x_accum[:, current_block_start:current_block_start + block_length]
            x_block = x_accum.narrow(1, current_block_start, active_len)
            # Refresh mask for the new active window (right block)
            torch.eq(x_block, mask_id, out=mask_block_idx[:, :active_len])
        else:
            # Single-block: standard predictor setup
            logits_use[:, 1:active_len, :].copy_(logits_block[:, :-1, :])

        # Sample & in-place update (x_block is a view into x_accum)
        # Only pass the active portion of buffers to avoid unnecessary processing
        active_mask = mask_block_idx[:, :active_len]
        
        transfer_idx = get_transfer_index_optimized(
            logits_use[:, :active_len, :],
            active_mask, x_block,
            sampling_scaling_factor=factor
        )

        # Device-side counters
        iter_count_t += 1  # Every iteration increments by 1

        # Update mask without a full re-eq: newly filled positions become unmasked
        active_mask.logical_and_(~transfer_idx)

        # Optional EOS (host sync of one scalar). Use block-only checker
        if detect_eos_in_block(x_block=x_block, eos_token_id=eos_token_id, mask_block_idx=active_mask):
            break

        active_mask_1d = mask_block_idx[0, :active_len]
        nz = torch.nonzero(active_mask_1d, as_tuple=False)
        first_pos_val = int(nz[0].item()) if nz.numel() > 0 else active_len
        
        # Advance if there is a non-zero unmasked prefix before the first masked token
        if 0 < first_pos_val:
            prefix_len = first_pos_val - 1
            active_len = prefix_len + block_length 
            # Advance or finish
            if (current_block_start + active_len) >= (prompt_len + gen_length):
                break
            # equivalent to x_block = x_accum[:, current_block_start:current_block_start + active_len]
            x_block = x_accum.narrow(1, current_block_start, active_len)
            continue

    # ---- Stats ----
    decode_time = time.perf_counter() - d0
    
    # Determine number of tokens in x_accum until first EOS token is found
    if eos_token_id is not None:
        eos_positions = torch.nonzero(x_accum[0].eq(eos_token_id), as_tuple=False)
        first_mask_idx = int(eos_positions[0].item()) if eos_positions.numel() > 0 else x_accum.size(1)
    else:
        first_mask_idx = x_accum.size(1)
    
    accepted_tokens = first_mask_idx-prompt_len+1
    iter_count = int(iter_count_t.item())
    nfe = iter_count
    average_window_size = int(average_window_size_t.item())/iter_count

    if dbg_print:
        print(f"Total iterations: {iter_count}")
        print(f"Accepted tokens: {accepted_tokens}")
        if iter_count > 0:
            print(f"Acceptance rate: {accepted_tokens / iter_count:.4f}")
        print(f"Max block size recorded: {max_block_size_recorded}")
        print(f"Max masked tokens per block: {max_masked_tokens_per_block}")

    return x_accum, nfe, iter_count, accepted_tokens, prefill_time, decode_time, average_window_size, max_block_size_recorded, max_masked_tokens_per_block, None

@torch.inference_mode()
def generate_static_block_size(
    model,
    prompt,
    *,
    steps=128,
    step_size=4,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    sampling_strategy="fixed",
    unmasking="low_confidence",
    threshold=0.1,
    factor=1.0,
    mask_id=151662,
    shift_logits=False,
    neg_entropy=False,
    dbg_print=False,
    eos_token_id=None,
    use_compile=True,
):
    """
    Mode 1 (fused): when a block finishes, the *next iteration* runs ONE double-block forward with
    committed_prefix_len=block_length and use_cache=True — this both (a) commits the left block to KV
    and (b) provides logits for the whole window so we start denoising the right block immediately.
    """
    model = model.eval()
    
    B, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = prompt.device

    # Sequence buffer
    x_accum = torch.empty((B, total_len + 2*block_length), dtype=prompt.dtype, device=device)
    x_accum[:, :prompt_len] = prompt
    x_accum[:, prompt_len:] = mask_id

    max_block_len = max(prompt_len, 2 * block_length)
    max_total_len = prompt_len + gen_length + max_block_len
    _ensure_commit_mask_capacity(model, B, max_block_len, max_total_len)
    
    position_buffer = torch.arange(
        max_total_len,
        device=device,
        dtype=torch.long,
    )

    past_cache_length = 0

    # ---- Prefill KV cache (last logit only) ----
    t0 = time.perf_counter()
    #prefill_cache = DynamicCache()
    prefill_cache = StaticCache(config=model.config,
                   max_batch_size=1,
                   max_cache_len=total_len+block_length,
                   device=model.device,
                   dtype=model.dtype)

    prefill_out = model(
        prompt,
        use_cache=True,
        past_cache_length=past_cache_length,
        committed_prefix_len=prompt_len,
        past_key_values=prefill_cache,
        cache_position=position_buffer[:prompt_len],
        block_length=prompt_len,
    )
    prefill_time = time.perf_counter() - t0
    past_key_values = prefill_cache

    # Make sure context predictor has shape [B, 1, V]
    next_logits_context = prefill_out.logits
    if next_logits_context.dim() != 3 or next_logits_context.size(1) != 1:
        next_logits_context = next_logits_context[:, -1:, :]

    # ---- Reusable device buffers ----
    vocab_size = int(next_logits_context.size(-1))
    # Pre-allocate for max size (2*block_length for fused double-block in Mode 1)
    logits_use = torch.empty((B, 2 * block_length, vocab_size), device=device, dtype=next_logits_context.dtype)
    mask_block_idx = torch.empty((B, 2 * block_length), device=device, dtype=torch.bool)

    iter_count_t = torch.zeros((), device=device, dtype=torch.int32)
    
    # Track max block size and max masked tokens across all iterations
    max_block_size_recorded = block_length  # Fixed block size in this mode
    max_masked_tokens_per_block = 0

    # Active block as a view
    current_block_start = prompt_len
    logits_use[:, 0, :].copy_(next_logits_context[:, 0, :])
    x_block = x_accum.narrow(1, current_block_start, block_length)
    # Initial mask for this block (only first block_length positions used initially)
    mask_block_idx[:, :block_length].fill_(True)

    prefix_len = 0
    active_len = block_length
    
    # ---- Decode loop ----
    d0 = time.perf_counter()
    while True:
        # Denoise forward (do NOT mutate cache during single-block; commit during double-block)
        cache_position_block = position_buffer[
            current_block_start : current_block_start + active_len
        ]
        
        # Track max masked tokens in the block (before sampling)
        is_masked_pre = (x_block == mask_id)
        current_masked_count = int(is_masked_pre.sum().item())
        if current_masked_count > max_masked_tokens_per_block:
            max_masked_tokens_per_block = current_masked_count
        
        logits_block = model(
            x_block,
            past_key_values=past_key_values,
            past_cache_length=current_block_start,
            committed_prefix_len=prefix_len,
            cache_position=cache_position_block,
            block_length=active_len,
        ).logits  # [B, L or 2L, V]

        logits_use[:, 0, :].copy_(next_logits_context[:, 0, :])
        logits_use[:, 1:active_len, :].copy_(logits_block[:, :-1, :])

        # Sample & in-place update (x_block is a view into x_accum)
        # Only pass the active portion of buffers to avoid unnecessary processing
        active_mask = mask_block_idx[:, :active_len]
        transfer_idx = get_transfer_index_optimized(
            logits_use[:, :active_len, :],
            active_mask, x_block,
            sampling_scaling_factor=factor
        )
        
        # Update mask without a full re-eq: newly filled positions become unmasked
        active_mask.logical_and_(~transfer_idx)
        
        if prefix_len > 0:
            # Move the block to the right
            next_logits_context[:, 0, :].copy_(logits_block[:, prefix_len,:]) 
            current_block_start += prefix_len + 1
            prefix_len=0
            active_len = block_length
            x_block = x_accum.narrow(1, current_block_start, active_len)
            torch.eq(x_block, mask_id, out=mask_block_idx[:, :active_len])

        # Device-side counters
        iter_count_t += 1  # Every iteration increments by 1

        block = x_block[:1, :active_len]                           # [1, L]
        is_mask = block.eq(mask_id)                                # [1, L] bool
        has_mask = is_mask.any(dim=1)                              # [1] bool

        # Index of first True when a mask exists; otherwise use active_len
        first_pos_val_t = torch.where(
            has_mask,
            is_mask.to(torch.int32).argmax(dim=1),                 # [1]
            torch.full((1,), active_len, device=x_block.device, dtype=torch.long)
        )[0]                                                       # scalar tensor on GPU

        # NOTE: The following .item() is the only host sync (needed for Python branching).
        first_pos_val = int(first_pos_val_t.item())

        # Advance if there is a non-zero unmasked prefix before the first masked token
        if 0 < first_pos_val:
            prefix_len = first_pos_val - 1
            active_len = block_length 
            # Advance or finish
            if (current_block_start + active_len) >= (prompt_len + gen_length):
                break
            # Optional EOS (host sync of one scalar). Use block-only checker
            if detect_eos_in_block(x_block=x_block, eos_token_id=eos_token_id, mask_block_idx=active_mask):
                break
            continue

    # ---- Stats ----
    decode_time = time.perf_counter() - d0
    
    # Determine number of tokens in x_accum until first EOS token is found
    if eos_token_id is not None:
        eos_positions = torch.nonzero(x_accum[0].eq(eos_token_id), as_tuple=False)
        first_mask_idx = int(eos_positions[0].item()) if eos_positions.numel() > 0 else x_accum.size(1)
    else:
        first_mask_idx = x_accum.size(1)
    
    accepted_tokens = first_mask_idx-prompt_len+1
    iter_count = int(iter_count_t.item())
    nfe = iter_count

    return x_accum, nfe, iter_count, accepted_tokens, prefill_time, decode_time, block_length, max_block_size_recorded, max_masked_tokens_per_block, None
