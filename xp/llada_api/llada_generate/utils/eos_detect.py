import torch

# CPU function to detect EOS within a block and all tokens before EOS are unmasked
@torch.no_grad()
def detect_eos(x_accum, eos_token_id, block_slice, mask_block_idx):
    block_tokens = x_accum[:, block_slice]
    eos_mask_block = (block_tokens == eos_token_id)
    any_eos = eos_mask_block.any(dim=1)
    if any_eos.any():
        B, Lb = block_tokens.shape
        # First eos position per row; set to Lb if none
        first_eos_pos = torch.argmax(eos_mask_block.to(torch.int64), dim=1)
        first_eos_pos = torch.where(any_eos, first_eos_pos, torch.full_like(first_eos_pos, Lb))
        positions = torch.arange(Lb, device=block_tokens.device).unsqueeze(0).expand(B, Lb)
        before_mask = positions < first_eos_pos.unsqueeze(1)
        masked_before_eos = (mask_block_idx & before_mask).sum(dim=1)
        satisfied = (first_eos_pos == 0) | (masked_before_eos == 0)
        if satisfied.any().item():
            print(f"Detected EOS within block and all tokens before EOS are unmasked. Breaking current block denoising loop.")
            return True
    return False

# CPU function to detect EOS within a block and all tokens before EOS are unmasked
def detect_eos_min_cpu(x_accum, eos_token_id, block_slice, mask_block_idx):
    """
    Optimized CPU interaction: compute everything on-device and sync once.
    Returns a Python bool.
    """
    block_tokens = x_accum[:, block_slice]
    eos_mask_block = (block_tokens == eos_token_id)
    any_eos_row = eos_mask_block.any(dim=1)  # (B,) on device

    B, Lb = block_tokens.shape
    first_eos_pos = torch.argmax(eos_mask_block.to(torch.int64), dim=1)
    first_eos_pos = torch.where(any_eos_row, first_eos_pos, torch.full_like(first_eos_pos, Lb))

    positions = torch.arange(Lb, device=block_tokens.device).unsqueeze(0).expand(B, Lb)
    before_mask = positions < first_eos_pos.unsqueeze(1)
    masked_before_eos = (mask_block_idx & before_mask).sum(dim=1)

    satisfied_row = (first_eos_pos == 0) | (masked_before_eos == 0)
    satisfied_row = any_eos_row & satisfied_row

    return bool(satisfied_row.any().item())

'''
def detect_eos_min_gpu_partial(x_accum, eos_token_id, block_slice, mask_block_idx):
    """
    GPU-optimized EOS detection with minimal ops:
      - Uses a prefix-sum over eos_mask to mark positions at/after the first EOS
      - Avoids argmax and range construction
      - Performs one CPU sync at the end
    Returns a Python bool.
    """
    block_tokens = x_accum[:, block_slice]                  # (B, Lb)
    eos_mask_block = (block_tokens == eos_token_id)         # (B, Lb)
    any_eos_row = eos_mask_block.any(dim=1)                 # (B,)

    # after_eos[b, j] == True iff exists k <= j with eos_mask_block[b, k] == True
    after_eos = eos_mask_block.cumsum(dim=1).clamp(max=1).bool()

    # Any masked token strictly before first EOS?
    masked_before_eos_any = (mask_block_idx & (~after_eos)).any(dim=1)  # (B,)

    # EOS at first position is acceptable
    eos_at_start = eos_mask_block[:, :1].any(dim=1)  # (B,)

    satisfied_row = any_eos_row & (eos_at_start | (~masked_before_eos_any))
    return satisfied_row

def detect_eos_optim_gpu(x_accum, eos_token_id, block_slice, mask_block_idx):
    satisfied_row = _detect_eos_min_gpu_compiled(x_accum, eos_token_id, block_slice, mask_block_idx)
    return bool(satisfied_row.any().item())
'''

@torch.no_grad()
def detect_eos_optim_gpu(x_accum, eos_token_id, block_slice, mask_block_idx):
    """
    GPU-optimized EOS detection with minimal ops:
      - Uses a prefix-sum over eos_mask to mark positions at/after the first EOS
      - Avoids argmax and range construction
      - Performs one CPU sync at the end
    Returns a Python bool.
    """
    block_tokens = x_accum[:, block_slice]                  # (B, Lb)
    eos_mask_block = (block_tokens == eos_token_id)         # (B, Lb)
    any_eos_row = eos_mask_block.any(dim=1)                 # (B,)

    # after_eos[b, j] == True iff exists k <= j with eos_mask_block[b, k] == True
    after_eos = eos_mask_block.cumsum(dim=1).clamp(max=1).bool()

    # Any masked token strictly before first EOS?
    masked_before_eos_any = (mask_block_idx & (~after_eos)).any(dim=1)  # (B,)

    # EOS at first position is acceptable
    eos_at_start = eos_mask_block[:, :1].any(dim=1)  # (B,)

    satisfied_row = any_eos_row & (eos_at_start | (~masked_before_eos_any))
    return bool(satisfied_row.any().item())


# Compile optimized EOS path if available and announce status
DETECT_EOS_COMPILED = False
try:
    _detect_eos_min_gpu_compiled = torch.compile(
        detect_eos_optim_gpu,
        fullgraph=False,
        mode="reduce-overhead",
        dynamic=True,
    )  # type: ignore[attr-defined]
    DETECT_EOS_COMPILED = True
    print("[eos_detect] torch.compile: detect_eos_min_gpu_partial compiled successfully.")
except Exception:
    _detect_eos_min_gpu_compiled = detect_eos_optim_gpu
    DETECT_EOS_COMPILED = False
    print("[eos_detect] torch.compile: using uncompiled detect_eos_min_gpu_partial.")


@torch.no_grad()
def detect_eos_in_block(x_block: torch.Tensor, eos_token_id: int, mask_block_idx: torch.Tensor) -> bool:
    """
    Fast on-device EOS detection for a single block view.

    Returns True if there exists an EOS token in the block and all tokens strictly
    to its left are already unmasked (i.e., mask_block_idx is False before first EOS).

    Args:
        x_block: (B, Lb) token ids for the active block.
        eos_token_id: EOS token id.
        mask_block_idx: (B, Lb) boolean mask indicating which positions are still masked.
    """
    eos_mask = (x_block == eos_token_id)                 # (B, Lb)
    any_eos_row = eos_mask.any(dim=1)                    # (B,)

    # after_eos[b, j] == True iff exists k <= j with eos_mask[b, k] == True
    after_eos = eos_mask.cumsum(dim=1).clamp(max=1).bool()

    # Any masked token strictly before first EOS?
    masked_before_eos_any = (mask_block_idx & (~after_eos)).any(dim=1)  # (B,)

    # EOS at first position is acceptable
    eos_at_start = eos_mask[:, :1].any(dim=1)  # (B,)

    satisfied_row = any_eos_row & (eos_at_start | (~masked_before_eos_any))
    return bool(satisfied_row.any().item())


@torch.no_grad()
def detect_eos_per_batch(
    x_block: torch.Tensor,
    eos_token_id: int,
    mask_token_id: int,
) -> tuple[bool, torch.Tensor]:
    """
    Detect EOS completion status for each batch element.
    
    A batch element is considered "completed" if:
    1. It contains an EOS token, AND
    2. All tokens before the first EOS are not equal to mask_token_id (i.e., they are filled)
    
    Args:
        x_block: (B, L) token ids for the active block.
        eos_token_id: EOS token id indicating end of sequence.
        mask_token_id: Mask token id indicating unfilled positions.
    
    Returns:
        tuple of:
        - all_completed (bool): True if ALL batches have completed, False otherwise.
        - completed_indices (Tensor): 1D tensor of batch indices that have completed (on same device).
    
    Example:
        >>> x_block = torch.tensor([
        ...     [101, 102, 103, 2],    # Batch 0: EOS at pos 3, no masks before -> completed
        ...     [101, 0, 103, 2],      # Batch 1: EOS at pos 3, mask at pos 1 -> NOT completed
        ...     [2, 0, 0, 0],          # Batch 2: EOS at pos 0 -> completed
        ... ])
        >>> all_done, completed = detect_eos_per_batch(x_block, eos_token_id=2, mask_token_id=0)
        >>> all_done
        False
        >>> completed
        tensor([0, 2])
    """
    # Find EOS and mask token positions
    eos_mask = (x_block == eos_token_id)        # (B, L)
    is_mask_token = (x_block == mask_token_id)  # (B, L)
    
    # after_eos[b, j] == True iff exists k <= j with eos_mask[b, k] == True
    after_eos = eos_mask.cumsum(dim=1).bool()  # (B, L) - removed clamp, cumsum > 0 is enough
    
    # Completed: has EOS AND no mask tokens strictly before first EOS
    # ~after_eos marks positions strictly before first EOS
    completed_mask = eos_mask.any(dim=1) & ~(is_mask_token & ~after_eos).any(dim=1)  # (B,)
    
    # Return all_completed (single CPU sync) and indices tensor (stays on device)
    return bool(completed_mask.all().item()), completed_mask.nonzero(as_tuple=False).flatten()