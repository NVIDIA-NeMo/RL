import numpy as np
import torch, torch._dynamo as dynamo
import torch.nn.functional as F
import math
import time
import os
from contextlib import nullcontext
from torch.profiler import profile as torch_profile, record_function, ProfilerActivity
from utils.eos_detect import detect_eos, detect_eos_min_cpu, detect_eos_optim_gpu, detect_eos_in_block, detect_eos_per_batch
from transformers.cache_utils import DynamicCache, StaticCache
from BatchStaticCache import BatchStaticCache
from utils.sampler import get_transfer_index_optimized, get_transfer_index


def _extract_base_model(model):
    base = getattr(model, "model", None)
    if base is not None:
        return base
    return model


class StaticBufferManager:
    """
    Pre-allocates all buffers for static/dynamic block generation.
    Enables CUDA graph capture by ensuring zero runtime allocation in decode loop.
    
    Usage:
        # Create once at startup
        buffers = StaticBufferManager(model, batch_size=4, gen_length=256, 
                                       block_length=128, prompt_len=512, mode="static")
        
        # For each generation (reuses pre-allocated buffers)
        output = generate_static_block_size_optimized(model, prompt, buffers, ...)
    """
    
    def __init__(
        self,
        model,
        batch_size: int,
        gen_length: int,
        block_length: int,
        prompt_len: int,
        mode: str = "static",  # "static" or "dynamic"
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.B = batch_size
        self.gen_length = gen_length
        self.block_length = block_length
        self.prompt_len = prompt_len
        self.mode = mode
        self.device = device or torch.device("cuda")
        self.dtype = dtype
        
        # Max block size: 1x for static, 2x for dynamic
        self.max_block = block_length if mode == "static" else 2 * block_length
        self.total_len = prompt_len + gen_length
        self.max_total_len = self.total_len + self.max_block
        
        # Get vocab size from model
        self.vocab_size = model.config.vocab_size
        
        # ============================================================
        # PRE-ALLOCATE ALL BUFFERS (single allocation, reused forever)
        # ============================================================
        
        # Sequence accumulator
        self.x_accum = torch.empty(
            (self.B, self.total_len + 2 * block_length),
            dtype=torch.long, device=self.device
        )
        
        # Working block buffer
        self.x_block = torch.empty(
            (self.B, self.max_block),
            dtype=torch.long, device=self.device
        )
        
        # Logits buffer (includes context position)
        self.logits_use = torch.empty(
            (self.B, self.max_block, self.vocab_size),
            dtype=self.dtype, device=self.device
        )
        
        # Mask tracking
        self.mask_block_idx = torch.empty(
            (self.B, self.max_block),
            dtype=torch.bool, device=self.device
        )
        
        # Context logits (predictor for position 0)
        self.next_logits_context = torch.empty(
            (self.B, 1, self.vocab_size),
            dtype=self.dtype, device=self.device
        )
        
        # Position buffers
        self.position_buffer = torch.arange(
            self.max_total_len, device=self.device, dtype=torch.long
        )
        self.cache_position_block = torch.empty(
            (self.B, self.max_block),
            dtype=torch.long, device=self.device
        )
        
        # Pre-computed arange for cache position calculation
        self.arange_max_block = torch.arange(
            self.max_block, device=self.device, dtype=torch.long
        ).unsqueeze(0)
        
        # Index buffers (constant, created once)
        self.block_indices = torch.arange(
            self.max_block, device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(self.B, -1).contiguous()
        self.batch_idx = torch.arange(self.B, device=self.device)
        
        # Per-batch state tensors
        self.current_block_start = torch.empty(
            (self.B,), dtype=torch.long, device=self.device
        )
        self.prefix_len = torch.zeros(
            (self.B,), dtype=torch.long, device=self.device
        )
        self.batch_completed = torch.zeros(
            (self.B,), dtype=torch.bool, device=self.device
        )
        
        # For dynamic mode: active_lens per batch
        if mode == "dynamic":
            self.active_lens = torch.empty(
                (self.B,), dtype=torch.long, device=self.device
            )
        
        # Scalar tensors (on device to avoid sync)
        self.iter_count_t = torch.zeros((), device=self.device, dtype=torch.int32)
        self.max_masked_tensor = torch.zeros((), device=self.device, dtype=torch.int32)
        
        # Gather positions buffer
        self.gather_positions = torch.empty(
            (self.B, self.max_block),
            dtype=torch.long, device=self.device
        )
        
        # Validity mask buffer
        self.validity_mask = torch.empty(
            (self.B, self.max_block),
            dtype=torch.bool, device=self.device
        )
        
        # Working buffers for vectorized operations
        self.needs_advance = torch.empty((self.B,), dtype=torch.bool, device=self.device)
        self.has_mask = torch.empty((self.B,), dtype=torch.bool, device=self.device)
        self.first_mask_pos = torch.empty((self.B,), dtype=torch.long, device=self.device)
        self.has_unmasked_prefix = torch.empty((self.B,), dtype=torch.bool, device=self.device)
        self.new_prefix = torch.empty((self.B,), dtype=torch.long, device=self.device)
        self.would_exceed = torch.empty((self.B,), dtype=torch.bool, device=self.device)
        
        # KV cache (created once, reused)
        self.past_key_values = None
        
        print(f"[StaticBufferManager] Allocated buffers for B={self.B}, "
              f"max_block={self.max_block}, vocab={self.vocab_size}, mode={mode}")
        
    def reset(self, prompt: torch.Tensor, mask_id: int):
        """Reset buffers for new generation (no allocation, only fills)."""
        B, prompt_len = prompt.shape
        assert B <= self.B, f"Batch size {B} exceeds allocated {self.B}"
        assert prompt_len <= self.prompt_len, f"Prompt length {prompt_len} exceeds allocated {self.prompt_len}"
        
        # Reset sequence buffer
        self.x_accum[:B, :prompt_len].copy_(prompt)
        self.x_accum[:B, prompt_len:].fill_(mask_id)
        
        # Reset state tensors
        self.current_block_start[:B].fill_(prompt_len)
        self.prefix_len[:B].zero_()
        self.batch_completed[:B].zero_()
        self.iter_count_t.zero_()
        self.max_masked_tensor.zero_()
        
        if self.mode == "dynamic":
            self.active_lens[:B].fill_(self.block_length)
        
        # Reset mask
        self.mask_block_idx[:B, :self.block_length].fill_(True)
    
    def init_kv_cache(self, model, batch_size: int, total_len: int):
        """Initialize or reset KV cache."""
        if self.past_key_values is None:
            self.past_key_values = BatchStaticCache(
                config=model.config,
                max_batch_size=self.B,
                max_cache_len=self.total_len + self.block_length,
                device=self.device,
                dtype=self.dtype
            )
        # Reset cache state if needed
        return self.past_key_values


def _ensure_commit_mask_capacity(model, batch_size, max_block_len, max_total_len):
    """
    Ensure the commit mask buffer has sufficient capacity for the given batch size.
    
    Searches for _commit_mask_buffer in multiple locations:
    - model.model (base model)
    - model.encoder (for DiffEncoderModel architecture)
    
    The buffer shape is [batch_size, 1, max_block_len, max_total_len] to support
    per-batch commit masks where each batch element can have different commit positions.
    """
    # Try multiple locations where the commit mask buffer might be
    base_model = _extract_base_model(model)
    buffer = getattr(base_model, "_commit_mask_buffer", None)
    buffer_owner = base_model
    
    # Also check model.encoder (for DiffEncoderModel architecture)
    if buffer is None or buffer.numel() == 0:
        encoder = getattr(model, "encoder", None)
        if encoder is not None:
            buffer = getattr(encoder, "_commit_mask_buffer", None)
            if buffer is not None and buffer.numel() > 0:
                buffer_owner = encoder
    
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
    buffer_owner._commit_mask_buffer = new_buffer

    if hasattr(buffer_owner, "config"):
        buffer_owner.config.commit_mask_max_batch = target_shape[0]
        buffer_owner.config.commit_mask_max_block = target_shape[2]
        buffer_owner.config.commit_mask_max_total_len = target_shape[3]

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
    prefix_bidir=False,  # If True, committed prefix can attend to uncommitted tokens
    distance_bidir=None,  # If set, limits attention to only distance_bidir FUTURE tokens
    profile=False,  # If True, return timing breakdown stats
    profile_detailed=False,  # If True, use torch.profiler for detailed CPU/GPU breakdown
    min_prefix_len=1,  # Minimum consecutive unmasked tokens required before advancement (1=new, 2=original)
    unified_prefix=False,  # If True, use minimum prefix_len across all batches (enables efficient shared masks)
    attn_kernel="sdpa",  # Attention kernel: "sdpa" (default), "te", "flash"
    use_fused_qkv=False,  # If True, use fused QKV projection (faster for large batches)
):
    """
    Multi-batch version with per-batch independent processing.
    
    Each batch maintains its own state:
    - current_block_start: position in x_accum
    - prefix_len: tokens to commit  
    - context logits: predictor for position 0
    - mask state
    
    Profiling options:
    - profile=True: Lightweight timing using CUDA events (returns stats dict)
    - profile_detailed=True: Full torch.profiler with CPU/GPU breakdown (prints table + saves trace)
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
    
    position_buffer = torch.arange(max_total_len, device=device, dtype=torch.long)

    # ---- Prefill KV cache ----
    t0 = time.perf_counter()
    prefill_cache = BatchStaticCache(
        config=model.config,
        max_batch_size=B,
        max_cache_len=total_len + block_length,
        device=model.device,
        dtype=model.dtype
    )

    prefill_out = model(
        prompt,
        use_cache=True,
        past_cache_length=0,
        committed_prefix_len=prompt_len,
        past_key_values=prefill_cache,
        cache_position=position_buffer[:prompt_len],
        block_length=prompt_len,
        prefix_bidir=prefix_bidir,
        distance_bidir=distance_bidir,
        attn_kernel=attn_kernel,
        use_fused_qkv=use_fused_qkv,
    )
    prefill_time = time.perf_counter() - t0
    past_key_values = prefill_cache

    # Context predictor [B, 1, V] - INDEPENDENT PER BATCH
    next_logits_context = prefill_out.logits.clone()
    if next_logits_context.dim() != 3 or next_logits_context.size(1) != 1:
        next_logits_context = next_logits_context[:, -1:, :].clone()

    # ---- Reusable device buffers ----
    vocab_size = int(next_logits_context.size(-1))
    logits_use = torch.empty((B, 2 * block_length, vocab_size), device=device, dtype=next_logits_context.dtype)
    mask_block_idx = torch.empty((B, 2 * block_length), device=device, dtype=torch.bool)
    x_block = torch.empty((B, block_length), dtype=prompt.dtype, device=device)

    iter_count_t = torch.zeros((), device=device, dtype=torch.int32)
    active_len = block_length
    
    # Track max block size and max masked tokens across all iterations
    max_block_size_recorded = block_length  # Fixed block size in this mode
    # Keep as tensor to avoid sync during loop - only convert at end
    max_masked_tensor = torch.zeros((), device=device, dtype=torch.int32)
    
    # Per-batch tracking tensors - INDEPENDENT
    current_block_start = torch.full((B,), prompt_len, device=device, dtype=torch.long)
    prefix_len = torch.zeros((B,), device=device, dtype=torch.long)
    batch_completed = torch.zeros((B,), device=device, dtype=torch.bool)
    
    block_indices = torch.arange(block_length, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
    
    # Initialize context logits per batch
    for b in range(B):
        logits_use[b, 0, :].copy_(next_logits_context[b, 0, :])
    
    # Gather initial x_block
    gather_positions = current_block_start.unsqueeze(1) + block_indices[:, :active_len]
    x_block[:, :active_len] = x_accum.gather(1, gather_positions)
    mask_block_idx[:, :block_length].fill_(True)
    
    # ---- Decode loop (VECTORIZED - no per-batch .item() syncs) ----
    d0 = time.perf_counter()
    batch_idx = torch.arange(B, device=device)  # Reusable index tensor
    
    # Profiling accumulators (only used if profile=True)
    if profile:
        prof_model_ms = 0.0
        prof_logits_setup_ms = 0.0
        prof_sampling_ms = 0.0
        prof_scatter_ms = 0.0
        prof_advance_ms = 0.0
        prof_detect_ms = 0.0
        prof_iterations = 0
        # Create reusable CUDA events
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)
    
    # Detailed profiler setup (torch.profiler)
    detailed_profiler = None
    if profile_detailed:
        detailed_profiler = torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,  # Set True for call stacks (slower)
        )
        detailed_profiler.__enter__()
    
    while True:
        cache_position_block = current_block_start.unsqueeze(1) + torch.arange(active_len, device=device).unsqueeze(0)
        
        # Track max masked tokens - keep as tensor, no .item() sync
        is_masked_pre = (x_block[:, :active_len] == mask_id)
        per_batch_masked = is_masked_pre.sum(dim=1)  # [B]
        max_masked_tensor = torch.maximum(max_masked_tensor, per_batch_masked.max().to(torch.int32))
        
        # ---- PROFILE: Model forward ----
        if profile:
            evt_start.record()
        
        # Model call commits prefix_len tokens from PREVIOUS iteration to KV cache
        with record_function("model_forward") if profile_detailed else nullcontext():
            logits_block = model(
                x_block[:, :active_len],
                past_key_values=past_key_values,
                past_cache_length=current_block_start,
                committed_prefix_len=prefix_len,
                cache_position=cache_position_block,
                block_length=active_len,
                prefix_bidir=prefix_bidir,
                distance_bidir=distance_bidir,
                attn_kernel=attn_kernel,
                use_fused_qkv=use_fused_qkv,
            ).logits

        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_model_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()

        # Set up logits for ALL batches
        with record_function("logits_setup") if profile_detailed else nullcontext():
            logits_use[:, 0, :].copy_(next_logits_context[:, 0, :])
            logits_use[:, 1:active_len, :].copy_(logits_block[:, :active_len - 1, :])

        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_logits_setup_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()

        # Sample tokens - mask out completed batches
        validity_mask = ~batch_completed.unsqueeze(1).expand(-1, active_len)
        active_mask = mask_block_idx[:, :active_len] & validity_mask
        
        with record_function("sampling") if profile_detailed else nullcontext():
            transfer_idx = get_transfer_index_optimized(
                logits_use[:, :active_len, :],
                active_mask, 
                x_block[:, :active_len],
                sampling_scaling_factor=factor
            )
            
            # Update mask per batch
            transferred_valid = transfer_idx & validity_mask
            mask_block_idx[:, :active_len] = mask_block_idx[:, :active_len] & ~transferred_valid

        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_sampling_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()

        iter_count_t += 1
        
        # Write sampled tokens back to x_accum BEFORE any advancement
        with record_function("scatter_gather") if profile_detailed else nullcontext():
            gather_positions = current_block_start.unsqueeze(1) + block_indices[:, :active_len]
            x_accum.scatter_(1, gather_positions, x_block[:, :active_len])
        
        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_scatter_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()

        # ============================================================
        # VECTORIZED: Advance batches that had prefix_len > 0
        # prefix_len stores (first_mask_pos - 1), so prefix_len >= (min_prefix_len - 1) 
        # is equivalent to first_mask_pos >= min_prefix_len
        # ============================================================
        with record_function("advancement") if profile_detailed else nullcontext():
            needs_advance = (prefix_len >= (min_prefix_len - 1)) & ~batch_completed  # [B] bool
            
            if needs_advance.any():  # Single sync instead of O(B) syncs
                # Vectorized context logit update: next_logits_context[b, 0, :] = logits_block[b, prefix_len[b], :]
                selected_logits = logits_block[batch_idx, prefix_len, :]  # [B, V]
                next_logits_context[:, 0, :] = torch.where(
                    needs_advance.unsqueeze(1),
                    selected_logits,
                    next_logits_context[:, 0, :]
                )
                
                # Vectorized position advancement
                current_block_start = torch.where(needs_advance, current_block_start + prefix_len + 1, current_block_start)
                prefix_len = torch.where(needs_advance, torch.zeros_like(prefix_len), prefix_len)
                
                # Vectorized regather for advancing batches
                new_gather_positions = current_block_start.unsqueeze(1) + block_indices[:, :block_length]
                new_x_block = x_accum.gather(1, new_gather_positions)
                x_block[:, :block_length] = torch.where(
                    needs_advance.unsqueeze(1),
                    new_x_block,
                    x_block[:, :block_length]
                )
                
                # Vectorized mask refresh for advancing batches
                new_mask = (x_block[:, :block_length] == mask_id)
                mask_block_idx[:, :block_length] = torch.where(
                    needs_advance.unsqueeze(1),
                    new_mask,
                    mask_block_idx[:, :block_length]
                )

        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_advance_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()

        # ============================================================
        # VECTORIZED: Check EOS and find first_pos for all batches
        # ============================================================
        with record_function("detection") if profile_detailed else nullcontext():
            # Vectorized EOS detection
            if eos_token_id is not None:
                has_eos = (x_block[:, :active_len] == eos_token_id).any(dim=1)  # [B]
                batch_completed = batch_completed | has_eos
            
            # Vectorized first mask position detection
            # argmax returns index of first True (or 0 if all False)
            current_mask = mask_block_idx[:, :active_len] & ~batch_completed.unsqueeze(1)
            has_mask = current_mask.any(dim=1)  # [B]
            first_mask_pos = current_mask.to(torch.int64).argmax(dim=1)  # [B]
            # If no mask, treat as active_len (all unmasked)
            first_mask_pos = torch.where(has_mask, first_mask_pos, torch.full_like(first_mask_pos, active_len))
            
            # Vectorized prefix calculation
            has_unmasked_prefix = (first_mask_pos > 0) & ~batch_completed
            new_prefix = torch.clamp(first_mask_pos - 1, min=0)  # [B]
            
            # UNIFIED PREFIX: Use minimum across all non-completed batches
            # This enables efficient shared attention masks [1, 1, Q, K] instead of [B, 1, Q, K]
            if unified_prefix:
                # Only consider non-completed batches for the minimum
                active_mask = ~batch_completed
                if active_mask.any():
                    # Set completed batches to max value so they don't affect min
                    masked_prefix = torch.where(active_mask, new_prefix, torch.full_like(new_prefix, active_len))
                    min_prefix = masked_prefix.min()  # scalar
                    # Apply minimum to all active batches
                    new_prefix = torch.where(active_mask, min_prefix.expand(B), new_prefix)
                    # Update has_unmasked_prefix based on unified prefix
                    has_unmasked_prefix = (min_prefix > 0) & active_mask
        
        # Vectorized bounds check
        would_exceed = (current_block_start + new_prefix + 1 + block_length) >= (prompt_len + gen_length)
        batch_completed = batch_completed | (has_unmasked_prefix & would_exceed)
        
        # Update prefix_len vectorized (stores first_mask_pos - 1)
        prefix_len = torch.where(
            has_unmasked_prefix & ~batch_completed & ~would_exceed,
            new_prefix,
            torch.zeros_like(prefix_len)
        )
        
        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_detect_ms += evt_start.elapsed_time(evt_end)
            prof_iterations += 1
        
        # Single sync for loop termination (unavoidable)
        if batch_completed.all():
            break

    # Close detailed profiler and print results
    if profile_detailed and detailed_profiler is not None:
        detailed_profiler.__exit__(None, None, None)
        print("\n" + "="*80)
        print("DETAILED PROFILER RESULTS (torch.profiler)")
        print("="*80)
        print(detailed_profiler.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=30
        ))
        print("\n--- By Custom Regions ---")
        print(detailed_profiler.key_averages(group_by_input_shape=False).table(
            sort_by="cuda_time_total",
            row_limit=15
        ))
        # Save trace file for Chrome trace viewer
        trace_path = f"profile_trace_batch_{B}.json"
        detailed_profiler.export_chrome_trace(trace_path)
        print(f"\nTrace saved to: {trace_path}")
        print("Open in chrome://tracing or https://ui.perfetto.dev/")
        print("="*80 + "\n")

    # ---- Stats (single sync point at end) ----
    decode_time = time.perf_counter() - d0
    max_masked_tokens_per_block = int(max_masked_tensor.item())  # Only sync at end
    
    # Collect profiling stats if enabled (returned, not printed)
    if profile:
        profile_stats = {
            'model_ms': prof_model_ms,
            'logits_setup_ms': prof_logits_setup_ms,
            'sampling_ms': prof_sampling_ms,
            'scatter_ms': prof_scatter_ms,
            'advance_ms': prof_advance_ms,
            'detect_ms': prof_detect_ms,
            'iterations': prof_iterations,
            'wall_clock_ms': decode_time * 1000,
        }
    else:
        profile_stats = None
    
    # Vectorized accepted tokens calculation
    if eos_token_id is not None:
        # Find first EOS position for each batch
        is_eos = x_accum.eq(eos_token_id)  # [B, seq_len]
        has_eos = is_eos.any(dim=1)  # [B]
        first_eos_pos = is_eos.to(torch.int64).argmax(dim=1)  # [B]
        first_eos_pos = torch.where(has_eos, first_eos_pos, torch.full_like(first_eos_pos, x_accum.size(1)))
        accepted_per_batch = torch.clamp(first_eos_pos - prompt_len + 1, min=0)
    else:
        accepted_per_batch = torch.full((B,), x_accum.size(1) - prompt_len + 1, device=device)
    
    accepted_tokens_total = int(accepted_per_batch.sum().item())
    accepted_tokens = accepted_tokens_total // B
    iter_count = int(iter_count_t.item())
    nfe = iter_count

    return x_accum, nfe, iter_count, accepted_tokens, prefill_time, decode_time, block_length, max_block_size_recorded, max_masked_tokens_per_block, profile_stats


@torch.inference_mode()
def generate_static_block_size_optimized(
    model,
    prompt,
    buffers: StaticBufferManager,
    *,
    gen_length=128,
    block_length=128,
    factor=1.0,
    mask_id=151662,
    eos_token_id=None,
    prefix_bidir=False,
    distance_bidir=None,
    profile=False,
    profile_detailed=False,
    min_prefix_len=1,  # Minimum consecutive unmasked tokens required before advancement
    unified_prefix=False,  # If True, use minimum prefix_len across all batches (enables efficient shared masks)
    attn_kernel="sdpa",  # Attention kernel: "sdpa" (default), "te", "flash"
    use_fused_qkv=False,  # If True, use fused QKV projection (faster for large batches)
):
    """
    Optimized static block generation with pre-allocated buffers.
    Zero runtime allocation in the decode loop.
    
    Args:
        model: The language model
        prompt: Input prompt tensor [B, prompt_len]
        buffers: Pre-allocated StaticBufferManager instance
        gen_length: Number of tokens to generate
        block_length: Block size for generation
        factor: Sampling scaling factor
        mask_id: Mask token ID
        eos_token_id: End of sequence token ID
        prefix_bidir: If True, committed prefix can attend to uncommitted tokens
        distance_bidir: If set, limits attention to only distance_bidir FUTURE tokens
        unified_prefix: If True, use minimum prefix_len across all batches (enables shared masks)
        attn_kernel: Attention kernel selection - "sdpa" (default), "te", "flash"
        use_fused_qkv: If True, use fused QKV projection (faster for large batches)
        profile: If True, return timing breakdown stats
        profile_detailed: If True, use torch.profiler for detailed breakdown
        min_prefix_len: Minimum consecutive unmasked tokens required before advancement (1=new, 2=original)
    
    Returns:
        Same as generate_static_block_size
    """
    model = model.eval()
    
    B, prompt_len = prompt.shape
    total_len = prompt_len + gen_length
    device = prompt.device
    active_len = block_length
    
    # Reset buffers for this generation (no allocation, only fills)
    buffers.reset(prompt, mask_id)
    
    # Ensure commit mask capacity
    max_block_len = max(prompt_len, 2 * block_length)
    max_total_len = prompt_len + gen_length + max_block_len
    _ensure_commit_mask_capacity(model, B, max_block_len, max_total_len)
    
    # ---- Aliases for cleaner code (all are views into buffers, no copy) ----
    x_accum = buffers.x_accum
    x_block = buffers.x_block
    logits_use = buffers.logits_use
    mask_block_idx = buffers.mask_block_idx
    next_logits_context = buffers.next_logits_context
    current_block_start = buffers.current_block_start
    prefix_len = buffers.prefix_len
    batch_completed = buffers.batch_completed
    block_indices = buffers.block_indices
    batch_idx = buffers.batch_idx
    gather_positions = buffers.gather_positions
    cache_position_block = buffers.cache_position_block
    iter_count_t = buffers.iter_count_t
    validity_mask = buffers.validity_mask
    max_masked_tensor = buffers.max_masked_tensor
    
    # Working buffers for vectorized ops
    needs_advance = buffers.needs_advance
    has_mask = buffers.has_mask
    first_mask_pos = buffers.first_mask_pos
    has_unmasked_prefix = buffers.has_unmasked_prefix
    new_prefix = buffers.new_prefix
    would_exceed = buffers.would_exceed
    
    # ---- Prefill KV cache ----
    t0 = time.perf_counter()
    past_key_values = buffers.init_kv_cache(model, B, total_len)
    
    prefill_out = model(
        prompt,
        use_cache=True,
        past_cache_length=0,
        committed_prefix_len=prompt_len,
        past_key_values=past_key_values,
        cache_position=buffers.position_buffer[:prompt_len],
        block_length=prompt_len,
        prefix_bidir=prefix_bidir,
        distance_bidir=distance_bidir,
        attn_kernel=attn_kernel,
        use_fused_qkv=use_fused_qkv,
    )
    prefill_time = time.perf_counter() - t0
    
    # Initialize context logits (IN-PLACE copy to pre-allocated buffer)
    next_logits_context[:B, :, :].copy_(prefill_out.logits[:, -1:, :])
    
    # Initialize logits_use position 0
    logits_use[:B, 0, :].copy_(next_logits_context[:B, 0, :])
    
    # Gather initial x_block (in-place)
    torch.add(
        current_block_start[:B].unsqueeze(1), 
        block_indices[:B, :active_len], 
        out=gather_positions[:B, :active_len]
    )
    x_block[:B, :active_len] = x_accum[:B].gather(1, gather_positions[:B, :active_len])
    
    # Pre-computed arange slice for this active_len
    arange_active = buffers.arange_max_block[:, :active_len]
    
    max_block_size_recorded = block_length
    
    # Profiling setup
    if profile:
        prof_model_ms = 0.0
        prof_logits_setup_ms = 0.0
        prof_sampling_ms = 0.0
        prof_scatter_ms = 0.0
        prof_advance_ms = 0.0
        prof_detect_ms = 0.0
        prof_iterations = 0
        evt_start = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)
    
    detailed_profiler = None
    if profile_detailed:
        detailed_profiler = torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        detailed_profiler.__enter__()
    
    # ---- Decode loop ----
    d0 = time.perf_counter()
    
    while True:
        # Build cache positions (in-place add)
        torch.add(
            current_block_start[:B].unsqueeze(1), 
            arange_active, 
            out=cache_position_block[:B, :active_len]
        )
        
        # Track max masked tokens
        is_masked_pre = (x_block[:B, :active_len] == mask_id)
        per_batch_masked = is_masked_pre.sum(dim=1)
        max_masked_tensor.copy_(torch.maximum(max_masked_tensor, per_batch_masked.max().to(torch.int32)))
        
        # ---- Model forward ----
        if profile:
            evt_start.record()
        
        with record_function("model_forward") if profile_detailed else nullcontext():
            logits_block = model(
                x_block[:B, :active_len],
                past_key_values=past_key_values,
                past_cache_length=current_block_start[:B],
                committed_prefix_len=prefix_len[:B],
                cache_position=cache_position_block[:B, :active_len],
                block_length=active_len,
                prefix_bidir=prefix_bidir,
                distance_bidir=distance_bidir,
                attn_kernel=attn_kernel,
                use_fused_qkv=use_fused_qkv,
            ).logits
        
        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_model_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()
        
        # ---- Logits setup (in-place) ----
        with record_function("logits_setup") if profile_detailed else nullcontext():
            logits_use[:B, 0, :].copy_(next_logits_context[:B, 0, :])
            logits_use[:B, 1:active_len, :].copy_(logits_block[:, :active_len - 1, :])
        
        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_logits_setup_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()
        
        # ---- Sampling ----
        with record_function("sampling") if profile_detailed else nullcontext():
            # Build validity mask (in-place)
            torch.logical_not(
                batch_completed[:B].unsqueeze(1).expand(-1, active_len), 
                out=validity_mask[:B, :active_len]
            )
            active_mask = mask_block_idx[:B, :active_len] & validity_mask[:B, :active_len]
            
            transfer_idx = get_transfer_index_optimized(
                logits_use[:B, :active_len, :],
                active_mask,
                x_block[:B, :active_len],
                sampling_scaling_factor=factor
            )
            
            # Update mask (in-place)
            transferred_valid = transfer_idx & validity_mask[:B, :active_len]
            mask_block_idx[:B, :active_len].logical_and_(~transferred_valid)
        
        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_sampling_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()
        
        iter_count_t += 1
        
        # ---- Scatter back to x_accum ----
        with record_function("scatter_gather") if profile_detailed else nullcontext():
            torch.add(
                current_block_start[:B].unsqueeze(1), 
                block_indices[:B, :active_len], 
                out=gather_positions[:B, :active_len]
            )
            x_accum[:B].scatter_(1, gather_positions[:B, :active_len], x_block[:B, :active_len])
        
        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_scatter_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()
        
        # ---- Advance batches with prefix_len >= (min_prefix_len - 1) ----
        # prefix_len stores (first_mask_pos - 1), so this is equivalent to first_mask_pos >= min_prefix_len
        with record_function("advancement") if profile_detailed else nullcontext():
            torch.logical_and(prefix_len[:B] >= (min_prefix_len - 1), ~batch_completed[:B], out=needs_advance[:B])
            
            if needs_advance[:B].any():
                # Update context logits
                selected_logits = logits_block[batch_idx[:B], prefix_len[:B], :]
                next_logits_context[:B, 0, :] = torch.where(
                    needs_advance[:B].unsqueeze(1),
                    selected_logits,
                    next_logits_context[:B, 0, :]
                )
                
                # Advance positions (in-place where possible)
                advance_amount = torch.where(needs_advance[:B], prefix_len[:B] + 1, torch.zeros_like(prefix_len[:B]))
                current_block_start[:B].add_(advance_amount)
                prefix_len[:B].zero_()
                
                # Regather x_block
                torch.add(
                    current_block_start[:B].unsqueeze(1), 
                    block_indices[:B, :block_length], 
                    out=gather_positions[:B, :block_length]
                )
                new_x_block = x_accum[:B].gather(1, gather_positions[:B, :block_length])
                x_block[:B, :block_length] = torch.where(
                    needs_advance[:B].unsqueeze(1),
                    new_x_block,
                    x_block[:B, :block_length]
                )
                
                # Refresh mask
                new_mask = (x_block[:B, :block_length] == mask_id)
                mask_block_idx[:B, :block_length] = torch.where(
                    needs_advance[:B].unsqueeze(1),
                    new_mask,
                    mask_block_idx[:B, :block_length]
                )
        
        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_advance_ms += evt_start.elapsed_time(evt_end)
            evt_start.record()
        
        # ---- Detection ----
        with record_function("detection") if profile_detailed else nullcontext():
            # EOS detection
            if eos_token_id is not None:
                has_eos = (x_block[:B, :active_len] == eos_token_id).any(dim=1)
                batch_completed[:B].logical_or_(has_eos)
            
            # First mask position detection
            current_mask = mask_block_idx[:B, :active_len] & ~batch_completed[:B].unsqueeze(1)
            torch.any(current_mask, dim=1, out=has_mask[:B])
            torch.argmax(current_mask.to(torch.int64), dim=1, out=first_mask_pos[:B])
            first_mask_pos[:B] = torch.where(has_mask[:B], first_mask_pos[:B], torch.full_like(first_mask_pos[:B], active_len))
            
            # Prefix calculation
            torch.logical_and(first_mask_pos[:B] > 0, ~batch_completed[:B], out=has_unmasked_prefix[:B])
            torch.clamp(first_mask_pos[:B] - 1, min=0, out=new_prefix[:B])
            
            # UNIFIED PREFIX: Use minimum across all non-completed batches
            # This enables efficient shared attention masks [1, 1, Q, K] instead of [B, 1, Q, K]
            if unified_prefix:
                active_mask = ~batch_completed[:B]
                if active_mask.any():
                    # Set completed batches to max value so they don't affect min
                    masked_prefix = torch.where(active_mask, new_prefix[:B], torch.full_like(new_prefix[:B], active_len))
                    min_prefix = masked_prefix.min()  # scalar
                    # Apply minimum to all active batches (in-place)
                    new_prefix[:B] = torch.where(active_mask, min_prefix.expand(B), new_prefix[:B])
                    # Update has_unmasked_prefix based on unified prefix
                    has_unmasked_prefix[:B] = (min_prefix > 0) & active_mask
            
            # Bounds check
            torch.ge(
                current_block_start[:B] + new_prefix[:B] + 1 + block_length, 
                prompt_len + gen_length, 
                out=would_exceed[:B]
            )
            batch_completed[:B].logical_or_(has_unmasked_prefix[:B] & would_exceed[:B])
            
            # Update prefix_len (stores first_mask_pos - 1)
            prefix_len[:B] = torch.where(
                has_unmasked_prefix[:B] & ~batch_completed[:B] & ~would_exceed[:B],
                new_prefix[:B],
                torch.zeros_like(prefix_len[:B])
            )
        
        if profile:
            evt_end.record()
            torch.cuda.synchronize()
            prof_detect_ms += evt_start.elapsed_time(evt_end)
            prof_iterations += 1
        
        # Single sync for loop termination
        if batch_completed[:B].all():
            break
    
    # Close detailed profiler
    if profile_detailed and detailed_profiler is not None:
        detailed_profiler.__exit__(None, None, None)
        print("\n" + "="*80)
        print("DETAILED PROFILER RESULTS (torch.profiler) - OPTIMIZED")
        print("="*80)
        print(detailed_profiler.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=30
        ))
        print("\n--- By Custom Regions ---")
        print(detailed_profiler.key_averages(group_by_input_shape=False).table(
            sort_by="cuda_time_total",
            row_limit=15
        ))
        trace_path = f"profile_trace_optimized_batch_{B}.json"
        detailed_profiler.export_chrome_trace(trace_path)
        print(f"\nTrace saved to: {trace_path}")
        print("="*80 + "\n")
    
    # ---- Stats ----
    decode_time = time.perf_counter() - d0
    max_masked_tokens_per_block = int(max_masked_tensor.item())
    
    if profile:
        profile_stats = {
            'model_ms': prof_model_ms,
            'logits_setup_ms': prof_logits_setup_ms,
            'sampling_ms': prof_sampling_ms,
            'scatter_ms': prof_scatter_ms,
            'advance_ms': prof_advance_ms,
            'detect_ms': prof_detect_ms,
            'iterations': prof_iterations,
            'wall_clock_ms': decode_time * 1000,
        }
    else:
        profile_stats = None
    
    # Accepted tokens calculation
    if eos_token_id is not None:
        is_eos = x_accum[:B].eq(eos_token_id)
        has_eos = is_eos.any(dim=1)
        first_eos_pos = is_eos.to(torch.int64).argmax(dim=1)
        first_eos_pos = torch.where(has_eos, first_eos_pos, torch.full_like(first_eos_pos, x_accum.size(1)))
        accepted_per_batch = torch.clamp(first_eos_pos - prompt_len + 1, min=0)
    else:
        accepted_per_batch = torch.full((B,), x_accum.size(1) - prompt_len + 1, device=device)
    
    accepted_tokens_total = int(accepted_per_batch.sum().item())
    accepted_tokens = accepted_tokens_total // B
    iter_count = int(iter_count_t.item())
    nfe = iter_count
    
    return x_accum, nfe, iter_count, accepted_tokens, prefill_time, decode_time, block_length, max_block_size_recorded, max_masked_tokens_per_block, profile_stats
