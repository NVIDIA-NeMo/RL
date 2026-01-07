import torch
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional


''' Confidence functions'''

def _unmask_low_confidence(logits: torch.Tensor, x0: torch.Tensor):
    # Probability assigned to chosen token
    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    return x0_p, None


def _unmask_high_confidence(logits: torch.Tensor, x0: torch.Tensor):
    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    # Delta between top-2 probabilities per position
    p_all = torch.softmax(logits, dim=-1)
    top2_idx = torch.topk(logits, k=2, dim=-1).indices
    top2_prob = p_all.gather(-1, top2_idx)
    x0_d = top2_prob[..., 0] - top2_prob[..., 1]
    return x0_p, x0_d


def _unmask_top_p_margin(logits: torch.Tensor, mask_index: torch.Tensor):
    # Margin between top-2 probabilities
    p = F.softmax(logits, dim=-1)
    top2 = torch.topk(p, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]

    # Normalize margin to [0,1] over masked tokens per row
    plus_inf  = torch.full_like(margin, float('inf'))
    minus_inf = torch.full_like(margin, float('-inf'))
    masked_for_min = torch.where(mask_index, margin, plus_inf)
    masked_for_max = torch.where(mask_index, margin, minus_inf)
    row_min = masked_for_min.amin(dim=1, keepdim=True)
    row_max = masked_for_max.amax(dim=1, keepdim=True)
    denom = (row_max - row_min)

    normalized = torch.zeros_like(margin)
    nonzero = denom > 0
    normalized = torch.where(mask_index & nonzero, (margin - row_min) / (denom + 1e-12), normalized)
    normalized = torch.where(mask_index & (~nonzero), torch.ones_like(normalized), normalized)
    return normalized, None


def _unmask_random(x0: torch.Tensor):
    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    return x0_p, None


''' Sampling functions'''
def _select_fixed(confidence: torch.Tensor, num_transfer_tokens: torch.Tensor, threshold: Optional[float]) -> torch.Tensor:
    B, L = confidence.shape
    sorted_idx = torch.argsort(confidence, dim=1, descending=True)
    positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    limits = num_transfer_tokens.view(-1, 1)
    base_keep = positions < limits
    if threshold is not None:
        sorted_conf = confidence.gather(1, sorted_idx)
        ge = (sorted_conf >= threshold).to(torch.int64)
        prefix_ok = torch.cumprod(ge, dim=1).bool()
        keep_sorted = base_keep & prefix_ok
    else:
        keep_sorted = base_keep
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    return transfer_index.scatter(1, sorted_idx, keep_sorted)


def _select_confidence_threshold_ref(confidence: torch.Tensor, num_transfer_tokens: torch.Tensor, sampling_scaling_factor: float, threshold: Optional[float]) -> torch.Tensor:
    B, L = confidence.shape
    sorted_idx = torch.argsort(confidence, dim=1, descending=True)
    sorted_conf = confidence.gather(1, sorted_idx)
    positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    limits = num_transfer_tokens.view(-1, 1)
    base_keep = positions < limits
    pos_float = positions.to(sorted_conf.dtype)
    threshs = 1.0 - (sampling_scaling_factor / (pos_float + 2.0))
    threshs[:, 0] = -1.0
    ge_pos = sorted_conf >= threshs
    criteria = ge_pos & (sorted_conf >= threshold) if threshold is not None else ge_pos
    prefix_ok = torch.cumprod(criteria.to(torch.int64), dim=1).bool()
    keep_sorted = base_keep & prefix_ok
    keep_sorted[:, 0] = True
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    return transfer_index.scatter(1, sorted_idx, keep_sorted)


def _select_confidence_threshold_bound(confidence: torch.Tensor, sampling_scaling_factor: float, lower_bound: float = 0.4, upper_bound: float = 0.95) -> torch.Tensor:
    B, L = confidence.shape
    sorted_idx = torch.argsort(confidence, dim=1, descending=True)
    sorted_conf = confidence.gather(1, sorted_idx)
    positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    keep_high_sorted = sorted_conf >= upper_bound
    pos_float = positions.to(sorted_conf.dtype)
    threshs = 1.0 - (sampling_scaling_factor / (pos_float + 2.0))
    threshs[:, 0] = -1.0
    ge_pos = sorted_conf >= threshs
    ge_low = sorted_conf >= lower_bound
    elig_sorted = ge_pos & ge_low
    keep_sorted_final = keep_high_sorted | elig_sorted
    keep_sorted_final[:, 0] = True
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    return transfer_index.scatter(1, sorted_idx, keep_sorted_final)


def _select_confidence_threshold_high(confidence: torch.Tensor, x0_d: torch.Tensor, num_transfer_tokens: torch.Tensor, sampling_scaling_factor: float, threshold: Optional[float]) -> torch.Tensor:
    B, L = confidence.shape
    sorted_idx = torch.argsort(confidence, dim=1, descending=True)
    sorted_conf = confidence.gather(1, sorted_idx)
    positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    limits = num_transfer_tokens.view(-1, 1)
    base_keep = positions < limits
    pos_float = positions.to(sorted_conf.dtype)
    threshs = 1.0 - (sampling_scaling_factor / (pos_float + 2.0))
    threshs[:, 0] = -1.0
    ge_pos = sorted_conf >= threshs
    criteria = ge_pos & (sorted_conf >= threshold) if threshold is not None else ge_pos
    prefix_ok = torch.cumprod(criteria.to(torch.int64), dim=1).bool()
    keep_sorted = base_keep & prefix_ok
    keep_sorted[:, 0] = True
    kept_counts = keep_sorted.sum(dim=1, keepdim=True)
    idx_delta_sorted = torch.argsort(x0_d, dim=1, descending=True)
    allow_sorted_delta = positions < kept_counts
    allow_by_delta = torch.zeros_like(keep_sorted, dtype=torch.bool)
    allow_by_delta = allow_by_delta.scatter(1, idx_delta_sorted, allow_sorted_delta)
    keep_orig = torch.zeros_like(keep_sorted, dtype=torch.bool)
    keep_orig = keep_orig.scatter(1, sorted_idx, keep_sorted)
    top1_idx_orig = sorted_idx[:, 0:1]
    top1_mask_orig = torch.zeros_like(keep_orig, dtype=torch.bool)
    top1_mask_orig.scatter_(1, top1_idx_orig, True)
    rest_mask = keep_orig & (~top1_mask_orig)
    final_keep = top1_mask_orig | (rest_mask & allow_by_delta)
    return final_keep


def _select_cumulative_error(confidence: torch.Tensor, num_transfer_tokens: torch.Tensor, threshold: Optional[float]) -> torch.Tensor:
    B, L = confidence.shape
    sorted_idx = torch.argsort(confidence, dim=1, descending=True)
    sorted_conf = confidence.gather(1, sorted_idx).clamp(min=1e-12)
    cum_log = torch.cumsum(torch.log(sorted_conf), dim=1)
    log_thresh = math.log(max(threshold, 1e-12)) if threshold is not None else float('-inf')
    keep_by_prod = (cum_log >= log_thresh)
    keep_by_prod[:, 0] = True
    positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    limits = num_transfer_tokens.view(-1, 1)
    base_keep = positions < limits
    keep = keep_by_prod & base_keep
    transfer_index = torch.zeros_like(confidence, dtype=torch.bool)
    return transfer_index.scatter(1, sorted_idx, keep)


@torch.no_grad()
def get_transfer_index(logits, temperature, sampling_strategy, unmasking, mask_index, x, num_transfer_tokens, threshold=None, neg_entropy=False, dbg_print=False, 
                      sampling_scaling_factor=1.0, is_complete_block=True):
    if dbg_print:
        print(logits)
    x0 = torch.argmax(logits, dim=-1)

    # Compute per-position confidence
    if unmasking == 'low_confidence':
        x0_p, x0_d = _unmask_low_confidence(logits, x0)
    elif unmasking == 'high_confidence':
        x0_p, x0_d = _unmask_high_confidence(logits, x0)
    elif unmasking == 'top_p_margin':
        x0_p, x0_d = _unmask_top_p_margin(logits, mask_index)
    elif unmasking == 'random':
        x0_p, x0_d = _unmask_random(x0)
    else:
        raise NotImplementedError(unmasking)

    if neg_entropy:
        p = F.softmax(logits, dim=-1)
        epsilon = 1e-10
        log_probs = torch.log(p + epsilon)
        confidence_scores = torch.sum(p * log_probs, dim=-1)
    else:
        confidence_scores = x0_p

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence_scores, -np.inf)
    if num_transfer_tokens is None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    if sampling_strategy == 'fixed':
        transfer_index = _select_fixed(confidence, num_transfer_tokens, threshold)
    elif sampling_strategy == 'confidence_threshold_ref':
        transfer_index = _select_confidence_threshold_ref(confidence, num_transfer_tokens, sampling_scaling_factor, threshold)
    elif sampling_strategy == 'confidence_threshold_bound':
        transfer_index = _select_confidence_threshold_bound(confidence, sampling_scaling_factor)
    elif sampling_strategy == 'confidence_threshold' and unmasking == 'high_confidence':
        transfer_index = _select_confidence_threshold_high(confidence, x0_d, num_transfer_tokens, sampling_scaling_factor, threshold)
    elif sampling_strategy == 'cumulative_error':
        transfer_index = _select_cumulative_error(confidence, num_transfer_tokens, threshold)
    else:
        raise NotImplementedError(f'Sampling strategy {sampling_strategy} not implemented')

    if dbg_print:
        print(f"Number of elements in transfer_index that are True per batch: {transfer_index.sum(dim=1)}")

    x.copy_(torch.where(transfer_index, x0, x))
    return x0, transfer_index


@torch.inference_mode()
#@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)
def get_transfer_index_optimized(logits, mask_index, x, sampling_scaling_factor=1.0):

    # Pick the highest score (greedy) index along the vocab dimension, yielding token IDs of shape (B, L)
    x0 = torch.argmax(logits, dim=-1) # b, l
    
    p = torch.softmax(logits.to(torch.float32), dim=-1)
    #p = F.softmax(logits, dim=-1)
    # x0_p[b, t] is exactly the probability assigned (by softmax) to the chosen token x0[b, t].
    confidence_scores = torch.squeeze(
        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence_scores, -np.inf)

    # Get transfer index of most confident tokens
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    # Two-tier rule:
    # 1) Always accept tokens with confidence >= upper_bound (unlimited)
    # 2) Among remaining, accept tokens that satisfy positional threshold AND >= lower_bound,
    #    up to per-row budget (num_transfer_tokens). Guarantee at least one per row.
    # factor=4
    lower_bound = 0.50       #0.5
    upper_bound = 0.95      #0.90

    B, L = confidence.shape
    sorted_idx = torch.argsort(confidence, dim=1, descending=True)
    sorted_conf = confidence.gather(1, sorted_idx)
    positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    #limits = num_transfer_tokens.view(-1, 1)

    # Always-accept set (does not count toward budget)
    keep_high_sorted = sorted_conf >= upper_bound

    # Budget applies ONLY to threshold-based tokens; highs do not reduce it
    #base_keep_remaining = positions < limits

    # Positional threshold: 1 - factor/(i+2); force i=0 to -1 (always allow top-1)
    pos_float = positions.to(sorted_conf.dtype)
    threshs = 1.0 - (sampling_scaling_factor / (pos_float + 2.0))
    threshs[:, 0] = -1.0
    ge_pos = sorted_conf >= threshs
    ge_low = sorted_conf >= lower_bound
    elig_sorted = ge_pos & ge_low

    # Respect remaining budget for non-highs
    # Deactivated maximum 
    keep_rest_sorted = elig_sorted #& base_keep_remaining

    # Final keep set in sorted order
    keep_sorted_final = keep_high_sorted | keep_rest_sorted

    # Guarantee progress per row
    keep_sorted_final[:, 0] = True

    # Map back to original token positions
    transfer_index = transfer_index.scatter(1, sorted_idx, keep_sorted_final)

    # Shape-stable in-place update avoiding boolean indexing (which lowers to nonzero)
    x.copy_(torch.where(transfer_index, x0, x))
    
    return transfer_index



@torch.no_grad()
def get_transfer_index(logits, temperature, sampling_strategy, unmasking, mask_index, x, num_transfer_tokens, threshold=None, neg_entropy=False, dbg_print=False, 
                      sampling_scaling_factor=1.0, is_complete_block=True):
    if dbg_print:
        print(logits)
    
    # Pick the highest score (greedy) index along the vocab dimension, yielding token IDs of shape (B, L)
    x0 = torch.argmax(logits, dim=-1) # b, l
    
    if unmasking == 'low_confidence':
        # p = F.softmax(logits.to(torch.float64), dim=-1)
        p = F.softmax(logits, dim=-1)
        # x0_p[b, t] is exactly the probability assigned (by softmax) to the chosen token x0[b, t].
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif unmasking == 'high_confidence':
        p = F.softmax(logits, dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
        
        # logits: [B, L, V]
        p_all = torch.softmax(logits, dim=-1)                 # [B, L, V] probabilities for every logit

        top2_idx = torch.topk(logits, k=2, dim=-1).indices    # [B, L, 2] top-2 token ids per position
        top2_prob = p_all.gather(-1, top2_idx)                # [B, L, 2] corresponding probabilities
        
        # Compute the delta between the top-2 probabilities
        delta = top2_prob[..., 0] - top2_prob[..., 1]
        x0_d = delta
        
    elif unmasking == 'top_p_margin':
        # Compute probabilities
        p = F.softmax(logits, dim=-1)                       # (B, L, V)
        # Top-2 per position
        top2 = torch.topk(p, k=2, dim=-1).values            # (B, L, 2)
        margin = top2[..., 0] - top2[..., 1]                # (B, L)

        # Normalize margin to [0,1] over MASKED positions per row
        plus_inf  = torch.full_like(margin, float('inf'))
        minus_inf = torch.full_like(margin, float('-inf'))
        masked_for_min = torch.where(mask_index, margin, plus_inf)
        masked_for_max = torch.where(mask_index, margin, minus_inf)
        row_min = masked_for_min.amin(dim=1, keepdim=True)  # (B, 1)
        row_max = masked_for_max.amax(dim=1, keepdim=True)  # (B, 1)
        denom = (row_max - row_min)

        # If denom==0 (all equal), set normalized=1 on masked; 0 elsewhere by default
        normalized = torch.zeros_like(margin)
        nonzero = denom > 0
        normalized = torch.where(
            mask_index & nonzero,
            (margin - row_min) / (denom + 1e-12),
            normalized
        )
        normalized = torch.where(
            mask_index & (~nonzero),
            torch.ones_like(normalized),
            normalized
        )
        x0_p = normalized  # ∈ [0,1] on masked positions
    elif unmasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(unmasking)
    
    # Calculate negative entropy if requested
    if neg_entropy:
        # p = F.softmax(logits.to(torch.float64), dim=-1)
        p = F.softmax(logits, dim=-1)
        epsilon = 1e-10
        log_probs = torch.log(p + epsilon)
        confidence_scores = torch.sum(p * log_probs, dim=-1)  # negative entropy per position
    else:
        confidence_scores = x0_p
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence_scores, -np.inf)

    # Get transfer index of most confident tokens
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    
    # If no transfer tokens are provided, use all masked tokens
    if num_transfer_tokens is None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
        
    if sampling_strategy == 'fixed':
        # Vectorized selection: argsort once per row, then keep prefix per row length
        B, L = confidence.shape
        sorted_idx = torch.argsort(confidence, dim=1, descending=True)  # (B, L)
        # positions along sorted order
        positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
        # per-row limit
        limits = num_transfer_tokens.view(-1, 1)  # (B, 1)
        base_keep = positions < limits  # (B, L)

        if threshold is not None:
            sorted_conf = confidence.gather(1, sorted_idx)
            ge = (sorted_conf >= threshold).to(torch.int64)
            # Keep until the first element that falls below threshold (inclusive behavior of prefix of 1s)
            prefix_ok = torch.cumprod(ge, dim=1).bool()
            keep = base_keep & prefix_ok
        else:
            keep = base_keep

        # Scatter back to original token positions
        transfer_index = transfer_index.scatter(1, sorted_idx, keep)
    elif sampling_strategy == 'confidence_threshold_ref':
        # Vectorized confidence-threshold selection
        B, L = confidence.shape
        sorted_idx = torch.argsort(confidence, dim=1, descending=True)
        sorted_conf = confidence.gather(1, sorted_idx)
        positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
        limits = num_transfer_tokens.view(-1, 1)
        base_keep = positions < limits

        # Decreasing positional threshold: 1 - factor/(i+2); force i=0 to -1 (always allow top1)
        pos_float = positions.to(sorted_conf.dtype)
        threshs = 1.0 - (sampling_scaling_factor / (pos_float + 2.0))
        threshs[:, 0] = -1.0
        ge_pos = sorted_conf >= threshs
        if threshold is not None:
            ge_global = sorted_conf >= threshold
            criteria = ge_pos & ge_global
        else:
            criteria = ge_pos
        prefix_ok = torch.cumprod(criteria.to(torch.int64), dim=1).bool()
        keep_sorted = base_keep & prefix_ok
        keep_sorted[:, 0] = True
        transfer_index = transfer_index.scatter(1, sorted_idx, keep_sorted)

    elif sampling_strategy == 'confidence_threshold_bound':
        # Two-tier rule:
        # 1) Always accept tokens with confidence >= upper_bound (unlimited)
        # 2) Among remaining, accept tokens that satisfy positional threshold AND >= lower_bound,
        #    up to per-row budget (num_transfer_tokens). Guarantee at least one per row.
        # factor=4
        lower_bound = 0.5       #0.5
        upper_bound = 0.95      #0.90

        B, L = confidence.shape
        sorted_idx = torch.argsort(confidence, dim=1, descending=True)
        sorted_conf = confidence.gather(1, sorted_idx)
        positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
        #limits = num_transfer_tokens.view(-1, 1)

        # Always-accept set (does not count toward budget)
        keep_high_sorted = sorted_conf >= upper_bound

        # Budget applies ONLY to threshold-based tokens; highs do not reduce it
        #base_keep_remaining = positions < limits

        # Positional threshold: 1 - factor/(i+2); force i=0 to -1 (always allow top-1)
        pos_float = positions.to(sorted_conf.dtype)
        threshs = 1.0 - (sampling_scaling_factor / (pos_float + 2.0))
        threshs[:, 0] = -1.0
        ge_pos = sorted_conf >= threshs
        ge_low = sorted_conf >= lower_bound
        elig_sorted = ge_pos & ge_low

        # Respect remaining budget for non-highs
        # Deactivated maximum 
        keep_rest_sorted = elig_sorted #& base_keep_remaining

        # Final keep set in sorted order
        keep_sorted_final = keep_high_sorted | keep_rest_sorted

        # Guarantee progress per row
        keep_sorted_final[:, 0] = True

        # Map back to original token positions
        transfer_index = transfer_index.scatter(1, sorted_idx, keep_sorted_final)
 
    elif sampling_strategy == 'confidence_threshold' and unmasking == 'high_confidence':
        # Vectorized version with delta gating
        B, L = confidence.shape
        sorted_idx = torch.argsort(confidence, dim=1, descending=True)
        sorted_conf = confidence.gather(1, sorted_idx)
        positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
        limits = num_transfer_tokens.view(-1, 1)
        base_keep = positions < limits

        pos_float = positions.to(sorted_conf.dtype)
        threshs = 1.0 - (sampling_scaling_factor / (pos_float + 2.0))
        threshs[:, 0] = -1.0
        ge_pos = sorted_conf >= threshs
        if threshold is not None:
            ge_global = sorted_conf >= threshold
            criteria = ge_pos & ge_global
        else:
            criteria = ge_pos
        prefix_ok = torch.cumprod(criteria.to(torch.int64), dim=1).bool()
        keep_sorted = base_keep & prefix_ok
        keep_sorted[:, 0] = True  # guarantee progress

        # Determine kept count per row
        kept_counts = keep_sorted.sum(dim=1, keepdim=True)  # (B,1)

        # Delta gating: keep only top-k by delta where k == kept_counts
        idx_delta_sorted = torch.argsort(x0_d, dim=1, descending=True)  # (B, L)
        allow_sorted_delta = positions < kept_counts
        allow_by_delta = torch.zeros_like(keep_sorted, dtype=torch.bool)
        allow_by_delta = allow_by_delta.scatter(1, idx_delta_sorted, allow_sorted_delta)

        # Map keep_sorted back to original order
        keep_orig = torch.zeros_like(keep_sorted, dtype=torch.bool)
        keep_orig = keep_orig.scatter(1, sorted_idx, keep_sorted)
        # Force top-1 by confidence to be kept
        top1_idx_orig = sorted_idx[:, 0:1]
        top1_mask_orig = torch.zeros_like(keep_orig, dtype=torch.bool)
        top1_mask_orig.scatter_(1, top1_idx_orig, True)
        rest_mask = keep_orig & (~top1_mask_orig)
        final_keep = top1_mask_orig | (rest_mask & allow_by_delta)
        transfer_index = final_keep

                

    elif sampling_strategy == 'cumulative_error':
        # Vectorized: sort per row, keep prefix while cumulative product >= threshold
        # Using log-domain to convert product constraint to sum of logs
        B, L = confidence.shape
        sorted_idx = torch.argsort(confidence, dim=1, descending=True)  # (B, L)
        sorted_conf = confidence.gather(1, sorted_idx).clamp(min=1e-12)
        cum_log = torch.cumsum(torch.log(sorted_conf), dim=1)
        log_thresh = math.log(max(threshold, 1e-12)) if threshold is not None else float('-inf')
        keep_by_prod = (cum_log >= log_thresh)
        # Always accept at least one token per original behavior
        keep_by_prod[:, 0] = True
        # Restrict by per-row token budget
        positions = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
        limits = num_transfer_tokens.view(-1, 1)  # (B, 1)
        base_keep = positions < limits
        keep = keep_by_prod & base_keep
        transfer_index = transfer_index.scatter(1, sorted_idx, keep)
    else:
        raise NotImplementedError(f'Sampling strategy {sampling_strategy} not implemented')

    # Print number of elements in transfer_index that are True
    if dbg_print:
        # Report per-batch counts without Python loops
        print(f"Number of elements in transfer_index that are True per batch: {transfer_index.sum(dim=1)}")

    # Efficient in-place update of the provided block view `x` using masked scatter
    # This writes selected token ids directly into the original storage (i.e., x_accum[:, block_slice])
    #x.masked_scatter_(transfer_index, x0[transfer_index])


    # Shape-stable in-place update avoiding boolean indexing (which lowers to nonzero)
    # x = where(transfer_index, x0, x)
    x.copy_(torch.where(transfer_index, x0, x))

    return x0, transfer_index