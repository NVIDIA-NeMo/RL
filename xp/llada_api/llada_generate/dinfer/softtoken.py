#!/usr/bin/env python3
"""
dInfer Soft Token generation algorithm.

This module implements the Soft Token Sampling method for LLaDA,
based on the experiment in notebooks/soft_token_experiment.py.
"""

import logging
from typing import Tuple, Optional
import torch
from transformers import PreTrainedModel

from .base import DInferGeneration
from ._imports import (
    DINFER_AVAILABLE,
    ParallelDecoder,
    ThresholdParallelDecoder,
    get_num_transfer_tokens,
    get_transfer_index,
    TokenArray,
    BlockIteratorFactory,
    KVCacheFactory,
    MASK_ID,
    EOS_ID
)
from ._utils import FixedParallelDecoder

logger = logging.getLogger(__name__)


class BlockWiseSoftTokenLLM:
    """
    Block-wise diffusion LLM with Soft Token Sampling.
    Adapted from BlockWiseSoftTokenLLM in soft_token_experiment.py.
    
    Supports multi-round soft token inference where each round:
    - Selects non-overlapping mask positions as soft tokens
    - Computes soft embeddings from the current logits
    - Accumulates all soft embeddings for the next forward pass
    
    soft_rounds controls the number of soft token inference rounds:
    - soft_rounds=0: Standard blockwise (no soft tokens, decode from logits1)
    - soft_rounds=1: Single round of soft tokens (original behavior)
    - soft_rounds>1: Multi-round soft token inference
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, soft_token_ratio=0.2, treat_soft_tokens_as_candidates=False, soft_temperature=1.0, intensity=None, soft_rounds=1):
        self.model = model
        self.cache_factory = cache_factory
        self.decoder = decoder
        self.iterator_factory = iterator_factory
        self.num_forwards = 0
        self.cache_updates = 0
        self.early_stop = early_stop
        self.maximum_unroll = maximum_unroll
        self.expected_tpf = expected_tpf
        self.soft_token_ratio = soft_token_ratio
        self.treat_soft_tokens_as_candidates = treat_soft_tokens_as_candidates
        self.soft_temperature = soft_temperature
        self.intensity = intensity
        self.soft_rounds = soft_rounds
        self.input_embeddings = self.model.get_input_embeddings()

    def _compute_logits(self, x, block_loc, kv_cache, use_input_embeds=None):
        """Helper to run model with correct context and embeddings."""
        # Determine input context based on cache type
        if kv_cache is None:
            # Full context (no cache)
            if use_input_embeds is not None:
                logits = self.model(inputs_embeds=use_input_embeds).logits
            else:
                logits = self.model(x.data).logits
            return logits[:, block_loc.start:block_loc.end]
            
        elif kv_cache.cache_type == 'prefix':
            # Prefix Cache: past_key_values contains context up to block_start
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            
            if use_input_embeds is not None:
                # Input embeddings should correspond to x[block_loc.start:]
                logits = self.model(inputs_embeds=use_input_embeds, past_key_values=past_key_values, use_cache=True,
                                  replace_position=replace_position).logits
            else:
                logits = self.model(x[block_loc.start:], past_key_values=past_key_values, use_cache=True,
                                  replace_position=replace_position).logits
            
            curr_len = block_loc.end - block_loc.start
            return logits[:, :curr_len]

        else:
            # Dual/Sliding Cache: typically uses block context
            past_key_values, replace_position = kv_cache.get_key_values(block_loc.start, block_loc.end)
            
            if use_input_embeds is not None:
                 logits = self.model(inputs_embeds=use_input_embeds, past_key_values=past_key_values, use_cache=True,
                                  replace_position=replace_position).logits
            else:
                 # Use x slice instead of block to ensure we have the latest updates
                 logits = self.model(x[block_loc.start:block_loc.end], past_key_values=past_key_values, use_cache=True,
                                  replace_position=replace_position).logits
            return logits

    def _compute_soft_embeddings(self, logits, soft_indices, soft_temperature, intensity):
        """
        Compute soft token embeddings from logits at specified positions.
        
        Supports batch size > 1. Uses the same soft_indices for all batch items.
        
        Args:
            logits: Logits tensor of shape [batch_size, seq_len, vocab_size]
            soft_indices: Tensor of indices (relative to logits dim 1) to compute soft embeddings for
            soft_temperature: Temperature for softmax
            intensity: Intensity for mixing with mask embedding (None = auto-compute from entropy)
            
        Returns:
            soft_embeds: Tensor of shape [batch_size, num_soft, d_model] containing soft embeddings
        """
        batch_size = logits.shape[0]
        
        # Extract logits for soft token positions: [batch_size, num_soft, vocab]
        selected_logits = logits[:, soft_indices]
        
        # Apply soft temperature
        if soft_temperature > 0:
            selected_logits = selected_logits / soft_temperature
            
        probs = torch.softmax(selected_logits, dim=-1)  # [batch_size, num_soft, vocab]
        
        # Compute soft embeddings: weighted average of token embeddings
        # [batch_size, num_soft, vocab] @ [vocab, d_model] -> [batch_size, num_soft, d_model]
        soft_embeds = torch.matmul(probs, self.input_embeddings.weight)
        
        # Apply intensity: auto-compute from entropy if None, or use fixed value if < 1.0
        if intensity is None:
            # Auto-compute per-token intensity based on entropy of logits
            # Entropy ranges from 0 (one-hot/certain) to log(vocab_size) (uniform/uncertain)
            # Intensity should be 1.0 for one-hot (low entropy), 0.0 for uniform (high entropy)
            # Add small epsilon to avoid log(0)
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size, num_soft]
            vocab_size = probs.shape[-1]
            max_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float, device=self.model.device))
            normalized_entropy = entropy / max_entropy  # [batch_size, num_soft], ranges 0-1
            per_token_intensity = 1.0 - normalized_entropy  # [batch_size, num_soft]: 1 for certain, 0 for uncertain
            
            mask_token_tensor = torch.tensor(self.decoder.mask_id, device=self.model.device)
            mask_embed = self.input_embeddings(mask_token_tensor)  # [d_model]
            # Apply per-token intensity: soft_embeds is [batch_size, num_soft, d_model]
            # per_token_intensity is [batch_size, num_soft], broadcast via unsqueeze
            soft_embeds = mask_embed * (1.0 - per_token_intensity).unsqueeze(-1) + soft_embeds * per_token_intensity.unsqueeze(-1)
        elif intensity < 1.0:
            mask_token_tensor = torch.tensor(self.decoder.mask_id, device=self.model.device)
            mask_embed = self.input_embeddings(mask_token_tensor)  # [d_model]
            soft_embeds = mask_embed * (1.0 - intensity) + soft_embeds * intensity
            
        return soft_embeds

    def _run_soft_token_rounds(self, x, block_loc, kv_cache, logits1, mask_indices, 
                                soft_token_ratio, soft_temperature, intensity, soft_rounds):
        """
        Run multiple rounds of soft token inference.
        
        Supports batch size > 1. Uses the same soft token positions for all batch items
        (since mask positions are the same across the batch in diffusion models).
        
        Each round:
        1. Selects non-overlapping mask positions as soft tokens
        2. Computes soft embeddings from the current logits
        3. Runs a forward pass with all accumulated soft embeddings
        
        Args:
            x: TokenArray containing the current sequence
            block_loc: Block location object with start/end
            kv_cache: KV cache object or None
            logits1: Initial logits from standard forward pass [batch_size, block_len, vocab]
            mask_indices: Tensor of all mask indices in the block (relative to block start)
            soft_token_ratio: Ratio of available masks to use as soft tokens per round
            soft_temperature: Temperature for softmax in soft embedding computation
            intensity: Intensity for mixing with mask embedding (None = auto)
            soft_rounds: Number of soft token inference rounds
            
        Returns:
            final_logits: Logits to use for decoding [batch_size, block_len, vocab]
            all_soft_indices: List of soft index tensors from all rounds (for exclusion during decoding)
        """
        if soft_rounds == 0 or mask_indices.numel() == 0 or soft_token_ratio <= 0:
            # No soft token rounds - return original logits
            return logits1, []
        
        batch_size = logits1.shape[0]
        
        # Track all soft indices and embeddings across rounds
        all_soft_indices = []  # List of index tensors (relative to block start)
        all_soft_embeds = []   # List of (indices, embeddings) tuples, embeddings are [batch_size, num_soft, d_model]
        
        # Set of already-used mask positions (as a set for fast lookup)
        used_mask_set = set()
        
        current_logits = logits1
        curr_block_ids = x[block_loc.start:block_loc.end]
        
        for round_idx in range(soft_rounds):
            # Get available mask indices (not yet used as soft tokens)
            available_mask_indices = torch.tensor(
                [idx.item() for idx in mask_indices if idx.item() not in used_mask_set],
                device=self.model.device,
                dtype=torch.long
            )
            
            if available_mask_indices.numel() == 0:
                # No more masks available
                logger.debug(f"Soft token round {round_idx}: No available masks, stopping early")
                break
                
            # Calculate number of soft tokens for this round
            num_soft = int(available_mask_indices.numel() * soft_token_ratio)
            if num_soft == 0:
                logger.debug(f"Soft token round {round_idx}: num_soft=0, stopping early")
                break
            
            # Randomly select soft token positions for this round
            perm = torch.randperm(available_mask_indices.numel(), device=self.model.device)
            soft_indices_round = available_mask_indices[perm[:num_soft]]
            
            # Track these indices
            all_soft_indices.append(soft_indices_round)
            for idx in soft_indices_round.tolist():
                used_mask_set.add(idx)
            
            # Compute soft embeddings from current logits: [batch_size, num_soft, d_model]
            soft_embeds_round = self._compute_soft_embeddings(
                current_logits, soft_indices_round, soft_temperature, intensity
            )
            all_soft_embeds.append((soft_indices_round, soft_embeds_round))
            
            # Prepare input embeddings with ALL soft embeddings accumulated so far
            target_ids = None
            global_offset = 0
            
            if kv_cache is None:
                target_ids = x.data
                global_offset = block_loc.start
            elif kv_cache.cache_type == 'prefix':
                target_ids = x[block_loc.start:]
                global_offset = 0
            else:
                target_ids = curr_block_ids
                global_offset = 0
            
            # Get base embeddings for the input context: [batch_size, len, d_model]
            inputs_embeds = self.input_embeddings(target_ids).clone()
            
            # Replace all soft token positions with their embeddings for all batch items
            for indices, embeds in all_soft_embeds:
                # indices: [num_soft], embeds: [batch_size, num_soft, d_model]
                inputs_embeds[:, global_offset + indices] = embeds
            
            # Forward pass with soft embeddings
            current_logits = self._compute_logits(x, block_loc, kv_cache, use_input_embeds=inputs_embeds)
            self.num_forwards += 1
            
            logger.debug(f"Soft token round {round_idx + 1}/{soft_rounds}: "
                        f"used {num_soft} soft tokens, {available_mask_indices.numel() - num_soft} masks remaining")
        
        return current_logits, all_soft_indices

    def validate_schedule(self, block_length, soft_token_ratio, treat_soft_tokens_as_candidates):
        """ Validates that the decoding schedule can be satisfied with the given soft token ratio.
        """
        # Only validate for FixedParallelDecoder which has steps
        if not hasattr(self.decoder, 'steps') or treat_soft_tokens_as_candidates:
            return

        steps = self.decoder.steps
        current_masks = block_length
        
        # Calculate the schedule for a full block
        base = current_masks // steps
        remainder = current_masks % steps
        
        schedule = []
        for i in range(steps):
            count = base + (1 if i < remainder else 0)
            schedule.append(count)
            
        # Simulate decoding
        for step_idx, num_to_decode in enumerate(schedule):
            num_soft = int(current_masks * soft_token_ratio)
            available = current_masks - num_soft
            
            if available < num_to_decode:
                # Just warn instead of raising error to prevent crashing server
                logger.warning(
                    f"Decoding Schedule Violation: Step {step_idx} requires decoding {num_to_decode} tokens, "
                    f"but only {available} masks are available ({current_masks} total - {num_soft} soft tokens). "
                    f"Reduce soft_token_ratio or enable treat_soft_tokens_as_candidates."
                )
                return
            current_masks -= num_to_decode

    @torch.no_grad()
    def generate(self, prompt, gen_length=128, block_length=128, soft_token_ratio=None, treat_soft_tokens_as_candidates=None, steps=None, threshold=None, soft_temperature=None, intensity=None, soft_rounds=None):
        ''' Generate tokens with diffusion iterations block by block using Soft Token Sampling.
        
        Args:
            prompt: Input prompt tensor
            gen_length: Number of tokens to generate
            block_length: Block size for semi-autoregressive generation
            soft_token_ratio: Ratio of masks to use as soft tokens per round
            treat_soft_tokens_as_candidates: If True, soft token positions can be decoded
            steps: Number of diffusion steps (for FixedParallelDecoder)
            threshold: Confidence threshold (for ThresholdParallelDecoder)
            soft_temperature: Temperature for softmax in soft embedding computation
            intensity: Intensity for mixing with mask (None = auto-compute from entropy)
            soft_rounds: Number of soft token inference rounds:
                - 0: Standard blockwise (no soft tokens)
                - 1: Single round (original soft token behavior)
                - >1: Multi-round soft token inference
        '''
        # Use instance defaults if not provided
        if soft_token_ratio is None:
            soft_token_ratio = self.soft_token_ratio
        if treat_soft_tokens_as_candidates is None:
            treat_soft_tokens_as_candidates = self.treat_soft_tokens_as_candidates
        if soft_temperature is None:
            soft_temperature = self.soft_temperature
        if intensity is None:
            intensity = self.intensity
        if soft_rounds is None:
            soft_rounds = self.soft_rounds
            
        # Update decoder parameters
        if steps is not None and hasattr(self.decoder, 'steps'):
            self.decoder.steps = steps
            
        if threshold is not None and hasattr(self.decoder, 'threshold'):
            self.decoder.threshold = threshold
            
        self.validate_schedule(block_length, soft_token_ratio, treat_soft_tokens_as_candidates)

        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)
            
            while (block == self.decoder.mask_id).sum() > 0:
                
                # Calculate unroll_k based on mask count and expected TPF
                unroll_k = max(min((block == self.decoder.mask_id).sum()//self.expected_tpf, self.maximum_unroll), 1)
                
                for unroll_i in range(unroll_k):
                    # Pre-check: Ensure we can satisfy the soft token ratio without violating the decoding schedule
                    # if we choose to exclude soft tokens from candidacy.
                    current_masks = (x[block_loc.start:block_loc.end] == self.decoder.mask_id).sum().item()
                    
                    # Optimization: If no masks left, stop unrolling (matches blockwise behavior)
                    if current_masks == 0:
                        break

                    num_soft = int(current_masks * soft_token_ratio)
                    
                    # Determine num_to_decode for the current step
                    num_to_decode = 0
                    if hasattr(self.decoder, 'num_transfer_tokens'):
                        # Fixed schedule
                        if self.decoder.iter < self.decoder.num_transfer_tokens.shape[1]:
                            num_to_decode = self.decoder.num_transfer_tokens[0, self.decoder.iter].item()
                    else:
                        # Dynamic schedule (Threshold decoder) - estimation not straightforward here without logits
                        pass
                    
                    if not treat_soft_tokens_as_candidates and num_to_decode > 0:
                        # If soft tokens CANNOT be decoded, we must have enough pure masks left to satisfy decoder demand
                        available_for_decoding = current_masks - num_soft
                        if available_for_decoding < num_to_decode:
                            # Log warning instead of crashing
                            logger.warning(
                                f"Decoding Schedule Violation: Step {self.decoder.iter} requires decoding {num_to_decode} tokens, "
                                f"but only {available_for_decoding} masks are available ({current_masks} total - {num_soft} soft tokens). "
                                f"Reduce soft_token_ratio or enable treat_soft_tokens_as_candidates."
                            )
                            # Adjust num_soft to make it work
                            num_soft = max(0, current_masks - num_to_decode)

                    # 1. Handle KV Cache Update (Initial step for block or periodically)
                    if kv_cache is not None and kv_cache.require_update(iter_no, block_loc.start, block_loc.end):
                        output = self.model(x.data, use_cache=True)
                        self.num_forwards += 1
                        
                        # Update cache
                        kv_cache.update(output.past_key_values)
                        self.cache_updates += 1
                        
                        # Decode using these initial logits (Standard dInfer behavior)
                        self.decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)

                    # 2. Pass 1: Standard Logits (with current masks)
                    logits1 = self._compute_logits(x, block_loc, kv_cache, use_input_embeds=None)
                    self.num_forwards += 1
                    
                    # 3. Soft Token Logic - Multi-round inference
                    # Identify masks in the current block
                    curr_block_ids = x[block_loc.start:block_loc.end]
                    mask_mask = (curr_block_ids == self.decoder.mask_id)
                    mask_indices = torch.nonzero(mask_mask).flatten()  # Indices relative to block start
                    
                    # Run soft token rounds (0 = standard blockwise, 1 = original soft token, >1 = multi-round)
                    decoding_logits, all_soft_indices = self._run_soft_token_rounds(
                        x=x,
                        block_loc=block_loc,
                        kv_cache=kv_cache,
                        logits1=logits1,
                        mask_indices=mask_indices,
                        soft_token_ratio=soft_token_ratio,
                        soft_temperature=soft_temperature,
                        intensity=intensity,
                        soft_rounds=soft_rounds
                    )

                    # 4. Decode using the latest logits
                    if not treat_soft_tokens_as_candidates and len(all_soft_indices) > 0:
                        # Combine all soft indices from all rounds
                        combined_soft_indices = torch.cat(all_soft_indices)
                        # We want to prevent these indices from being selected.
                        # Set logits for soft tokens to very low value (effectively zero probability)
                        # Supports batch size > 1 by using : for batch dimension
                        decoding_logits_modified = decoding_logits.clone()
                        decoding_logits_modified[:, combined_soft_indices] = -1000.0
                        
                        self.decoder.decode(decoding_logits_modified, block_loc.start, block_loc.end, x)
                    else:
                        self.decoder.decode(decoding_logits, block_loc.start, block_loc.end, x)
                        
                    iter_no += 1

            # Early stop at EOS - handles batch size > 1
            if self.early_stop:
                block_tokens = x[block_loc.start:block_loc.end]
                # Check which batch items have EOS in this block
                # block_tokens shape: [batch_size, block_len] or [block_len] for batch_size=1
                if block_tokens.dim() == 1:
                    # Single batch item
                    if torch.any(block_tokens == self.decoder.eos_id):
                        x[block_loc.end:] = self.decoder.eos_id
                        break
                else:
                    # Multiple batch items: check per-batch-item
                    # has_eos: [batch_size] - True if batch item has EOS in this block
                    has_eos = (block_tokens == self.decoder.eos_id).any(dim=-1)
                    if has_eos.any():
                        # Fill remaining positions with EOS only for batch items that have EOS
                        # x[block_loc.end:] shape: [batch_size, remaining_len]
                        remaining = x[block_loc.end:]
                        if remaining.numel() > 0:
                            # Expand has_eos to broadcast: [batch_size, 1]
                            remaining[has_eos] = self.decoder.eos_id
                        # Only break if ALL batch items have EOS
                        if has_eos.all():
                            break

        # DEBUG: Check for EOS tokens to explain short output
        eos_count = (x.data == self.decoder.eos_id).sum().item()
        if eos_count > 0:
            total_len = x.total_length
            logger.warning(f"SoftTokenLLM Generated {eos_count} EOS tokens out of {total_len} total positions. "
                  f"This will shorten the output by {eos_count} tokens.")

        logger.debug(f'SoftTokenLLM - Total diffusion iterations: {self.num_forwards}')
        return x.get_generated_tokens()


class SoftTokenGeneration(DInferGeneration):
    """
    dInfer Soft Token generation with FixedParallelDecoder or ThresholdParallelDecoder.
    """
    
    def __init__(self):
        super().__init__(
            name="dinfer_soft",
            description="dInfer Soft Token generation (experimental)"
        )
        self.early_stop = True  # Default value
        
    def create_diffusion_llm(self):
        """
        Create dInfer BlockWiseSoftTokenLLM.
        Uses ThresholdParallelDecoder by default (dynamic steps) but can switch to FixedParallelDecoder (fixed steps)
        """
        if not DINFER_AVAILABLE:
            raise RuntimeError("dInfer is not available")
        
        # Default to ThresholdParallelDecoder if no steps provided or steps is large
        # However, the user can override this via request parameters
        
        # For initialization, we'll use a default decoder.
        # Ideally, we should be able to swap decoders per request, but the architecture wraps the model in 
        # BlockWiseSoftTokenLLM which holds ONE decoder.
        # A workaround is to initialize with ThresholdParallelDecoder as default, or handle it in generate.
        
        # Let's use ThresholdParallelDecoder as the base, as it's more flexible.
        # BUT BlockWiseSoftTokenLLM was designed with FixedParallelDecoder in mind (validate_schedule uses steps).
        # Let's stick to FixedParallelDecoder for now as default for 'softtoken' to match original experiment,
        # but allow switching or updating threshold if available.
        
        # Actually, 'softtoken' implies we want to use soft tokens. The decoding strategy (fixed vs threshold) is separate.
        # Let's initialize with ThresholdParallelDecoder as it is generally better (faster).

        decoder = FixedParallelDecoder(
            temperature=0,
            steps=64,
            mask_id=MASK_ID,
            eos_id=EOS_ID
        )
        
        # decoder = ThresholdParallelDecoder(
        #     temperature=0,
        #     threshold=0.9,
        #     mask_id=MASK_ID,
        #     eos_id=EOS_ID
        # )
        
        # Create the Soft Token LLM
        diffusion_llm = BlockWiseSoftTokenLLM(
            model=self.model,
            decoder=decoder,
            iterator_factory=BlockIteratorFactory(),
            cache_factory=KVCacheFactory('dual'),
            early_stop=self.early_stop,
            soft_token_ratio=0.2,
            treat_soft_tokens_as_candidates=False,
            soft_temperature=0.2,
            intensity=None,  # Auto-compute intensity based on entropy
            soft_rounds=1    # Default to single round (original behavior)
        )
        
        logger.info(f"Created BlockWiseSoftTokenLLM with ThresholdParallelDecoder (early_stop={self.early_stop})")
        
        return diffusion_llm
    
    def generate(
        self,
        model: PreTrainedModel,
        prompt: torch.Tensor,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float = 0,
        remasking: bool = True,
        threshold: float = 0.9,
        factor: float = 1.0,
        soft_temperature: float = 1.0,
        early_stop: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate text using Soft Token LLM.
        """
        if self.diffusion_llm is None:
            raise RuntimeError("Diffusion LLM not created")
            
        # Extract soft token params from kwargs to avoid duplicate arguments
        soft_token_ratio = kwargs.pop('soft_token_ratio', 0.2)
        treat_soft_tokens_as_candidates = kwargs.pop('treat_soft_tokens_as_candidates', False)
        intensity = kwargs.pop('intensity', None)  # None = auto-compute based on entropy
        soft_rounds = kwargs.pop('soft_rounds', 1)  # Default to 1 (original behavior)
        
        validated_args = self.validate_args(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            soft_token_ratio=soft_token_ratio,
            treat_soft_tokens_as_candidates=treat_soft_tokens_as_candidates,
            soft_temperature=soft_temperature,
            intensity=intensity,
            soft_rounds=soft_rounds,
            **kwargs
        )
            
        # Extract soft token specific parameters from validated_args
        soft_token_ratio = validated_args.get('soft_token_ratio', 0.2)
        treat_soft_tokens_as_candidates = validated_args.get('treat_soft_tokens_as_candidates', False)
        soft_temperature = validated_args.get('soft_temperature', soft_temperature)
        intensity = validated_args.get('intensity', None)
        soft_rounds = validated_args.get('soft_rounds', 1)
            
        # Update early_stop if provided
        if early_stop is not None:
            self.diffusion_llm.early_stop = early_stop
        
        # Determine which decoder to use based on 'steps' vs 'threshold' presence?
        # Or just update the existing decoder if compatible.
        
        # If 'steps' is provided and small (e.g. < 100), user might want FixedParallelDecoder.
        # If 'threshold' is provided, user wants ThresholdParallelDecoder.
        
        # For now, we'll stick with the initialized decoder (ThresholdParallelDecoder) 
        # and just update its parameters.
        
        # Update decoder temperature
        if hasattr(self.diffusion_llm.decoder, 'temperature'):
            self.diffusion_llm.decoder.temperature = temperature
            
        # Update decoder threshold if applicable
        if hasattr(self.diffusion_llm.decoder, 'threshold'):
            self.diffusion_llm.decoder.threshold = threshold
            
        # NOTE: If we initialized with ThresholdParallelDecoder, 'steps' parameter is effectively ignored by the decoder logic
        # unless we dynamically switch the decoder class, which is complex.
        # However, the user asked to "accept ThresholdDecoder as its decoder", so using ThresholdParallelDecoder 
        # as the primary one is correct.
        
        logger.debug(f"Using dInfer Soft Token generation with args: {validated_args}")
            
        with torch.no_grad():
            output_ids = self.diffusion_llm.generate(
                prompt=prompt,
                gen_length=validated_args['gen_length'],
                block_length=validated_args['block_length'],
                soft_token_ratio=soft_token_ratio,
                treat_soft_tokens_as_candidates=treat_soft_tokens_as_candidates,
                steps=steps,
                threshold=threshold,
                soft_temperature=soft_temperature,
                intensity=intensity,
                soft_rounds=soft_rounds
            )
            
        return output_ids, self.diffusion_llm.num_forwards

    def is_available(self) -> bool:
        """Check if dInfer Soft Token generation is available."""
        return (
            DINFER_AVAILABLE and 
            ParallelDecoder is not None and
            ThresholdParallelDecoder is not None and
            TokenArray is not None
        )

    def get_required_args(self):
        """Get the required arguments."""
        return {
            'steps': 64,
            'gen_length': 256,
            'block_length': 64,
            'temperature': 0,
            'soft_token_ratio': 0.2,
            'treat_soft_tokens_as_candidates': False,
            'threshold': 0.9,
            'soft_temperature': 0.2,
            'early_stop': True,
            'intensity': None,  # None = auto-compute intensity based on entropy
            'soft_rounds': 1    # Number of soft token inference rounds (0=standard, 1=original, >1=multi-round)
        }
