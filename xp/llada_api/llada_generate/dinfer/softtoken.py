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
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, soft_token_ratio=0.2, treat_soft_tokens_as_candidates=False, soft_temperature=1.0):
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
    def generate(self, prompt, gen_length=128, block_length=128, soft_token_ratio=None, treat_soft_tokens_as_candidates=None, steps=None, threshold=None, soft_temperature=None):
        ''' Generate tokens with diffusion iterations block by block using Soft Token Sampling.
        '''
        # Use instance defaults if not provided
        if soft_token_ratio is None:
            soft_token_ratio = self.soft_token_ratio
        if treat_soft_tokens_as_candidates is None:
            treat_soft_tokens_as_candidates = self.treat_soft_tokens_as_candidates
        if soft_temperature is None:
            soft_temperature = self.soft_temperature
            
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
                    
                    decoding_logits = logits1
                    soft_indices = None
                    
                    # 3. Soft Token Logic
                    # Identify masks in the current block
                    curr_block_ids = x[block_loc.start:block_loc.end]
                    mask_mask = (curr_block_ids == self.decoder.mask_id)
                    mask_indices = torch.nonzero(mask_mask).flatten() # Indices relative to block start
                    
                    if mask_indices.numel() > 0 and soft_token_ratio > 0:
                        if num_soft > 0:
                            perm = torch.randperm(mask_indices.numel(), device=self.model.device)
                            soft_indices = mask_indices[perm[:num_soft]] # Indices relative to block start
                            
                            # Extract logits for these positions
                            # logits1 shape: [1, block_len, vocab]
                            selected_logits = logits1[0, soft_indices]
                            
                            # Apply soft temperature
                            if soft_temperature > 0:
                                selected_logits = selected_logits / soft_temperature
                                
                            probs = torch.softmax(selected_logits, dim=-1)
                            
                            # Compute Soft Embeddings: Weighted average of token embeddings
                            # [num_soft, vocab] @ [vocab, d_model] -> [num_soft, d_model]
                            soft_embeds = torch.matmul(probs, self.input_embeddings.weight)
                            
                            # Prepare Input Embeddings
                            target_ids = None
                            global_offset = 0
                            
                            if kv_cache is None:
                                target_ids = x.data
                                global_offset = block_loc.start # Offset in target_ids
                            elif kv_cache.cache_type == 'prefix':
                                target_ids = x[block_loc.start:]
                                global_offset = 0 # relative to start of target_ids
                            else:
                                target_ids = curr_block_ids
                                global_offset = 0
                            
                            # Get base embeddings for the input context
                            inputs_embeds = self.input_embeddings(target_ids).clone() # [1, len, d_model]
                            
                            # Replace masks with soft embeddings
                            inputs_embeds[0, global_offset + soft_indices] = soft_embeds
                            
                            # Pass 2: Get logits with Soft Tokens
                            logits2 = self._compute_logits(x, block_loc, kv_cache, use_input_embeds=inputs_embeds)
                            self.num_forwards += 1
                            decoding_logits = logits2

                    # 4. Decode using the latest logits
                    if not treat_soft_tokens_as_candidates and soft_indices is not None and soft_indices.numel() > 0:
                        # We want to prevent these indices from being selected.
                        # Set logits for soft tokens to a uniform distribution (max entropy -> min confidence)
                        decoding_logits_modified = decoding_logits.clone()
                        decoding_logits_modified[0, soft_indices] = -1000.0 # Effectively zero probability for all tokens
                        
                        self.decoder.decode(decoding_logits_modified, block_loc.start, block_loc.end, x)
                    else:
                        self.decoder.decode(decoding_logits, block_loc.start, block_loc.end, x)
                        
                    iter_no += 1

            # Early stop at EOS
            if self.early_stop and torch.any(x[block_loc.start:block_loc.end] == self.decoder.eos_id):
                x[block_loc.end:] = self.decoder.eos_id
                break

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
        
        decoder = ThresholdParallelDecoder(
            temperature=0,
            threshold=0.9,
            mask_id=MASK_ID,
            eos_id=EOS_ID
        )
        
        # Create the Soft Token LLM
        diffusion_llm = BlockWiseSoftTokenLLM(
            model=self.model,
            decoder=decoder,
            iterator_factory=BlockIteratorFactory(),
            cache_factory=KVCacheFactory('dual'),
            early_stop=self.early_stop,
            soft_token_ratio=0.2,
            treat_soft_tokens_as_candidates=False,
            soft_temperature=1.0
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
            
        validated_args = self.validate_args(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            soft_token_ratio=kwargs.get('soft_token_ratio', 0.2),
            treat_soft_tokens_as_candidates=kwargs.get('treat_soft_tokens_as_candidates', False),
            soft_temperature=soft_temperature,
            **kwargs
        )
            
        # Extract soft token specific parameters from validated_args
        soft_token_ratio = validated_args.get('soft_token_ratio', 0.2)
        treat_soft_tokens_as_candidates = validated_args.get('treat_soft_tokens_as_candidates', False)
        soft_temperature = validated_args.get('soft_temperature', soft_temperature)
            
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
                soft_temperature=soft_temperature
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
            'gen_length': 128,
            'block_length': 128,
            'temperature': 0,
            'soft_token_ratio': 0.2,
            'treat_soft_tokens_as_candidates': False,
            'threshold': 0.9,
            'soft_temperature': 1.0,
            'early_stop': True
        }
