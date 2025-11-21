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
    get_num_transfer_tokens,
    get_transfer_index,
    TokenArray,
    BlockIteratorFactory,
    KVCacheFactory,
    MASK_ID,
    EOS_ID
)

logger = logging.getLogger(__name__)


class FixedParallelDecoder(ParallelDecoder):
    """ 
    This decoder decodes tokens in a fixed number of steps.
    Adapted from _FixedParallelDecoder in soft_token_experiment.py.
    """
    def __init__(self, temperature, steps, remasking='low_confidence', mask_id=MASK_ID, eos_id=EOS_ID):
        super().__init__(temperature, remasking, mask_id)
        self.steps = steps
        self.iter = 0
        self.eos_id = eos_id

    def block_init(self, block_x, block_id):
        # TODO(zhengda) we need to handle steps correctly here when the distributed version changes the gen length.
        block_mask_index = block_x == self.mask_id
        self.num_transfer_tokens = get_num_transfer_tokens(block_mask_index, self.steps)
        self.iter = 0

    def decode(self, logits, block_start, block_end, x, iter_threshold = None):
        """ Decode the logits in a block.
        """
        mask_index = (x[block_start:block_end] == self.mask_id)
        assert mask_index.shape[1] == logits.shape[1]

        curr_x = x[block_start:block_end]
        x0, transfer_index = get_transfer_index(logits, self.temperature, self.remasking, mask_index, curr_x, self.num_transfer_tokens[:, self.iter], None)
        self.iter += 1
        x[block_start:block_end][transfer_index] = x0[transfer_index]


class BlockWiseSoftTokenLLM:
    """
    Block-wise diffusion LLM with Soft Token Sampling.
    Adapted from BlockWiseSoftTokenLLM in soft_token_experiment.py.
    """
    def __init__(self, model, decoder, iterator_factory, early_stop=True, cache_factory=None, maximum_unroll=4, expected_tpf=8, soft_token_ratio=0.2, treat_soft_tokens_as_candidates=False):
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
        self.input_embeddings = self.model.get_input_embeddings()

    def validate_schedule(self, block_length, soft_token_ratio, treat_soft_tokens_as_candidates):
        """ Validates that the decoding schedule can be satisfied with the given soft token ratio.
        """
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
    def generate(self, prompt, gen_length=128, block_length=128, soft_token_ratio=None, treat_soft_tokens_as_candidates=None, steps=None):
        ''' Generate tokens with diffusion iterations block by block using Soft Token Sampling.
        '''
        # Use instance defaults if not provided
        if soft_token_ratio is None:
            soft_token_ratio = self.soft_token_ratio
        if treat_soft_tokens_as_candidates is None:
            treat_soft_tokens_as_candidates = self.treat_soft_tokens_as_candidates
            
        # Update decoder steps if provided
        if steps is not None and hasattr(self.decoder, 'steps'):
            self.decoder.steps = steps
            
        self.validate_schedule(block_length, soft_token_ratio, treat_soft_tokens_as_candidates)

        x = TokenArray(prompt, gen_length, self.decoder.mask_id, self.decoder.eos_id, self.model.device)
        it = self.iterator_factory.create(x, block_length)

        iter_no = 0
        kv_cache = self.cache_factory.create() if self.cache_factory is not None else None
        
        for block_id, (block_loc, block) in enumerate(it):
            self.decoder.block_init(block, block_id)

            while (block == self.decoder.mask_id).sum() > 0:
                
                # Pre-check: Ensure we can satisfy the soft token ratio without violating the decoding schedule
                # if we choose to exclude soft tokens from candidacy.
                current_masks = (x[block_loc.start:block_loc.end] == self.decoder.mask_id).sum().item()
                num_soft = int(current_masks * soft_token_ratio)
                
                # The decoder wants to transfer (decode) N tokens this step
                if self.decoder.iter < self.decoder.num_transfer_tokens.shape[1]:
                    num_to_decode = self.decoder.num_transfer_tokens[0, self.decoder.iter].item()
                else:
                    num_to_decode = 0 # Should not happen usually
                
                if not treat_soft_tokens_as_candidates:
                    # If soft tokens CANNOT be decoded, we must have enough pure masks left to satisfy decoder demand
                    available_for_decoding = current_masks - num_soft
                    if available_for_decoding < num_to_decode:
                        # Log warning instead of crashing
                         logger.warning(
                            f"Decoding Schedule Violation: Step {self.decoder.iter} requires decoding {num_to_decode} tokens, "
                            f"but only {available_for_decoding} masks are available ({current_masks} total - {num_soft} soft tokens). "
                            f"Auto-adjusting soft tokens for this step."
                        )
                         # Adjust num_soft to make it work
                         num_soft = max(0, current_masks - num_to_decode)

                # Helper to run model with correct context and embeddings
                def run_model(use_input_embeds=None):
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
                             logits = self.model(block, past_key_values=past_key_values, use_cache=True,
                                              replace_position=replace_position).logits
                        return logits

                # 1. Handle KV Cache Update (Initial step for block or periodically)
                if kv_cache is not None and kv_cache.require_update(iter_no, block_loc.start, block_loc.end):
                    output = self.model(x.data, use_cache=True)
                    self.num_forwards += 1
                    
                    # Update cache
                    kv_cache.update(output.past_key_values)
                    self.cache_updates += 1
                    
                    # Decode using these initial logits
                    self.decoder.decode(output.logits[:, block_loc.start:block_loc.end], block_loc.start, block_loc.end, x)
                    iter_no += 1
                    continue

                # 2. Pass 1: Standard Logits (with current masks)
                logits1 = run_model(use_input_embeds=None)
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
                        logits2 = run_model(use_input_embeds=inputs_embeds)
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
    dInfer Soft Token generation with FixedParallelDecoder and dual cache.
    """
    
    def __init__(self):
        super().__init__(
            name="dinfer_soft",
            description="dInfer Soft Token generation (experimental)"
        )
    
    def create_diffusion_llm(self):
        """
        Create dInfer BlockWiseSoftTokenLLM.
        """
        if not DINFER_AVAILABLE:
            raise RuntimeError("dInfer is not available")
        
        # Use default steps of 64 if not specified (will be updated in generate)
        decoder = FixedParallelDecoder(
            temperature=0,
            steps=64,
            mask_id=MASK_ID,
            eos_id=EOS_ID
        )
        
        # Create the Soft Token LLM
        # Default ratio 0.2, treating soft tokens as candidates=False
        diffusion_llm = BlockWiseSoftTokenLLM(
            model=self.model,
            decoder=decoder,
            iterator_factory=BlockIteratorFactory(True), # True for random order? notebook uses True
            cache_factory=KVCacheFactory('dual'),
            early_stop=True,
            soft_token_ratio=0.2,
            treat_soft_tokens_as_candidates=False
        )
        
        logger.info("Created BlockWiseSoftTokenLLM with FixedParallelDecoder")
        
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
        threshold: float = 0.95,
        factor: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate text using Soft Token LLM.
        """
        if self.diffusion_llm is None:
            raise RuntimeError("Diffusion LLM not created")
            
        # Extract soft token specific parameters from kwargs
        soft_token_ratio = kwargs.get('soft_token_ratio', 0.2)
        treat_soft_tokens_as_candidates = kwargs.get('treat_soft_tokens_as_candidates', False)
        
        # Update decoder temperature if needed
        if hasattr(self.diffusion_llm.decoder, 'temperature'):
            self.diffusion_llm.decoder.temperature = temperature
            
        with torch.no_grad():
            output_ids = self.diffusion_llm.generate(
                prompt=prompt,
                gen_length=gen_length,
                block_length=block_length,
                soft_token_ratio=soft_token_ratio,
                treat_soft_tokens_as_candidates=treat_soft_tokens_as_candidates,
                steps=steps
            )
            
        return output_ids, self.diffusion_llm.num_forwards

    def is_available(self) -> bool:
        """Check if dInfer Soft Token generation is available."""
        return (
            DINFER_AVAILABLE and 
            ParallelDecoder is not None and
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
            'treat_soft_tokens_as_candidates': False
        }

