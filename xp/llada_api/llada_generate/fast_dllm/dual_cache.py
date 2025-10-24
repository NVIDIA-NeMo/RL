#!/usr/bin/env python3
"""
Fast-dLLM generation with dual caching.
"""

import logging
from typing import Tuple
import torch
from transformers import PreTrainedModel

from .base import FastDLLMGeneration
from ._imports import generate_with_dual_cache, FAST_DLLM_AVAILABLE
from ..utils import split_batch_across_gpus

logger = logging.getLogger(__name__)

# Check if dual cache generation is available
GENERATION_AVAILABLE = FAST_DLLM_AVAILABLE and generate_with_dual_cache is not None


class DualCacheGeneration(FastDLLMGeneration):
    """Fast-dLLM generation with dual caching for maximum performance."""
    
    def __init__(self):
        super().__init__(
            name="dual_cache",
            description="Fast-dLLM generation with dual caching for optimal performance with repeated patterns"
        )
    
    def generate(
        self,
        model: PreTrainedModel,
        prompt: torch.Tensor,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float = 1.0,
        remasking: bool = True,
        threshold: float = 0.95,
        factor: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """Generate text using Fast-dLLM generation with dual caching."""
        if not self.is_available():
            raise RuntimeError("Fast-dLLM dual cache generation is not available")
        
        validated_args = self.validate_args(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            **kwargs
        )
        
        logger.debug(f"Using dual cache Fast-dLLM generation with args: {validated_args}")
        
        # Use the shared multi-GPU batch splitting utility
        def fast_dllm_dual_generate_fn(model_instance, prompt_batch, **kwargs):
            return generate_with_dual_cache(
                model=model_instance,
                prompt=prompt_batch,
                **kwargs
            )
        
        output, nfe = split_batch_across_gpus(
            model=model,
            prompt=prompt,
            generate_fn=fast_dllm_dual_generate_fn,
            steps=validated_args['steps'],
            gen_length=validated_args['gen_length'],
            block_length=validated_args['block_length'],
            temperature=validated_args['temperature'],
            remasking=validated_args['remasking'],
            threshold=validated_args['threshold'],
            factor=validated_args['factor']
        )
        
        return output, nfe
    
    def is_available(self) -> bool:
        """Check if Fast-dLLM dual cache generation is available."""
        return GENERATION_AVAILABLE and generate_with_dual_cache is not None

