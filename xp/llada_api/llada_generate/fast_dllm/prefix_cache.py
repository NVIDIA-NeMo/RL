#!/usr/bin/env python3
"""
Fast-dLLM generation with prefix caching.
"""

import logging
from typing import Tuple
import torch
from transformers import PreTrainedModel

from .base import FastDLLMGeneration
from ._imports import generate_with_prefix_cache, FAST_DLLM_AVAILABLE
from ..utils import split_batch_across_gpus

logger = logging.getLogger(__name__)

# Check if prefix cache generation is available
GENERATION_AVAILABLE = FAST_DLLM_AVAILABLE and generate_with_prefix_cache is not None


class PrefixCacheGeneration(FastDLLMGeneration):
    """Fast-dLLM generation with prefix caching for improved efficiency."""
    
    def __init__(self):
        super().__init__(
            name="prefix_cache",
            description="Fast-dLLM generation with prefix caching to accelerate repeated prompt prefixes"
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
        """Generate text using Fast-dLLM generation with prefix caching."""
        if not self.is_available():
            raise RuntimeError("Fast-dLLM prefix cache generation is not available")
        
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
        
        logger.debug(f"Using prefix cache Fast-dLLM generation with args: {validated_args}")
        
        # Use the shared multi-GPU batch splitting utility
        def fast_dllm_prefix_generate_fn(model_instance, prompt_batch, **kwargs):
            return generate_with_prefix_cache(
                model=model_instance,
                prompt=prompt_batch,
                **kwargs
            )
        
        output, nfe = split_batch_across_gpus(
            model=model,
            prompt=prompt,
            generate_fn=fast_dllm_prefix_generate_fn,
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
        """Check if Fast-dLLM prefix cache generation is available."""
        return GENERATION_AVAILABLE and generate_with_prefix_cache is not None

