#!/usr/bin/env python3
"""
LLaDA generation algorithm with dual caching.
"""

import logging
from typing import Tuple
import torch
from transformers import PreTrainedModel

from .base import GenerationAlgorithm

logger = logging.getLogger(__name__)

# Try to import Fast-dLLM dual cache generate function
try:
    from generate import generate_with_dual_cache
    FAST_DLLM_AVAILABLE = True
except ImportError:
    generate_with_dual_cache = None
    FAST_DLLM_AVAILABLE = False


class DualCacheGeneration(GenerationAlgorithm):
    """LLaDA generation with dual caching for maximum performance."""
    
    def __init__(self):
        super().__init__(
            name="dual_cache",
            description="LLaDA generation with dual caching for optimal performance with repeated patterns"
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
        
        logger.debug(f"Using dual cache generation with args: {validated_args}")
        
        output, nfe = generate_with_dual_cache(
            model=model,
            prompt=prompt,
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
        return FAST_DLLM_AVAILABLE and generate_with_dual_cache is not None
