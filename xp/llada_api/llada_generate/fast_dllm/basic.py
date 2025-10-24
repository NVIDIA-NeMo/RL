#!/usr/bin/env python3
"""
Basic Fast-dLLM generation without caching.
"""

import logging
from typing import Tuple
import torch
from transformers import PreTrainedModel

from .base import FastDLLMGeneration
from ._imports import generate, FAST_DLLM_AVAILABLE
from ..utils import split_batch_across_gpus

logger = logging.getLogger(__name__)

# Check if basic generation is available
GENERATION_AVAILABLE = FAST_DLLM_AVAILABLE and generate is not None


class BasicGeneration(FastDLLMGeneration):
    """Basic Fast-dLLM generation without any caching mechanisms."""
    
    def __init__(self):
        super().__init__(
            name="basic",
            description="Basic Fast-dLLM generation without caching for maximum compatibility"
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
        """Generate text using basic Fast-dLLM generation."""
        if not self.is_available():
            raise RuntimeError("Fast-dLLM basic generation is not available")
        
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
        
        logger.debug(f"Using basic Fast-dLLM generation with args: {validated_args}")
        
        # Use the shared multi-GPU batch splitting utility
        def fast_dllm_generate_fn(model_instance, prompt_batch, **kwargs):
            return generate(
                model=model_instance,
                prompt=prompt_batch,
                **kwargs
            )
        
        output, nfe = split_batch_across_gpus(
            model=model,
            prompt=prompt,
            generate_fn=fast_dllm_generate_fn,
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
        """Check if Fast-dLLM basic generation is available."""
        return GENERATION_AVAILABLE and generate is not None

