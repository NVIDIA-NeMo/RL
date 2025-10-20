#!/usr/bin/env python3
"""
Base class for Fast-dLLM generation algorithms.

This module provides the common base class for all LLaDA generation algorithms
that use Fast-dLLM optimizations. It handles model loading with the optimized
LLaDAModelLM class and provides shared functionality for all Fast-dLLM variants.
"""

import logging
from abc import abstractmethod
from typing import Tuple
import torch
from transformers import AutoModel, PreTrainedModel

from ..base import GenerationAlgorithm
from ._imports import LLaDAModelLM, FAST_DLLM_AVAILABLE

logger = logging.getLogger(__name__)

# Check if LLaDAModelLM is available
FAST_DLLM_MODEL_AVAILABLE = FAST_DLLM_AVAILABLE and LLaDAModelLM is not None


class FastDLLMGeneration(GenerationAlgorithm):
    """
    Base class for Fast-dLLM generation algorithms.
    
    Engine: 'fast-dllm'
    
    This class provides common functionality for all Fast-dLLM based generation
    algorithms, including optimized model loading with LLaDAModelLM when available.
    
    Subclasses should implement:
    - The specific Fast-dLLM generation function (basic, prefix_cache, dual_cache)
    - is_available() to check if their specific generation function is available
    """
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description, engine="fast-dllm")
    
    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        """
        Load model class optimized for Fast-dLLM.
        
        Uses LLaDAModelLM if available for Fast-dLLM optimizations,
        otherwise falls back to standard AutoModel for compatibility.
        
        Args:
            model_path: Path to the model (local or HuggingFace)
            **kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model instance
        """
        if FAST_DLLM_MODEL_AVAILABLE and LLaDAModelLM is not None:
            logger.info("Loading model with Fast-dLLM optimized LLaDAModelLM class")
            return LLaDAModelLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs
            )
        else:
            logger.warning("Fast-dLLM model class not available, falling back to standard AutoModel")
            logger.warning("For optimal performance, ensure Fast-dLLM is properly installed")
            return AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                **kwargs
            )
    
    @abstractmethod
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
        """
        Generate text using Fast-dLLM.
        
        Each subclass implements this with their specific Fast-dLLM variant
        (basic, prefix_cache, or dual_cache).
        
        Args:
            model: The LLaDA model
            prompt: Input prompt tensor of shape (batch_size, seq_len)
            steps: Number of diffusion steps
            gen_length: Length of text to generate
            block_length: Block length for generation
            temperature: Sampling temperature
            remasking: Whether to use remasking
            threshold: Confidence threshold for parallel decoding
            factor: Factor for dynamic parallel decoding
            **kwargs: Additional algorithm-specific parameters
        
        Returns:
            Tuple of (generated_tokens, num_forward_passes)
        """
        pass

