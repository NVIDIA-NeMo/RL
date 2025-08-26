#!/usr/bin/env python3
"""
Base interface for LLaDA generation algorithms.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


class GenerationAlgorithm(ABC):
    """Base class for all LLaDA generation algorithms."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
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
        Generate text using the specific algorithm.
        
        Args:
            model: The LLaDA model
            prompt: Input prompt tensor of shape (batch_size, seq_len)
            steps: Number of diffusion steps
            gen_length: Length of text to generate
            block_length: Block length for generation
            temperature: Sampling temperature
            remasking: Whether to use remasking
            threshold: Threshold parameter for algorithm
            factor: Factor parameter for algorithm
            **kwargs: Additional algorithm-specific parameters
        
        Returns:
            Tuple of (generated_tokens, num_forward_passes)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this generation algorithm is available (dependencies loaded)."""
        pass
    
    def get_required_args(self) -> Dict[str, Any]:
        """Get the required arguments and their default values for this algorithm."""
        return {
            'steps': 16,
            'gen_length': 128,
            'block_length': 32,
            'temperature': 1.0,
            'remasking': True,
            'threshold': 0.95,
            'factor': 1.0
        }
    
    def validate_args(self, **kwargs) -> Dict[str, Any]:
        """Validate and set default values for generation arguments."""
        required_args = self.get_required_args()
        validated_args = {}
        
        for key, default_value in required_args.items():
            validated_args[key] = kwargs.get(key, default_value)
        
        # Ensure gen_length is divisible by block_length
        gen_length = validated_args['gen_length']
        block_length = validated_args['block_length']
        
        if gen_length % block_length != 0:
            validated_args['gen_length'] = ((gen_length // block_length) + 1) * block_length
        
        return validated_args
