#!/usr/bin/env python3
"""
Nemotron generation algorithm using the model's built-in generate method.
"""

import logging
import inspect
from typing import Tuple
import torch
from transformers import AutoModel, AutoModelForCausalLM, PreTrainedModel

from .base import GenerationAlgorithm

logger = logging.getLogger(__name__)


class HuggingFaceGeneration(GenerationAlgorithm):
    """Nemotron generation using the model's built-in generate method."""
    
    def __init__(self):
        super().__init__(
            name="huggingface",
            description="HuggingFace generator using native model.generate",
            engine="hf"
        )
    
    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        """
        Load model class for Nemotron generation.
        Always uses standard AutoModel for Nemotron.
        """
        logger.info("Loading HF model with standard AutoModel")
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="cpu",
            **kwargs
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
        threshold: float = 0.9,
        factor: float = 1.0,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """Generate text using Nemotron's native generate method."""
        if not self.is_available():
            raise RuntimeError("Nemotron generation is not available - model does not have native generate method")
        
        if not self._is_hf_model(model):
            raise RuntimeError("Model does not appear to be a Nemotron model with native generate method")
        
        # Validate and adjust parameters for Nemotron
        #validated_args = self.validate_args(
        #    steps=steps,
        #    gen_length=gen_length,
        #    block_length=block_length,
        #    temperature=temperature,
        #    remasking=remasking,
        #    threshold=threshold,
        #    factor=factor,
        #    **kwargs
        #)
        
        #logger.debug(f"Using Nemotron native generation with args: {validated_args}")
        
        try:
            # Call Nemotron's native generate method
            # Note: Nemotron doesn't use temperature, remasking, or factor - these are LLaDA-specific
            output_ids = model.generate(
                prompt,
                #max_new_tokens=validated_args['gen_length'],
                #steps=validated_args['steps'],
                #block_length=validated_args['block_length'],
                #threshold=validated_args['threshold'],
                #shift_logits=validated_args['shift_logits'],
                #temperature=validated_args['temperature'],
            )
            
            return output_ids, -1
            
        except Exception as e:
            logger.error(f"Nemotron generation failed: {e}")
            raise RuntimeError(f"Nemotron generation failed: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Nemotron generation is available.
        This requires checking if the currently loaded model has a native generate method.
        Since we don't have access to the model here, we return True and let the generate
        method do the actual validation.
        """
        return True
    
    def _is_hf_model(self, model: PreTrainedModel) -> bool:
        """Check if the model is a Nemotron model with native generate method."""
        if not hasattr(model, 'generate'):
            return False
        else:
            return True
        
        # Check if the generate method has the expected Nemotron signature
        try:
            sig = inspect.signature(model.generate)
            params = list(sig.parameters.keys())
            
            # Nemotron's generate method should have these parameters
            expected_params = ['max_new_tokens', 'steps', 'block_length', 'threshold', 'shift_logits']
            return all(param in params for param in expected_params)
            
        except Exception as e:
            logger.debug(f"Could not inspect model.generate signature: {e}")
            return False
    
    def get_required_args(self):
        """Get the required arguments with Nemotron-specific defaults."""
        return {}


def is_hf_model_loaded(model) -> bool:
    """Helper function to check if a HF model is currently loaded."""
    hf_gen = HuggingFaceGeneration()
    return hf_gen._is_hf_model(model) if model else False
