#!/usr/bin/env python3
"""
Nemotron generation algorithm using the model's built-in generate method.
"""

import logging
import inspect
from typing import Tuple
import torch
from transformers import AutoModel, PreTrainedModel

from .base import GenerationAlgorithm
from .utils import split_batch_across_gpus

logger = logging.getLogger(__name__)


class NemotronGeneration(GenerationAlgorithm):
    """Nemotron generation using the model's built-in generate method."""
    
    def __init__(self):
        super().__init__(
            name="nemotron",
            description="Native Nemotron diffusion generation using model's built-in generate method",
            engine="nemotron"
        )
    
    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        """
        Load model class for Nemotron generation.
        Always uses standard AutoModel for Nemotron.
        """
        logger.info("Loading Nemotron model with standard AutoModel")
        return AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
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
        
        if not self._is_nemotron_model(model):
            logger.error(f"Model validation failed!")
            logger.error(f"Model type: {type(model)}")
            logger.error(f"Model class name: {model.__class__.__name__}")
            if hasattr(model, 'module'):
                logger.error(f"Model.module type: {type(model.module)}")
                logger.error(f"Model.module class: {model.module.__class__.__name__}")
            
            unwrapped = self.unwrap_model(model)
            logger.error(f"Unwrapped model type: {type(unwrapped)}")
            logger.error(f"Unwrapped model class: {unwrapped.__class__.__name__}")
            logger.error(f"Has generate: {hasattr(unwrapped, 'generate')}")
            
            raise RuntimeError(
                f"Model does not appear to be a Nemotron model with native generate method. "
                f"Model class: {unwrapped.__class__.__name__}. "
                f"Are you trying to use Nemotron engine with a LLaDA model? "
                f"Use --engine dinfer or --engine fast-dllm for LLaDA models."
            )
        
        # Validate and adjust parameters for Nemotron
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
        
        logger.debug(f"Using Nemotron native generation with args: {validated_args}")
        
        try:
            # Use the shared multi-GPU batch splitting utility
            # This handles DataParallel batch splitting for .generate() methods
            def nemotron_generate_fn(model_instance, prompt_batch, **kwargs):
                return model_instance.generate(prompt_batch, **kwargs)
            
            output_ids, nfe = split_batch_across_gpus(
                model=model,
                prompt=prompt,
                generate_fn=nemotron_generate_fn,
                max_new_tokens=validated_args['gen_length'],
                steps=validated_args['steps'],
                block_length=validated_args['block_length'],
                threshold=validated_args['threshold'],
                shift_logits=validated_args['shift_logits']
            )
            
            return output_ids, nfe
            
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
    
    def _is_nemotron_model(self, model: PreTrainedModel) -> bool:
        """Check if the model is a Nemotron model with native generate method."""
        logger.info(f"=== NEMOTRON MODEL VALIDATION ===")
        logger.info(f"Input model type: {type(model)}")
        logger.info(f"Input model class: {model.__class__.__name__}")
        
        # Unwrap to get the base model (removes DataParallel and LeftPaddingStripWrapper)
        actual_model = self.unwrap_model(model)
        
        logger.info(f"After unwrap_model():")
        logger.info(f"  Unwrapped model type: {type(actual_model)}")
        logger.info(f"  Unwrapped model class: {actual_model.__class__.__name__}")
        
        if not hasattr(actual_model, 'generate'):
            logger.error(f"ERROR: Model does not have 'generate' method")
            return False
        
        logger.info(f"  Has 'generate' method: True")
        
        # Check if the generate method has the expected Nemotron signature
        try:
            sig = inspect.signature(actual_model.generate)
            params = list(sig.parameters.keys())
            
            logger.info(f"  generate() parameters: {params}")
            
            # Nemotron's generate method should have these parameters
            expected_params = ['max_new_tokens', 'steps', 'block_length', 'threshold', 'shift_logits']
            has_all_params = all(param in params for param in expected_params)
            
            if not has_all_params:
                missing = [p for p in expected_params if p not in params]
                logger.error(f"ERROR: Missing Nemotron parameters: {missing}")
                logger.error(f"  Expected: {expected_params}")
                logger.error(f"  Got: {params}")
            else:
                logger.info(f"  ✓ All Nemotron parameters present")
            
            return has_all_params
            
        except Exception as e:
            logger.error(f"ERROR: Could not inspect model.generate signature: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def get_required_args(self):
        """Get the required arguments with Nemotron-specific defaults."""
        return {
            'steps': 128,
            'gen_length': 128,
            'block_length': 32,
            'temperature': 1.0,  # Not used by Nemotron but kept for compatibility
            'remasking': True,   # Not used by Nemotron but kept for compatibility
            'threshold': 0.9,    # Nemotron default
            'factor': 1.0,       # Not used by Nemotron but kept for compatibility
            # TODO(mfathi): this is a hack to fix inference for our trained checkpoints using nemo-rl. need to investigate a coherent solution.
            'shift_logits': self.use_chat_template  # True when using chat template, False for raw text
        }


def is_nemotron_model_loaded(model) -> bool:
    """Helper function to check if a Nemotron model is currently loaded."""
    nemotron_gen = NemotronGeneration()
    return nemotron_gen._is_nemotron_model(model) if model else False
