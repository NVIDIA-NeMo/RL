#!/usr/bin/env python3
"""
dInfer BlockWise generation with hierarchy decoder.
"""

import logging
from typing import Tuple, Optional
import torch
from transformers import PreTrainedModel

from .base import DInferGeneration
from ._imports import (
    DINFER_AVAILABLE,
    BlockWiseDiffusionLLM,
    HierarchyDecoder,
    BlockIteratorFactory,
    KVCacheFactory,
    MASK_ID,
    EOS_ID
)

logger = logging.getLogger(__name__)


class HierarchyGeneration(DInferGeneration):
    """
    dInfer BlockWise generation with hierarchical decoding.
    
    Uses HierarchyDecoder for enhanced parallel decoding with hierarchical structure.
    """
    
    def __init__(self):
        super().__init__(
            name="dinfer_hierarchy",
            description="dInfer BlockWise generation with hierarchical parallel decoding"
        )
        self.early_stop = True  # Default value
    
    def create_diffusion_llm(self):
        """Create dInfer BlockWiseDiffusionLLM with hierarchy decoder."""
        if not DINFER_AVAILABLE:
            raise RuntimeError("dInfer is not available")
        
        # Create hierarchy decoder (updated API)
        decoder = HierarchyDecoder(
            temperature=1.0,
            remasking='low_confidence',
            mask_id=MASK_ID,
            eos_id=EOS_ID,
            threshold=0.9,
            low_threshold=0.4
        )
        
        # Create the BlockWise diffusion LLM
        diffusion_llm = BlockWiseDiffusionLLM(
            model=self.model,
            decoder=decoder,
            iterator_factory=BlockIteratorFactory(),
            cache_factory=KVCacheFactory('dual'),
            early_stop=self.early_stop
        )
        
        logger.info(f"Created BlockWiseDiffusionLLM with HierarchyDecoder and dual cache (early_stop={self.early_stop})")
        
        return diffusion_llm
    
    def is_available(self) -> bool:
        """Check if dInfer hierarchy generation is available."""
        return (
            DINFER_AVAILABLE and 
            BlockWiseDiffusionLLM is not None and
            HierarchyDecoder is not None and
            BlockIteratorFactory is not None and
            KVCacheFactory is not None
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
        early_stop: Optional[bool] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate text using dInfer diffusion LLM.
        """
        if self.diffusion_llm is None:
            raise RuntimeError("Diffusion LLM not created. Call load_model_from_hf first.")
        
        # Update early_stop if provided
        if early_stop is not None:
            self.diffusion_llm.early_stop = early_stop
            
        # Update decoder threshold if provided (HierarchyDecoder also has threshold)
        if hasattr(self.diffusion_llm.decoder, 'threshold'):
            self.diffusion_llm.decoder.threshold = threshold
            
        # Update decoder temperature if provided
        if hasattr(self.diffusion_llm.decoder, 'temperature'):
            self.diffusion_llm.decoder.temperature = temperature
            
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
        
        logger.debug(f"Using dInfer generation with args: {validated_args}")
        
        # Generate using dInfer
        with torch.no_grad():
            output_ids = self.diffusion_llm.generate(
                prompt=prompt,
                gen_length=validated_args['gen_length'],
                block_length=validated_args['block_length']
            )
        
        # dInfer doesn't return NFE directly, estimate it
        nfe = -1
        
        return output_ids, nfe

    def get_required_args(self):
        """Get the required arguments with dInfer-specific defaults."""
        return {
            'steps': 128,
            'gen_length': 256,
            'block_length': 64,
            'temperature': 1.0,
            'remasking': True,
            'threshold': 0.9,
            'factor': 1.0,
            'early_stop': True
        }

