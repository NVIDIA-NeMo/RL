#!/usr/bin/env python3
"""
dInfer BlockWise generation with hierarchy decoder.
"""

import logging
from typing import Tuple
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
            early_stop=True
        )
        
        logger.info("Created BlockWiseDiffusionLLM with HierarchyDecoder and dual cache")
        
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

