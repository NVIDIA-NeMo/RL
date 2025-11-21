#!/usr/bin/env python3
"""
dInfer BlockWise generation with dual cache and threshold decoder.
"""

import logging
from typing import Tuple
import torch
from transformers import PreTrainedModel

from .base import DInferGeneration
from ._imports import (
    DINFER_AVAILABLE,
    BlockWiseDiffusionLLM,
    ThresholdParallelDecoder,
    BlockIteratorFactory,
    KVCacheFactory,
    MASK_ID,
    EOS_ID
)
from ._utils import FixedParallelDecoder

logger = logging.getLogger(__name__)


class BlockWiseGeneration(DInferGeneration):
    """
    dInfer BlockWise generation with threshold decoder and dual cache.
    
    This is the standard dInfer algorithm that provides:
    - Block-wise diffusion iteration
    - Threshold-based parallel decoding
    - Dual KV-cache management
    - Early stopping
    """
    
    def __init__(self):
        super().__init__(
            name="dinfer_blockwise",
            description="dInfer BlockWise generation with threshold decoder and dual cache (10x+ faster than Fast-dLLM)"
        )
    
    def create_diffusion_llm(self):
        """
        Create dInfer BlockWiseDiffusionLLM with threshold decoder.
        
        Based on the working example, uses:
        - ThresholdParallelDecoder with temperature=1.0, threshold=0.9
        - BlockIteratorFactory (default)
        - KVCacheFactory with 'dual' cache
        - early_stop=True for efficiency
        """
        if not DINFER_AVAILABLE:
            raise RuntimeError("dInfer is not available")
        
        # Use default steps of 64 if not specified (will be updated in generate)
        decoder = FixedParallelDecoder(
            temperature=0,
            steps=64,
            mask_id=MASK_ID,
            eos_id=EOS_ID
        )
        
        # Create the BlockWise diffusion LLM
        diffusion_llm = BlockWiseDiffusionLLM(
            model=self.model,
            decoder=decoder,
            iterator_factory=BlockIteratorFactory(),
            cache_factory=KVCacheFactory('dual'),
            early_stop=True
        )
        
        logger.info("Created BlockWiseDiffusionLLM with ThresholdParallelDecoder and dual cache")
        
        return diffusion_llm
    
    def is_available(self) -> bool:
        """Check if dInfer BlockWise generation is available."""
        return (
            DINFER_AVAILABLE and 
            BlockWiseDiffusionLLM is not None and
            ThresholdParallelDecoder is not None and
            BlockIteratorFactory is not None and
            KVCacheFactory is not None
        )

