#!/usr/bin/env python3
"""
dInfer BlockWise generation with credit threshold decoder.
"""

import logging
from typing import Tuple
import torch
from transformers import PreTrainedModel

from .base import DInferGeneration
from ._imports import (
    DINFER_AVAILABLE,
    BlockWiseDiffusionLLM,
    CreditThresholdParallelDecoder,
    BlockIteratorFactory,
    KVCacheFactory,
    MASK_ID,
    EOS_ID
)

logger = logging.getLogger(__name__)


class CreditGeneration(DInferGeneration):
    """
    dInfer BlockWise generation with credit threshold decoding.
    
    Uses CreditThresholdParallelDecoder for credit-based parallel decoding strategy.
    """
    
    def __init__(self):
        super().__init__(
            name="dinfer_credit",
            description="dInfer BlockWise generation with credit threshold parallel decoding"
        )
    
    def create_diffusion_llm(self):
        """Create dInfer BlockWiseDiffusionLLM with credit threshold decoder."""
        if not DINFER_AVAILABLE:
            raise RuntimeError("dInfer is not available")
        
        # Create credit threshold decoder
        decoder = CreditThresholdParallelDecoder(
            temperature=1.0,
            threshold=0.9,
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
        
        logger.info("Created BlockWiseDiffusionLLM with CreditThresholdParallelDecoder and dual cache")
        
        return diffusion_llm
    
    def is_available(self) -> bool:
        """Check if dInfer credit generation is available."""
        return (
            DINFER_AVAILABLE and 
            BlockWiseDiffusionLLM is not None and
            CreditThresholdParallelDecoder is not None and
            BlockIteratorFactory is not None and
            KVCacheFactory is not None
        )

