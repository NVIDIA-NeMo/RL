#!/usr/bin/env python3
"""
dInfer import utilities.

This module handles the dynamic import of dInfer components from the
submodule at 3rdparty/dInfer/python.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

# Add dInfer to Python path if it exists
DINFER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../../3rdparty/dInfer/python'
)

# Initialize availability flags
DINFER_AVAILABLE = False
LLaDAModelLM = None
BlockWiseDiffusionLLM = None
SlidingWindowDiffusionLLM = None
BlockWiseDiffusionLLMWithSP = None
ThresholdParallelDecoder = None
CreditThresholdParallelDecoder = None
HierarchyDecoder = None
BlockIteratorFactory = None
KVCacheFactory = None

if os.path.exists(DINFER_PATH):
    sys.path.insert(0, DINFER_PATH)
    logger.debug(f"Added dInfer to path: {DINFER_PATH}")
    
    try:
        from dinfer.model import LLaDAModelLM as _LLaDAModelLM
        from dinfer import (
            BlockWiseDiffusionLLM as _BlockWiseDiffusionLLM,
            SlidingWindowDiffusionLLM as _SlidingWindowDiffusionLLM,
            BlockWiseDiffusionLLMWithSP as _BlockWiseDiffusionLLMWithSP,
            ThresholdParallelDecoder as _ThresholdParallelDecoder,
            CreditThresholdParallelDecoder as _CreditThresholdParallelDecoder,
            HierarchyDecoder as _HierarchyDecoder,
            BlockIteratorFactory as _BlockIteratorFactory,
            KVCacheFactory as _KVCacheFactory,
        )
        
        LLaDAModelLM = _LLaDAModelLM
        BlockWiseDiffusionLLM = _BlockWiseDiffusionLLM
        SlidingWindowDiffusionLLM = _SlidingWindowDiffusionLLM
        BlockWiseDiffusionLLMWithSP = _BlockWiseDiffusionLLMWithSP
        ThresholdParallelDecoder = _ThresholdParallelDecoder
        CreditThresholdParallelDecoder = _CreditThresholdParallelDecoder
        HierarchyDecoder = _HierarchyDecoder
        BlockIteratorFactory = _BlockIteratorFactory
        KVCacheFactory = _KVCacheFactory
        
        DINFER_AVAILABLE = True
        logger.info("dInfer components loaded successfully")
        
    except ImportError as e:
        logger.warning(f"dInfer path exists but import failed: {e}")
        logger.warning("dInfer algorithms will not be available")
        DINFER_AVAILABLE = False
else:
    logger.debug(f"dInfer path not found: {DINFER_PATH}")
    logger.debug("dInfer algorithms will not be available")
    DINFER_AVAILABLE = False

# Constants for LLaDA tokenizer
MASK_ID = 126336
EOS_ID = 126081

__all__ = [
    'DINFER_AVAILABLE',
    'LLaDAModelLM',
    'BlockWiseDiffusionLLM',
    'SlidingWindowDiffusionLLM',
    'BlockWiseDiffusionLLMWithSP',
    'ThresholdParallelDecoder',
    'CreditThresholdParallelDecoder',
    'HierarchyDecoder',
    'BlockIteratorFactory',
    'KVCacheFactory',
    'MASK_ID',
    'EOS_ID',
]
