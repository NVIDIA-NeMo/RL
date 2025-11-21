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
ParallelDecoder = None
get_num_transfer_tokens = None
get_transfer_index = None
TokenArray = None
BlockWiseDiffusionLLM = None
VicinityCacheDiffusionLLM = None
IterSmoothDiffusionLLM = None
IterSmoothWithVicinityCacheDiffusionLLM = None
BlockWiseDiffusionLLMWithSP = None
BlockDiffusionLLMAttnmask = None
BlockDiffusionLLM = None
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
        from dinfer.decoding.parallel_strategy import (
            ParallelDecoder as _ParallelDecoder,
            get_num_transfer_tokens as _get_num_transfer_tokens,
            get_transfer_index as _get_transfer_index,
        )
        from dinfer.decoding.utils import TokenArray as _TokenArray
        from dinfer import (
            BlockWiseDiffusionLLM as _BlockWiseDiffusionLLM,
            VicinityCacheDiffusionLLM as _VicinityCacheDiffusionLLM,
            IterSmoothDiffusionLLM as _IterSmoothDiffusionLLM,
            IterSmoothWithVicinityCacheDiffusionLLM as _IterSmoothWithVicinityCacheDiffusionLLM,
            BlockWiseDiffusionLLMWithSP as _BlockWiseDiffusionLLMWithSP,
            BlockDiffusionLLMAttnmask as _BlockDiffusionLLMAttnmask,
            BlockDiffusionLLM as _BlockDiffusionLLM,
            ThresholdParallelDecoder as _ThresholdParallelDecoder,
            CreditThresholdParallelDecoder as _CreditThresholdParallelDecoder,
            HierarchyDecoder as _HierarchyDecoder,
            BlockIteratorFactory as _BlockIteratorFactory,
            KVCacheFactory as _KVCacheFactory,
        )
        
        LLaDAModelLM = _LLaDAModelLM
        ParallelDecoder = _ParallelDecoder
        get_num_transfer_tokens = _get_num_transfer_tokens
        get_transfer_index = _get_transfer_index
        TokenArray = _TokenArray
        BlockWiseDiffusionLLM = _BlockWiseDiffusionLLM
        VicinityCacheDiffusionLLM = _VicinityCacheDiffusionLLM
        IterSmoothDiffusionLLM = _IterSmoothDiffusionLLM
        IterSmoothWithVicinityCacheDiffusionLLM = _IterSmoothWithVicinityCacheDiffusionLLM
        BlockWiseDiffusionLLMWithSP = _BlockWiseDiffusionLLMWithSP
        BlockDiffusionLLMAttnmask = _BlockDiffusionLLMAttnmask
        BlockDiffusionLLM = _BlockDiffusionLLM
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
    'ParallelDecoder',
    'get_num_transfer_tokens',
    'get_transfer_index',
    'TokenArray',
    'BlockWiseDiffusionLLM',
    'VicinityCacheDiffusionLLM',
    'IterSmoothDiffusionLLM',
    'IterSmoothWithVicinityCacheDiffusionLLM',
    'BlockWiseDiffusionLLMWithSP',
    'BlockDiffusionLLMAttnmask',
    'BlockDiffusionLLM',
    'ThresholdParallelDecoder',
    'CreditThresholdParallelDecoder',
    'HierarchyDecoder',
    'BlockIteratorFactory',
    'KVCacheFactory',
    'MASK_ID',
    'EOS_ID',
]
