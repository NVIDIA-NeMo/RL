#!/usr/bin/env python3
"""
Fast-dLLM import utilities.

This module handles the dynamic import of Fast-dLLM components from the
submodule at 3rdparty/Fast-dLLM/llada.
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

# Add Fast-dLLM to Python path if it exists
FAST_DLLM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../../../3rdparty/Fast-dLLM/llada'
)

# Initialize availability flags
FAST_DLLM_AVAILABLE = False
generate = None
generate_with_prefix_cache = None
generate_with_dual_cache = None
LLaDAModelLM = None

if os.path.exists(FAST_DLLM_PATH):
    sys.path.insert(0, FAST_DLLM_PATH)
    logger.debug(f"Added Fast-dLLM to path: {FAST_DLLM_PATH}")
    
    try:
        from generate import (
            generate as _generate,
            generate_with_prefix_cache as _generate_with_prefix_cache,
            generate_with_dual_cache as _generate_with_dual_cache
        )
        from model.modeling_llada import LLaDAModelLM as _LLaDAModelLM
        
        generate = _generate
        generate_with_prefix_cache = _generate_with_prefix_cache
        generate_with_dual_cache = _generate_with_dual_cache
        LLaDAModelLM = _LLaDAModelLM
        
        FAST_DLLM_AVAILABLE = True
        logger.info("Fast-dLLM components loaded successfully")
        
    except ImportError as e:
        logger.warning(f"Fast-dLLM path exists but import failed: {e}")
        logger.warning("Fast-dLLM algorithms will not be available")
        FAST_DLLM_AVAILABLE = False
else:
    logger.debug(f"Fast-dLLM path not found: {FAST_DLLM_PATH}")
    logger.debug("Fast-dLLM algorithms will not be available")
    FAST_DLLM_AVAILABLE = False


__all__ = [
    'FAST_DLLM_AVAILABLE',
    'generate',
    'generate_with_prefix_cache',
    'generate_with_dual_cache',
    'LLaDAModelLM',
]

