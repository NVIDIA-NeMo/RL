#!/usr/bin/env python3
"""
Fast-dLLM generation algorithms.

This module provides LLaDA generation algorithms using Fast-dLLM optimizations.
All algorithms in this module share a common base class that handles model loading
and Fast-dLLM specific functionality.

The module automatically handles Fast-dLLM imports from the submodule at
3rdparty/Fast-dLLM/llada. If Fast-dLLM is not available, the algorithms will
gracefully report as unavailable.
"""

from ._imports import FAST_DLLM_AVAILABLE
from .base import FastDLLMGeneration
from .basic import BasicGeneration
from .prefix_cache import PrefixCacheGeneration
from .dual_cache import DualCacheGeneration

__all__ = [
    'FAST_DLLM_AVAILABLE',
    'FastDLLMGeneration',
    'BasicGeneration',
    'PrefixCacheGeneration',
    'DualCacheGeneration',
]

