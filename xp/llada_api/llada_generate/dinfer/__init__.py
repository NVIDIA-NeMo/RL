#!/usr/bin/env python3
"""
dInfer generation algorithms.

This module provides LLaDA generation algorithms using dInfer optimizations.
dInfer is a high-performance inference framework for diffusion language models
that provides 10x+ speedup over Fast-dLLM with maintained accuracy.

The module automatically handles dInfer imports from the submodule at
3rdparty/dInfer/python. If dInfer is not available, the algorithms will
gracefully report as unavailable.
"""

from ._imports import DINFER_AVAILABLE
from .base import DInferGeneration
from .blockwise import BlockWiseGeneration
from .hierarchy import HierarchyGeneration
from .credit import CreditGeneration
from .softtoken import SoftTokenGeneration

__all__ = [
    'DINFER_AVAILABLE',
    'DInferGeneration',
    'BlockWiseGeneration',
    'HierarchyGeneration',
    'CreditGeneration',
    'SoftTokenGeneration',
]

