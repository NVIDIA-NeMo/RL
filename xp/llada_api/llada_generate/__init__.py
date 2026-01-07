#!/usr/bin/env python3
"""
LLaDA Generation Registry

This module provides a registry system for different LLaDA generation algorithms.
It allows for easy registration and discovery of generation methods.
"""

import logging
from typing import Dict, List, Optional, Type
from .base import GenerationAlgorithm

logger = logging.getLogger(__name__)


class GenerationRegistry:
    """Registry for LLaDA generation algorithms."""
    
    def __init__(self):
        self._algorithms: Dict[str, GenerationAlgorithm] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(self, algorithm: GenerationAlgorithm, aliases: Optional[List[str]] = None):
        """
        Register a generation algorithm.
        
        Args:
            algorithm: The algorithm instance to register
            aliases: Optional list of alternative names for the algorithm
        """
        if not isinstance(algorithm, GenerationAlgorithm):
            raise TypeError("Algorithm must be an instance of GenerationAlgorithm")
        
        name = algorithm.name
        if name in self._algorithms:
            logger.warning(f"Algorithm '{name}' is already registered. Overwriting.")
        
        self._algorithms[name] = algorithm
        logger.info(f"Registered generation algorithm: {name}")
        
        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    logger.warning(f"Alias '{alias}' is already registered. Overwriting.")
                self._aliases[alias] = name
                logger.debug(f"Registered alias '{alias}' -> '{name}'")
    
    def get(self, name: str) -> Optional[GenerationAlgorithm]:
        """Get a generation algorithm by name or alias."""
        # Check if it's a direct name
        if name in self._algorithms:
            return self._algorithms[name]
        
        # Check if it's an alias
        if name in self._aliases:
            actual_name = self._aliases[name]
            return self._algorithms.get(actual_name)
        
        return None
    
    def list_algorithms(self) -> List[str]:
        """List all registered algorithm names."""
        return list(self._algorithms.keys())
    
    def list_available_algorithms(self) -> List[str]:
        """List all registered algorithms that are currently available."""
        available = []
        for name, algorithm in self._algorithms.items():
            if algorithm.is_available():
                available.append(name)
        return available
    
    def get_algorithm_info(self, name: str) -> Optional[Dict[str, str]]:
        """Get information about an algorithm."""
        algorithm = self.get(name)
        if algorithm is None:
            return None
        
        return {
            'name': algorithm.name,
            'description': algorithm.description,
            'engine': algorithm.engine,
            'available': algorithm.is_available()
        }
    
    def list_algorithms_by_engine(self, engine: str) -> List[str]:
        """List all registered algorithms for a specific engine."""
        return [name for name, algo in self._algorithms.items() if algo.engine == engine]
    
    def list_available_algorithms_by_engine(self, engine: str) -> List[str]:
        """List available algorithms for a specific engine."""
        return [name for name, algo in self._algorithms.items() 
                if algo.engine == engine and algo.is_available()]
    
    def get_default_algorithm_for_engine(self, engine: str) -> Optional[str]:
        """Get a recommended default algorithm for an engine."""
        engine_defaults = {
            'fast-dllm': 'dual_cache',
            'dinfer': 'dinfer_blockwise',
            'nemotron': 'nemotron',
            'hf': 'hf'
        }
        default_name = engine_defaults.get(engine)
        if default_name and self.get(default_name):
            return default_name
        
        # Fallback: return first available algorithm for this engine
        available = self.list_available_algorithms_by_engine(engine)
        return available[0] if available else None
    
    def clear(self):
        """Clear all registered algorithms."""
        self._algorithms.clear()
        self._aliases.clear()


# Global registry instance
registry = GenerationRegistry()


def register_algorithm(algorithm: GenerationAlgorithm, aliases: Optional[List[str]] = None):
    """Register a generation algorithm in the global registry."""
    registry.register(algorithm, aliases)


def get_algorithm(name: str) -> Optional[GenerationAlgorithm]:
    """Get a generation algorithm from the global registry."""
    return registry.get(name)


def list_algorithms() -> List[str]:
    """List all registered algorithm names."""
    return registry.list_algorithms()


def list_available_algorithms() -> List[str]:
    """List all available algorithm names."""
    return registry.list_available_algorithms()


def get_algorithm_info(name: str) -> Optional[Dict[str, str]]:
    """Get information about an algorithm."""
    return registry.get_algorithm_info(name)


def list_algorithms_by_engine(engine: str) -> List[str]:
    """List all algorithms for a specific engine."""
    return registry.list_algorithms_by_engine(engine)


def list_available_algorithms_by_engine(engine: str) -> List[str]:
    """List available algorithms for a specific engine."""
    return registry.list_available_algorithms_by_engine(engine)


def get_default_algorithm_for_engine(engine: str) -> Optional[str]:
    """Get the recommended default algorithm for an engine."""
    return registry.get_default_algorithm_for_engine(engine)


# Auto-register built-in algorithms
def _register_builtin_algorithms():
    """Register all built-in generation algorithms."""
    # Register Fast-dLLM algorithms
    try:
        from .fast_dllm import BasicGeneration
        register_algorithm(BasicGeneration(), aliases=['basic', 'no_cache', 'simple'])
        logger.debug("Registered basic Fast-dLLM generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register basic generation: {e}")
    
    try:
        from .fast_dllm import PrefixCacheGeneration
        register_algorithm(PrefixCacheGeneration(), aliases=['prefix_cache', 'prefix', 'cache'])
        logger.debug("Registered prefix cache Fast-dLLM generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register prefix cache generation: {e}")
    
    try:
        from .fast_dllm import DualCacheGeneration
        register_algorithm(DualCacheGeneration(), aliases=['dual_cache', 'dual', 'double_cache'])
        logger.debug("Registered dual cache Fast-dLLM generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register dual cache generation: {e}")
    
    # Register Nemotron algorithm
    try:
        from .nemotron import NemotronGeneration
        register_algorithm(NemotronGeneration(), aliases=['nemotron', 'nemotron_native', 'nemotron_diffusion'])
        logger.debug("Registered Nemotron generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register Nemotron generation: {e}")
    
    # Register dInfer algorithms
    try:
        from .dinfer import BlockWiseGeneration
        register_algorithm(BlockWiseGeneration(), aliases=['dinfer', 'dinfer_block', 'blockwise'])
        logger.debug("Registered dInfer BlockWise generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register dInfer BlockWise generation: {e}")
    
    try:
        from .dinfer import HierarchyGeneration
        register_algorithm(HierarchyGeneration(), aliases=['dinfer_hierarchical'])
        logger.debug("Registered dInfer Hierarchy generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register dInfer Hierarchy generation: {e}")
    
    try:
        from .dinfer import CreditGeneration
        register_algorithm(CreditGeneration(), aliases=['dinfer_credit_threshold'])
        logger.debug("Registered dInfer Credit generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register dInfer Credit generation: {e}")

    try:
        from .dllmeval import DLLMEval
        register_algorithm(DLLMEval(), aliases=['dllm_eval'])
        logger.debug("Registered DLLM-Eval generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register DLLM-Eval: {e}")

    try:
        from .huggingface import HuggingFaceGeneration
        register_algorithm(HuggingFaceGeneration(), aliases=['hf', 'huggingface', 'hugging_face', 'AR', 'ar'])
        logger.debug("Registered HuggingFace generation algorithm")
    except Exception as e:
        logger.warning(f"Failed to register HuggingFace generation: {e}")


# Register built-in algorithms on import
_register_builtin_algorithms()




# Export main components
__all__ = [
    'GenerationRegistry',
    'GenerationAlgorithm',
    'registry',
    'register_algorithm',
    'get_algorithm',
    'list_algorithms',
    'list_available_algorithms',
    'get_algorithm_info',
    'list_algorithms_by_engine',
    'list_available_algorithms_by_engine',
    'get_default_algorithm_for_engine',
]
