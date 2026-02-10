# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Recipe-based configuration for NeMo-RL Megatron integration.

This module provides a clean integration with Megatron-Bridge recipes,
allowing NeMo-RL to use pre-configured training recipes as a base and
layer RL-specific settings on top.

Recipes are specified via their fully qualified Python import path in the
YAML config under ``policy.megatron_recipe``. For example:

    policy:
      megatron_recipe: megatron.bridge.recipes.llama.llama3.llama31_8b_pretrain_config
      megatron_cfg:
        ...

The import path is resolved at runtime using ``load_recipe()``.
"""

import importlib

from megatron.bridge.training.config import ConfigContainer


def load_recipe(recipe_path: str) -> ConfigContainer:
    """
    Dynamically import and call a Megatron-Bridge recipe function.

    Args:
        recipe_path: Fully qualified Python import path to the recipe function.
            For example: ``megatron.bridge.recipes.llama.llama3.llama31_8b_pretrain_config``

    Returns:
        A ConfigContainer produced by calling the recipe function.

    Raises:
        ValueError: If the recipe path is invalid or the function cannot be found.
        TypeError: If the resolved object is not callable.
    """
    module_path, _, func_name = recipe_path.rpartition(".")
    if not module_path or not func_name:
        raise ValueError(
            f"Invalid recipe path '{recipe_path}'. "
            "Expected a fully qualified Python path like "
            "'megatron.bridge.recipes.llama.llama3.llama31_8b_pretrain_config'"
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ValueError(
            f"Could not import module '{module_path}' from recipe path '{recipe_path}': {e}"
        ) from e

    recipe_fn = getattr(module, func_name, None)
    if recipe_fn is None:
        raise ValueError(
            f"Module '{module_path}' has no attribute '{func_name}'. "
            f"Check that the recipe function name is correct in '{recipe_path}'."
        )

    if not callable(recipe_fn):
        raise TypeError(
            f"'{recipe_path}' resolved to a non-callable object of type {type(recipe_fn).__name__}. "
            "Expected a recipe function."
        )

    return recipe_fn()
