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

from typing import Any


def worker_tokenizer_kwargs(
    config: dict[str, Any],
    tokenizer: Any,
    processor: Any,
    *,
    reconstruct_in_worker: bool,
) -> dict[str, Any]:
    """Build serialization-safe tokenizer arguments for a policy worker."""
    if reconstruct_in_worker:
        config["tokenizer"]["use_processor"] = processor is not None
        return {}
    return {"tokenizer": tokenizer, "processor": processor}


def resolve_worker_tokenizer(
    config: dict[str, Any], tokenizer: Any = None, processor: Any = None
) -> tuple[Any, Any]:
    """Reconstruct a tokenizer or processor inside a DTensor worker."""
    if tokenizer is not None:
        return tokenizer, processor

    from nemo_rl.algorithms.utils import get_tokenizer

    use_processor = config["tokenizer"].get("use_processor", False)
    result = get_tokenizer(config["tokenizer"], get_processor=use_processor)
    if use_processor:
        return result.tokenizer, result
    return result, processor
