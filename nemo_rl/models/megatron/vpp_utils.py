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

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any, TypeVar

TModel = TypeVar("TModel")


def model_chunks(model: TModel | list[TModel]) -> list[TModel]:
    """Return Megatron model chunks as a list for non-VPP and VPP layouts."""
    return model if isinstance(model, list) else [model]


def primary_model(model: TModel | list[TModel]) -> TModel:
    """Return the first Megatron model chunk."""
    return model_chunks(model)[0]


def set_models_train_mode(model: Any | list[Any], training: bool) -> None:
    """Set train/eval mode for every Megatron model chunk."""
    for chunk in model_chunks(model):
        chunk.train(training)


def zero_grad_buffer(model: Any | list[Any]) -> None:
    """Zero grad buffers for every Megatron model chunk."""
    for chunk in model_chunks(model):
        chunk.zero_grad_buffer()


def iter_model_modules(model: Any | list[Any]) -> Iterator[Any]:
    """Yield modules from every Megatron model chunk."""
    for chunk in model_chunks(model):
        yield from chunk.modules()


@contextmanager
def model_no_sync(model: Any | list[Any]) -> Iterator[None]:
    """Enter ``no_sync`` for every Megatron model chunk."""
    with ExitStack() as stack:
        for chunk in model_chunks(model):
            stack.enter_context(chunk.no_sync())
        yield


def map_model_chunks(
    model: TModel | list[TModel], fn: Callable[[TModel], TModel]
) -> TModel | list[TModel]:
    """Apply ``fn`` to one model or every VPP chunk, preserving the input shape."""
    if isinstance(model, list):
        return [fn(chunk) for chunk in model]
    return fn(model)
