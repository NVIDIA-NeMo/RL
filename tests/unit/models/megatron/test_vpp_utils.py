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

from collections.abc import Iterator
from contextlib import contextmanager

from nemo_rl.models.megatron.vpp_utils import (
    iter_model_modules,
    map_model_chunks,
    model_chunks,
    model_no_sync,
    primary_model,
    set_models_train_mode,
    zero_grad_buffer,
)


class FakeChunk:
    def __init__(self, name: str):
        self.name = name
        self.train_modes: list[bool] = []
        self.zero_grad_calls = 0
        self.no_sync_events: list[str] = []

    def train(self, training: bool) -> None:
        self.train_modes.append(training)

    def zero_grad_buffer(self) -> None:
        self.zero_grad_calls += 1

    def modules(self) -> Iterator[str]:
        yield f"{self.name}.root"
        yield f"{self.name}.child"

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        self.no_sync_events.append("enter")
        try:
            yield
        finally:
            self.no_sync_events.append("exit")


def test_model_chunk_helpers_apply_to_all_chunks() -> None:
    chunk0 = FakeChunk("chunk0")
    chunk1 = FakeChunk("chunk1")
    chunks = [chunk0, chunk1]

    assert model_chunks(chunk0) == [chunk0]
    assert model_chunks(chunks) is chunks
    assert primary_model(chunks) is chunk0

    set_models_train_mode(chunks, False)
    zero_grad_buffer(chunks)

    assert chunk0.train_modes == [False]
    assert chunk1.train_modes == [False]
    assert chunk0.zero_grad_calls == 1
    assert chunk1.zero_grad_calls == 1
    assert list(iter_model_modules(chunks)) == [
        "chunk0.root",
        "chunk0.child",
        "chunk1.root",
        "chunk1.child",
    ]


def test_model_no_sync_enters_all_chunks() -> None:
    chunk0 = FakeChunk("chunk0")
    chunk1 = FakeChunk("chunk1")

    with model_no_sync([chunk0, chunk1]):
        assert chunk0.no_sync_events == ["enter"]
        assert chunk1.no_sync_events == ["enter"]

    assert chunk0.no_sync_events == ["enter", "exit"]
    assert chunk1.no_sync_events == ["enter", "exit"]


def test_map_model_chunks_preserves_shape() -> None:
    chunk0 = FakeChunk("chunk0")
    chunk1 = FakeChunk("chunk1")

    mapped_chunk = map_model_chunks(
        chunk0, lambda chunk: FakeChunk(f"{chunk.name}.mapped")
    )
    assert mapped_chunk.name == "chunk0.mapped"
    mapped_chunks = map_model_chunks(
        [chunk0, chunk1], lambda chunk: FakeChunk(f"{chunk.name}.mapped")
    )

    assert [chunk.name for chunk in mapped_chunks] == ["chunk0.mapped", "chunk1.mapped"]
