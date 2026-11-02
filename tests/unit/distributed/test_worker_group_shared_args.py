# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo_rl.distributed.worker_groups import _share_large_arguments


class _Counter:
    """A stand-in for ``ray.put`` that records what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, value):
        self.calls.append(value)
        return f"ref-{len(self.calls)}"


def _size(value):
    return len(value) if isinstance(value, (str, bytes)) else 8


def test_large_argument_is_put_once_and_replaced_by_its_reference():
    put = _Counter()
    big = "x" * 100
    args, kwargs, refs = _share_large_arguments(
        ("class.fqn", {"a": 1}),
        {"tokenizer": big, "init_optimizer": True},
        threshold_bytes=50,
        put=put,
        serialized_size=_size,
    )
    assert put.calls == [big]
    assert kwargs == {"tokenizer": "ref-1", "init_optimizer": True}
    assert args == ("class.fqn", {"a": 1})
    assert refs == ["ref-1"]


def test_arguments_at_or_below_the_threshold_are_passed_through_unchanged():
    put = _Counter()
    obj = {"k": "v"}
    args, kwargs, refs = _share_large_arguments(
        (obj, "x" * 50), {"b": b"y" * 10}, threshold_bytes=50, put=put, serialized_size=_size
    )
    assert put.calls == []
    assert refs == []
    assert args[0] is obj
    assert args[1] == "x" * 50
    assert kwargs == {"b": b"y" * 10}


def test_two_large_arguments_get_one_reference_each():
    put = _Counter()
    args, kwargs, refs = _share_large_arguments(
        ("a" * 100,), {"b": "b" * 100}, threshold_bytes=50, put=put, serialized_size=_size
    )
    assert args == ("ref-1",)
    assert kwargs == {"b": "ref-2"}
    assert refs == ["ref-1", "ref-2"]


def test_argument_that_cannot_be_sized_is_left_for_the_callers_own_serialization():
    put = _Counter()

    def unsizable(value):
        raise TypeError("cannot pickle")

    thing = object()
    args, kwargs, refs = _share_large_arguments(
        (thing,), {"k": thing}, threshold_bytes=1, put=put, serialized_size=unsizable
    )
    assert args[0] is thing
    assert kwargs["k"] is thing
    assert put.calls == []
    assert refs == []


def test_default_size_check_shares_a_large_real_object():
    put = _Counter()
    big = list(range(200_000))  # pickles to ~1 MB+
    small = {"a": 1}
    args, kwargs, refs = _share_large_arguments(
        (small,), {"big": big}, threshold_bytes=100_000, put=put
    )
    assert args[0] is small
    assert kwargs["big"] == "ref-1"
    assert put.calls == [big]
