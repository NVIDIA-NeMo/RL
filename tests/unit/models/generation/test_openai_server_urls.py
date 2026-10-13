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
"""The backends must agree on how they report "no OpenAI server".

``dp_openai_server_base_urls`` holds one base URL per data-parallel rank. When a
backend has no HTTP server there is no per-rank slot to hold open, so the answer
is an empty list. A placeholder ``[None]`` stands for nothing and is truthy,
which makes a plain ``if not urls`` read as "we have URLs" when there are none.

Each case below exercises the branch that returns before any worker or cluster is
touched, so no GPU is needed.
"""

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.trtllm.trtllm_generation import TrtllmGeneration
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


def test_vllm_sync_engine_reports_no_urls():
    generation = SimpleNamespace(cfg={"vllm_cfg": {"async_engine": False}})

    urls = VllmGeneration._report_dp_openai_server_base_urls(generation)

    assert urls == []


def test_vllm_sync_engine_reserves_no_urls():
    generation = SimpleNamespace(cfg={"vllm_cfg": {"async_engine": False}})

    urls = VllmGeneration._collect_reserved_urls(generation)

    assert urls == []


@pytest.mark.parametrize("expose_http_server", [False, None])
def test_trtllm_without_http_server_reports_no_urls(expose_http_server):
    # dp_size is deliberately > 1: the old sentinel was [None] * dp_size, so a
    # length-carrying placeholder would survive this case.
    generation = SimpleNamespace(
        cfg={"trtllm_cfg": {"expose_http_server": expose_http_server}},
        dp_size=4,
    )

    urls = TrtllmGeneration._report_dp_openai_server_base_urls(generation)

    assert urls == []


@pytest.mark.parametrize(
    "report",
    [
        (VllmGeneration._report_dp_openai_server_base_urls, {"async_engine": False}),
        (VllmGeneration._collect_reserved_urls, {"async_engine": False}),
    ],
)
def test_no_server_is_falsy(report):
    """A plain truth test is enough to mean "no URLs"."""
    method, vllm_cfg = report
    generation = SimpleNamespace(cfg={"vllm_cfg": vllm_cfg})

    assert not method(generation)
