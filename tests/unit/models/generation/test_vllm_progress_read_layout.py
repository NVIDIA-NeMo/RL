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

"""The vLLM layout generation-prefix recovery relies on, checked on the installed vLLM.

The other progress-read tests fake vLLM's version and OutputProcessor, so a vLLM
upgrade would pass them all and first break a run with ``target_level: prefix``.
These fail on the upgrade instead. A field that still exists can still change
meaning; the version pin is what sends a person to re-check.
"""

import importlib.metadata
import inspect

import pytest
from packaging.version import Version

from nemo_rl.models.generation.vllm.progress_reader import (
    _PROGRESS_READ_VLLM_RELEASE,
)

pytestmark = pytest.mark.vllm

_RECHECK = """\
Before moving _PROGRESS_READ_VLLM_RELEASE, re-read against the new vLLM:
  - VllmProgressReader.__init__ (vllm/progress_reader.py): the process_outputs
    lock wrap and the _finish_request retention hook;
  - VllmProgressReader._read_one_request: request_states, external_req_ids,
    detokenizer.output_token_ids, logprobs_processor.logprobs;
  - vllm/utils.py _flat_logprob_arrays: FlatLogprobs' arrays;
  - _bad_words_can_straddle_a_cut: SamplingParams._bad_words_token_ids;
  - the restore splice in preprocess_chat: truncate_prompt_tokens must still
    apply only while the chat is tokenized, before the splice;
then update the checks in this file."""


def test_progress_reads_are_pinned_to_the_checked_vllm_release():
    installed = Version(importlib.metadata.version("vllm")).release[:2]

    assert installed == _PROGRESS_READ_VLLM_RELEASE, (
        f"vLLM {installed} is installed; progress reads were checked on "
        f"{_PROGRESS_READ_VLLM_RELEASE}.\n{_RECHECK}"
    )


def test_output_processor_has_the_hooks_and_maps_the_reads_use():
    from vllm.v1.engine.output_processor import OutputProcessor

    assert callable(OutputProcessor.process_outputs)
    # The retention hook replaces it with a one-argument function.
    assert list(inspect.signature(OutputProcessor._finish_request).parameters) == [
        "self",
        "req_state",
    ]
    init = inspect.getsource(OutputProcessor.__init__)
    assert "self.request_states" in init
    assert "self.external_req_ids" in init


def test_engine_output_loop_calls_process_outputs_through_the_instance():
    """A bound method cached before the wrap would bypass the progress lock."""
    from vllm.v1.engine import async_llm

    source = inspect.getsource(async_llm.AsyncLLM)
    assert "output_processor = self.output_processor" in source
    assert "output_processor.process_outputs(" in source


def test_request_state_has_the_fields_a_read_takes():
    from vllm.v1.engine import detokenizer, logprobs
    from vllm.v1.engine.output_processor import RequestOutputCollector, RequestState

    state = inspect.getsource(RequestState.__init__)
    for field in (
        "self.external_req_id",
        "self.queue",
        "self.detokenizer",
        "self.logprobs_processor",
    ):
        assert field in state, field
    assert "self.output" in inspect.getsource(RequestOutputCollector)
    assert "output_token_ids" in inspect.getsource(detokenizer)
    assert "self.logprobs" in inspect.getsource(logprobs.LogprobsProcessor)


def test_flat_logprobs_keeps_the_arrays_the_read_slices():
    from vllm.logprobs import FlatLogprobs

    assert {"start_indices", "end_indices", "token_ids", "logprobs"} <= set(
        FlatLogprobs.__dataclass_fields__
    )


def test_bad_words_expand_to_token_id_lists():
    from vllm import SamplingParams

    class OneTokenPerWord:
        max_token_id = 1000

        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            return [len(word) for word in text.split()]

    params = SamplingParams(bad_words=["two words"])
    params.update_from_tokenizer(OneTokenPerWord())

    assert params._bad_words_token_ids == [[3, 5]]


def test_prompt_truncation_happens_only_while_the_chat_is_tokenized():
    """A restored call's spliced prompt must not be truncated again.

    preprocess_chat replaces the rendered prompt with the exact saved tokens
    after vLLM tokenized (and truncated) the chat, so truncate_prompt_tokens
    leaves the restored context identical to the original call's. If vLLM ever
    truncates in the engine instead, the restore could drop the start of the
    original prompt; that request setting must then block cuts.
    """
    from vllm import SamplingParams
    from vllm.renderers.params import TokenizeParams

    assert "truncate_prompt_tokens" not in SamplingParams.__struct_fields__
    assert "truncate_prompt_tokens" in TokenizeParams.__dataclass_fields__
    assert "self._token_truncation" in inspect.getsource(
        TokenizeParams._validate_tokens
    )
    assert "_validate_tokens" in inspect.getsource(
        TokenizeParams.apply_post_tokenization
    )


def test_full_generator_takes_the_arguments_a_restored_call_rewrites():
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    from nemo_rl.models.generation.prefix_recovery.restore import (
        _full_generator_signature,
    )

    _full_generator_signature(OpenAIServingChat.chat_completion_full_generator)
