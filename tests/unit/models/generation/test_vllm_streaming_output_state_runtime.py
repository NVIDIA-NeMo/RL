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

import importlib.util
import inspect
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from nemo_rl.models.generation.vllm import patches


def _run_streaming_output_state_runtime_check(tmp_path: Path) -> None:
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine import output_processor as installed_output_processor

    patched_source = tmp_path / "output_processor.py"
    shutil.copy2(Path(inspect.getfile(installed_output_processor)), patched_source)
    with patch.object(patches, "_get_vllm_file", return_value=str(patched_source)):
        assert patches._patch_vllm_streaming_session_output_state(MagicMock())

    module_name = "_nemo_rl_patched_vllm_output_processor_test"
    spec = importlib.util.spec_from_file_location(module_name, patched_source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)

        initial_params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=1,
            max_tokens=1,
            output_kind=RequestOutputKind.DELTA,
        )
        initial_request = EngineCoreRequest(
            request_id="internal",
            prompt_token_ids=[10, 11],
            mm_features=None,
            sampling_params=initial_params,
            pooling_params=None,
            arrival_time=1.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            external_req_id="external",
            resumable=True,
        )
        state = module.RequestState.from_new_request(
            tokenizer=None,
            request=initial_request,
            prompt=None,
            parent_req=None,
            request_index=0,
            queue=None,
            log_stats=False,
            stream_interval=1,
        )
        assert state.logprobs_processor.num_logprobs is None
        state.detokenizer.update([999], stop_terminated=False)
        assert state.detokenizer.output_token_ids == [999]

        intermediate_params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=1,
            max_tokens=1,
            output_kind=RequestOutputKind.DELTA,
        )
        intermediate_request = EngineCoreRequest(
            request_id="internal",
            prompt_token_ids=[12, 13],
            mm_features=None,
            sampling_params=intermediate_params,
            pooling_params=None,
            arrival_time=2.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            external_req_id="external",
            resumable=True,
        )
        state.apply_streaming_update(
            module.StreamingUpdate(
                prompt=None,
                prompt_token_ids=intermediate_request.prompt_token_ids,
                arrival_time=intermediate_request.arrival_time,
                request=intermediate_request,
            )
        )

        assert state.prompt_token_ids == [10, 11, 12, 13]
        assert intermediate_request.prompt_token_ids == [12, 13]

        final_params = SamplingParams(
            temperature=0,
            top_p=1,
            top_k=1,
            max_tokens=8,
            logprobs=0,
            output_kind=RequestOutputKind.DELTA,
        )
        final_request = EngineCoreRequest(
            request_id="internal",
            prompt_token_ids=[14, 15],
            mm_features=None,
            sampling_params=final_params,
            pooling_params=None,
            arrival_time=3.0,
            lora_request=None,
            cache_salt=None,
            data_parallel_rank=None,
            external_req_id="external",
            resumable=True,
        )
        update = module.StreamingUpdate(
            prompt=None,
            prompt_token_ids=final_request.prompt_token_ids,
            arrival_time=final_request.arrival_time,
            request=final_request,
        )

        state.apply_streaming_update(update)

        assert state.prompt_token_ids == [10, 11, 12, 13, 14, 15]
        assert final_request.prompt_token_ids == [14, 15]
        assert state.detokenizer.output_token_ids == []
        assert state.logprobs_processor.num_logprobs == 0
        assert state.logprobs_processor.logprobs == []
        assert state.max_tokens_param == 8
    finally:
        sys.modules.pop(module_name, None)


def test_streaming_update_rebuilds_output_state_for_final_sampling_params(
    tmp_path,
) -> None:
    import pytest

    pytest.importorskip("vllm")
    _run_streaming_output_state_runtime_check(tmp_path)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp_dir:
        _run_streaming_output_state_runtime_check(Path(tmp_dir))
