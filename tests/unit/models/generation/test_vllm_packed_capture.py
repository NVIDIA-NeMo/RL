# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Exercise the private capture seam with the installed vLLM frontend."""

import numpy as np
import pytest

from nemo_rl.experience.packed_logprobs import PackedLogprobBuffer, PackedLogprobCapture


@pytest.mark.parametrize("kind", ["DELTA", "CUMULATIVE", "FINAL_ONLY"])
@pytest.mark.parametrize("export", [False, True])
def test_real_vllm_output_processor_merging_and_finalization(kind, export):
    pytest.importorskip("vllm")
    from vllm.sampling_params import RequestOutputKind, SamplingParams
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
    from vllm.v1.outputs import LogprobsLists

    output_kind = RequestOutputKind[kind]
    processor = OutputProcessor(tokenizer=None, log_stats=False, stream_interval=2)
    capture = PackedLogprobCapture(processor)
    buffer = PackedLogprobBuffer(8)
    capture.register("external", buffer, export_logprobs=export)
    queue = RequestOutputCollector(output_kind, "internal")
    sampling = SamplingParams(logprobs=0, max_tokens=8, output_kind=output_kind)
    request = EngineCoreRequest(
        request_id="internal",
        external_req_id="external",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=sampling,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    processor.add_request(request, None, queue=queue)
    routes = np.zeros((5, 2, 2), dtype=np.int16)
    # A speculative multi-token batch followed by one token. Leave the queue
    # unread so vLLM itself merges/replaces emitted chunks.
    for tokens, final in [([3, 4], False), ([5], True)]:
        ids = np.asarray(tokens, dtype=np.int32).reshape(-1, 1)
        values = np.full(ids.shape, -0.25, dtype=np.float32)
        processor.process_outputs(
            [
                EngineCoreOutput(
                    request_id="internal",
                    new_token_ids=tokens,
                    new_logprobs=LogprobsLists(
                        ids, values, np.ones(len(tokens), dtype=np.int64)
                    ),
                    finish_reason=FinishReason.LENGTH if final else None,
                    routed_experts=routes if final else None,
                )
            ]
        )
    http_output = queue.get_nowait()
    assert http_output.finished
    assert http_output.outputs[0].token_ids == [3, 4, 5]
    if export:
        assert [
            entry[token].logprob
            for token, entry in zip([3, 4, 5], http_output.outputs[0].logprobs)
        ] == [-0.25] * 3
    else:
        assert http_output.outputs[0].logprobs is None
    call = buffer.extract("r", ordinal=0, weight_version=7, packing="fused")
    assert list(call.token_ids) == [1, 2, 3, 4, 5]
    assert list(call.logprobs) == [-0.25] * 3
    assert np.array_equal(call.routed_experts, routes)
    assert not processor.request_states
    capture.unregister("external")
