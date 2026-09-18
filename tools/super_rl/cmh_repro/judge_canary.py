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
"""Native hosted-provider canary; reports accounting, never credentials."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace, MethodType

from omegaconf import OmegaConf
from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.openai_utils import (
    NeMoGymAsyncOpenAI,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.judge_verdict import response_verdict
from nemo_gym.server_utils import (
    GlobalAIOHTTPAsyncClientConfig,
    set_global_aiohttp_client,
)
from responses_api_models.inference_provider.app import (
    InferenceProvider,
    InferenceProviderConfig,
)


async def main() -> None:
    root = Path(os.environ["SUPER_RL_ROOT"])
    fragment = OmegaConf.load(
        "/opt/nemo-rl/training_configs/judges/nvidia_deepseek_v4_flash.yaml"
    )
    cfg = InferenceProviderConfig(
        name="judge_canary",
        host="127.0.0.1",
        port=8000,
        **OmegaConf.to_container(
            fragment.deepseek_v4_flash_judge_model.responses_api_models.inference_provider,
            resolve=True,
        ),
    )
    client = NeMoGymAsyncOpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    session = set_global_aiohttp_client(
        GlobalAIOHTTPAsyncClientConfig(global_aiohttp_connector_limit_per_host=16384)
    )
    receipts = []
    traces = []

    async def transport(**body):
        assert body["model"] == "nvidia/deepseek-ai/deepseek-v4-flash"
        assert body.get("max_tokens", body.get("max_completion_tokens")) == 8192
        assert body["temperature"] == body["top_p"] == 1.0
        assert body["reasoning_effort"] == "medium"
        raw = await client.create_chat_completion(**body)
        msg = raw["choices"][0]["message"]
        traces.append(
            dict(
                finish_reason=raw["choices"][0]["finish_reason"],
                reasoning_chars=len(
                    msg.get("reasoning_content") or msg.get("reasoning") or ""
                ),
                output_tokens=raw.get("usage", {}).get("completion_tokens"),
            )
        )
        return raw

    server = SimpleNamespace(
        config=cfg,
        _client=SimpleNamespace(create_chat_completion=transport),
        _converter=ResponsesConverter(
            return_token_id_information=False, uses_reasoning_parser=True
        ),
        _semaphore=asyncio.Semaphore(2),
    )
    server.chat_completions = MethodType(InferenceProvider.chat_completions, server)
    try:
        for answer, expected in (("4", True), ("5", False)):
            params = NeMoGymResponseCreateParamsNonStreaming(
                input=[
                    dict(
                        role="user",
                        content=f"Question: What is 2+2? Reference answer: 4. Candidate answer: {answer}. "
                        "Judge whether the answers are equivalent. End with exactly [[A=B]] if equal, "
                        "or [[A!=B]] if different.",
                    )
                ],
                max_output_tokens=8192,
                temperature=1.0,
                top_p=1.0,
            )
            response = await asyncio.wait_for(
                InferenceProvider.responses(server, None, params), timeout=240
            )
            actual = response_verdict(
                response, equal_label="[[A=B]]", not_equal_label="[[A!=B]]"
            )
            assert actual == expected
            assert traces[-1]["reasoning_chars"] > 0
            receipts.append(
                dict(
                    expected=expected,
                    actual=actual,
                    status=response.status,
                    **traces[-1],
                )
            )
    finally:
        await session.close()
    receipt = dict(
        complete=True,
        all_valid=True,
        reasoning_present=True,
        judge_cap=8192,
        judge_max_attempts=3,
        controls=receipts,
        scope="Two native protocol controls, not judge accuracy certification",
    )
    with (root / "manifests/judge-canary.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


asyncio.run(main())
