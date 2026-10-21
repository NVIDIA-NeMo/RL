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

"""Run ``dynamo.vllm`` with NeMo RL's engine host installed.

``python -m nemo_rl.models.generation.dynamo.engine_main <dynamo.vllm args>``
behaves as ``python -m dynamo.vllm`` except that:

- decode handlers are :class:`NemoRLDecodeWorkerHandler`, which runs NeMo RL's
  per-request hooks (``engine_host``) around Dynamo's ``generate``;
- the engine registers NeMo RL's engine routes next to Dynamo's own;
- prompt and total token overflow are owned by the engine, not the frontend:
  the frontend only sees the pre-splice prompt, so the engine clamps output
  against the prompt it actually runs and reports overflow in the form Gym
  recognizes.

ai-dynamo 1.5.1 offers no handler hook, so this rebinds three module-level
names. :func:`_check_dynamo_seams` fails at startup if a Dynamo upgrade moves
any of them.
"""

import inspect
import os
from collections.abc import AsyncIterator
from typing import Any

# As dynamo.vllm's own entry point: a fixed hash seed for engine subprocesses.
if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"


def _check_dynamo_seams() -> None:
    # Deferred: dynamo.vllm imports vLLM, which must not load before main() holds
    # restore standby.
    from dynamo.vllm import main as dynamo_main
    from dynamo.vllm import worker_factory

    for module, name in (
        (worker_factory, "DecodeWorkerHandler"),
        (dynamo_main, "WorkerFactory"),
        (dynamo_main, "publish_vllm_token_budget"),
    ):
        if not hasattr(module, name):
            raise RuntimeError(
                f"{module.__name__}.{name} is gone; NeMo RL's Dynamo engine host "
                "targets ai-dynamo 1.5.1 and must be updated for this version."
            )
    generate_params = inspect.signature(
        worker_factory.DecodeWorkerHandler.generate
    ).parameters
    route_params = inspect.signature(
        worker_factory.WorkerFactory.register_engine_routes
    ).parameters
    if "context" not in generate_params or "handler" not in route_params:
        raise RuntimeError(
            "dynamo.vllm's DecodeWorkerHandler.generate or "
            "WorkerFactory.register_engine_routes changed shape; NeMo RL's Dynamo "
            "engine host targets ai-dynamo 1.5.1 and must be updated."
        )


def _install_engine_host() -> None:
    """Rebind dynamo.vllm's handler, worker factory, and token-budget publisher."""
    # Deferred: dynamo.vllm imports vLLM, which must not load before main() holds
    # restore standby.
    from dynamo.common.token_budget import TokenBudget, publish_token_budget
    from dynamo.llm.exceptions import HttpError
    from dynamo.vllm import main as dynamo_main
    from dynamo.vllm import worker_factory

    from nemo_rl.models.generation.dynamo.engine_host import (
        SAMPLING_PROFILE_ENV,
        SET_ROLLOUT_WEIGHT_VERSION_ROUTE,
        SETUP_TOKEN_CAPTURE_ROUTE,
        EngineRequestError,
        NemoRLEngineHostMixin,
    )

    _check_dynamo_seams()

    class NemoRLDecodeWorkerHandler(
        NemoRLEngineHostMixin, worker_factory.DecodeWorkerHandler
    ):
        """Dynamo's decode handler with NeMo RL's per-request hooks."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.model_max_len is None:
                raise RuntimeError("NeMo RL's Dynamo engine host needs max_model_len")
            self._init_nemo_rl_engine_host(
                max_model_len=self.model_max_len,
                eos_token_id=self.engine_client.get_tokenizer().eos_token_id,
                sampling_profile=os.environ[SAMPLING_PROFILE_ENV],
            )

        async def generate(
            self, request: dict[str, Any], context: Any
        ) -> AsyncIterator[dict[str, Any]]:
            try:
                async for chunk in self._nemo_rl_generate(
                    request, context, super().generate
                ):
                    yield chunk
            except EngineRequestError as error:
                raise HttpError(error.status_code, error.message) from error

        def cleanup(self) -> None:
            self._nrl_executor.shutdown(wait=False, cancel_futures=True)
            super().cleanup()

    class NemoRLWorkerFactory(worker_factory.WorkerFactory):
        def register_engine_routes(
            self,
            runtime: Any,
            generate_endpoint: Any,
            handler: Any,
            lora_enabled: bool = False,
        ) -> None:
            super().register_engine_routes(
                runtime, generate_endpoint, handler, lora_enabled=lora_enabled
            )
            if isinstance(handler, NemoRLDecodeWorkerHandler):
                runtime.register_engine_route(
                    SETUP_TOKEN_CAPTURE_ROUTE, handler.nrl_setup_token_capture
                )
                runtime.register_engine_route(
                    SET_ROLLOUT_WEIGHT_VERSION_ROUTE,
                    handler.nrl_set_rollout_weight_version,
                )

    def publish_engine_owned_token_budget(
        runtime_config: Any, max_model_len: int | None
    ) -> None:
        if max_model_len is None:
            return
        publish_token_budget(
            runtime_config,
            TokenBudget(
                combined_limit=max_model_len,
                reject_prompt_overflow=False,
                reject_total_overflow=False,
            ),
        )

    worker_factory.DecodeWorkerHandler = NemoRLDecodeWorkerHandler
    dynamo_main.WorkerFactory = NemoRLWorkerFactory
    dynamo_main.publish_vllm_token_budget = publish_engine_owned_token_budget


def main() -> None:
    """Install NeMo RL's engine host, then run dynamo.vllm on this argv."""
    from dynamo.common.snapshot.restore_context import maybe_run_restore_standby_mode

    # As dynamo.vllm's entry point: hold restore standby before importing vLLM.
    maybe_run_restore_standby_mode()
    _install_engine_host()

    # Deferred until standby is held, as above.
    from dynamo.vllm.main import main as dynamo_vllm_main

    dynamo_vllm_main()


if __name__ == "__main__":
    main()
