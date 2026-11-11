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

"""Run Dynamo's frontend with NeMo RL's chat engine serving chat calls.

``python -m nemo_rl.models.generation.dynamo.vllm.frontend <dynamo.frontend args>``

The arguments are ``dynamo.frontend``'s own. Its ``main`` builds one
``EntrypointArgs``; this wraps that name to add NeMo RL's
``chat_engine_factory``, so every chat call reaches
:mod:`~nemo_rl.models.generation.dynamo.vllm.chat_engine` and never Dynamo's
Rust preprocessor. ``/v1/completions`` (NeMo RL's own rollouts, token ids in)
is unchanged. It wraps ``DistributedRuntime`` the same way, to serve NeMo-Gym's
checkpoint generation cuts through the frontend's own runtime
(:mod:`~nemo_rl.models.generation.dynamo.vllm.generation_cut`).
"""

import sys


def main() -> None:
    import dynamo.frontend.main as frontend_main

    from nemo_rl.models.generation.dynamo.vllm.chat_engine import ChatEngineFactory
    from nemo_rl.models.generation.dynamo.vllm.generation_cut import (
        start_generation_cut_server,
    )

    factory = ChatEngineFactory()
    entrypoint_args = frontend_main.EntrypointArgs

    def with_chat_engine(engine_type, **kwargs):
        if "chat_engine_factory" in kwargs:
            raise RuntimeError(
                "NeMo RL's frontend serves chat with its own chat engine; remove "
                "--dyn-chat-processor from dynamo_cfg.frontend_args.extra_cli_args"
            )
        return entrypoint_args(
            engine_type, chat_engine_factory=factory.chat_engine_factory, **kwargs
        )

    distributed_runtime = frontend_main.DistributedRuntime

    def with_generation_cuts(loop, *args, **kwargs):
        runtime = distributed_runtime(loop, *args, **kwargs)
        start_generation_cut_server(runtime, loop)
        return runtime

    frontend_main.EntrypointArgs = with_chat_engine
    frontend_main.DistributedRuntime = with_generation_cuts
    sys.argv[0] = "dynamo.frontend"
    frontend_main.main()


if __name__ == "__main__":
    main()
