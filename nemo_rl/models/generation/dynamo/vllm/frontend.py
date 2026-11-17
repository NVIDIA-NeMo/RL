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

"""Run Dynamo's frontend with NeMo-Gym's generation cuts served beside it.

``python -m nemo_rl.models.generation.dynamo.vllm.frontend <dynamo.frontend args>``

The arguments are ``dynamo.frontend``'s own, and they name NeMo RL's chat
engine as its chat processor. Its ``main`` builds one ``DistributedRuntime``;
this wraps that name to serve NeMo-Gym's checkpoint generation cuts through
the frontend's own runtime
(:mod:`~nemo_rl.models.generation.dynamo.vllm.generation_cut`).
"""

import sys


def main() -> None:
    import dynamo.frontend.main as frontend_main

    from nemo_rl.models.generation.dynamo.vllm.generation_cut import (
        start_generation_cut_server,
    )

    distributed_runtime = frontend_main.DistributedRuntime

    def on_the_frontends_loop(loop, *args, **kwargs):
        runtime = distributed_runtime(loop, *args, **kwargs)
        start_generation_cut_server(runtime, loop)
        return runtime

    frontend_main.DistributedRuntime = on_the_frontends_loop
    sys.argv[0] = "dynamo.frontend"
    frontend_main.main()


if __name__ == "__main__":
    main()
