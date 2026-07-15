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

from typing import Any

from nemo_rl.models.generation.vllm.patches import _get_vllm_file, _locked_file_patch


def patch_vllm_worker_nixl_preinit(
    checkpoint_engine_config: dict[str, Any] | None,
) -> None:
    """Inject early NIXL initialization into vLLM's internal worker wrapper."""
    if not (
        checkpoint_engine_config
        and checkpoint_engine_config["enabled"]
        and checkpoint_engine_config["backend"] == "nixl"
    ):
        return

    from nemo_rl.utils.checkpoint_engines.nixl import resolve_nixl_backend_kwargs

    nixl_kwargs = checkpoint_engine_config["engine_kwargs"]["nixl"]
    backend_name, backend_init_params = resolve_nixl_backend_kwargs(nixl_kwargs)
    old_snippet = (
        "        with set_current_vllm_config(self.vllm_config):\n"
        "            # To make vLLM config available during worker initialization\n"
        "            self.worker = worker_class(**kwargs)"
    )
    new_snippet = (
        "        from nemo_rl.models.generation.vllm.checkpoint_engine import "
        "maybe_preinit_nixl_for_vllm_worker\n"
        "\n"
        "        maybe_preinit_nixl_for_vllm_worker(\n"
        f"            self, backend_name={backend_name!r}, "
        f"backend_init_params={backend_init_params!r}\n"
        "        )\n"
        "\n"
        "        with set_current_vllm_config(self.vllm_config):\n"
        "            # To make vLLM config available during worker initialization\n"
        "            self.worker = worker_class(**kwargs)"
    )

    with _locked_file_patch(_get_vllm_file("v1/worker/worker_base.py")) as (
        content,
        write_back,
    ):
        if old_snippet not in content:
            return

        write_back(content.replace(old_snippet, new_snippet, 1))
