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

"""Export an RL checkpoint trained with MTP disabled to HF format."""

import runpy
from typing import Any

from megatron.bridge import AutoBridge


_save_hf_pretrained = AutoBridge.save_hf_pretrained


def _save_no_mtp(self: Any, model: Any, path: str, *args: Any, **kwargs: Any) -> Any:
    """Export only the non-MTP model represented by the RL checkpoint."""
    config = self.hf_pretrained.config
    llm_config = getattr(config, "llm_config", None)
    if llm_config is None:
        raise RuntimeError("Expected Nemotron Omni config.llm_config")
    if getattr(llm_config, "num_nextn_predict_layers", None) is None:
        raise RuntimeError("Expected llm_config.num_nextn_predict_layers")

    llm_config.num_nextn_predict_layers = 0
    kwargs["strict"] = False
    return _save_hf_pretrained(self, model, path, *args, **kwargs)


AutoBridge.save_hf_pretrained = _save_no_mtp
runpy.run_path(
    "/opt/nemo-rl/examples/converters/convert_megatron_to_hf_gpu.py",
    run_name="__main__",
)
