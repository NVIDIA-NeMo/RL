# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
from megatron.bridge import CausalLMBridge

def import_model_from_hf_name(hf_model_name: str, output_path: str):
    bridge = CausalLMBridge.from_hf_pretrained(hf_model_name)
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)
    bridge.save_megatron_model(megatron_model, output_path)

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()


def export_model_from_megatron(
    hf_model_name: str,
    input_path: str,
    output_path: str,
    hf_tokenizer_path: str,
    overwrite: bool = False,
):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {output_path}. Delete it to run or set overwrite=True."
        )

    bridge = CausalLMBridge.from_hf_pretrained(hf_model_name)
    megatron_model = bridge.load_megatron_model(input_path)
    bridge.save_hf_pretrained(megatron_model, output_path)

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()
