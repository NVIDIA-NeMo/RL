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

from transformers import AutoConfig


def import_model_from_hf_name(hf_model_name: str, output_path: str):
    hf_config = AutoConfig.from_pretrained(hf_model_name, trust_remote_code=True)
    if hf_config.model_type == "llama":
        from nemo.tron.converter.llama import HFLlamaImporter

        print(f"Importing model {hf_model_name} to {output_path}...")
        importer = HFLlamaImporter(
            hf_model_name,
            output_path=output_path,
        )
    elif hf_config.model_type == "qwen2":
        from nemo.tron.converter.qwen import HFQwen2Importer

        print(f"Importing model {hf_model_name} to {output_path}...")
        importer = HFQwen2Importer(
            hf_model_name,
            output_path=output_path,
        )
    elif hf_config.model_type in ("deepseek_v2", "deepseek_v3"):
        from nemo.tron.converter.deepseek import HFDeepSeekImporter

        print(f"Importing model {hf_model_name} to {output_path}...")
        importer = HFDeepSeekImporter(
            hf_model_name,
            output_path=output_path,
        )
    else:
        raise ValueError(f"Unknown model_type: {hf_config.model_type}")
    importer.apply()
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

    if "llama" in hf_model_name.lower():
        from nemo.tron.converter.llama import HFLlamaExporter

        exporter_cls = HFLlamaExporter
    elif "qwen" in hf_model_name.lower():
        from nemo.tron.converter.qwen import HFQwen2Exporter

        exporter_cls = HFQwen2Exporter
    else:
        raise ValueError(
            f"Unknown model: {hf_model_name}. Currently, only Qwen2 and Llama are supported. "
            "If you'd like to run with a different model, please raise an issue or consider adding your own converter."
        )
    print(f"Exporting model {hf_model_name} to {output_path}...")
    exporter = exporter_cls(
        input_path=input_path,
        output_path=output_path,
        hf_tokenizer_path=hf_tokenizer_path,
    )
    exporter.apply()
    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()
