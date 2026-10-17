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

"""Validate a resolved ``dynamo.vllm`` argv without starting an engine."""

import dataclasses
import json
import sys

# Fields NeMo RL sends in each ``update_weights`` round (see refit.py).
_UPDATE_INFO_FIELDS = {"names", "dtype_names", "shapes"}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: validate_dynamo_vllm_args.py '<json argv>' '<json packed geometry>'"
        )

    # Parser construction evaluates vLLM device defaults. Image builds and
    # unit preflights may run on CPU-only Slurm nodes, where a CUDA wheel leaves
    # current_platform unspecified even though no engine will be constructed.
    # Match Dynamo frontend's parser-only fallback without changing GPU runs.
    import vllm.platforms

    if vllm.platforms.current_platform.device_type == "":
        from vllm.platforms.cpu import CpuPlatform

        vllm.platforms.current_platform = CpuPlatform()

    from dynamo.vllm.args import parse_args
    from vllm.distributed.weight_transfer.nccl_common import (
        NCCLWeightTransferInitInfo,
    )
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLWeightTransferUpdateInfo,
    )

    argv = json.loads(sys.argv[1])
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise TypeError("resolved Dynamo argv must be a JSON list of strings")
    parse_args(argv)

    # NeMo RL declares the packed geometry in the init info and sends only
    # per-round tensor metadata in the update info. Fail before engine startup
    # if the installed vLLM weight-transfer API no longer accepts that shape.
    geometry = json.loads(sys.argv[2])
    try:
        NCCLWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=1,
            rank_offset=1,
            world_size=2,
            packed=True,
            packed_buffer_size_bytes=geometry["buffer_size_bytes"],
            packed_num_buffers=geometry["num_buffers"],
        )
    except TypeError as error:
        raise SystemExit(
            "vLLM NCCLWeightTransferInitInfo does not accept NeMo RL's packed "
            f"init info: {error}. Update the Dynamo refit protocol in refit.py."
        ) from error
    update_fields = {
        field.name for field in dataclasses.fields(NCCLWeightTransferUpdateInfo)
    }
    if update_fields != _UPDATE_INFO_FIELDS:
        raise SystemExit(
            "vLLM NCCLWeightTransferUpdateInfo fields changed: engine has "
            f"{sorted(update_fields)}, NeMo RL sends {sorted(_UPDATE_INFO_FIELDS)}. "
            "Update the Dynamo refit protocol in refit.py."
        )


if __name__ == "__main__":
    main()
