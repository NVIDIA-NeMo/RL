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
"""Validate online W&B credentials without exposing or persisting credentials."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("WANDB_MODE") != "online" or not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Online W&B mode and exported WANDB_API_KEY are required")
    import wandb

    entity = wandb.Api(timeout=45).default_entity
    if not entity:
        raise RuntimeError("W&B authentication did not resolve an entity")
    record = {
        "status": "pass",
        "entity": entity,
        "mode": "online",
        "source_artifact_uploads": False,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
