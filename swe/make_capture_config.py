# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Derive the SWE A/B arm configs from a site yaml (see swe/SWE_RUN.md).

Usage:
    python swe/make_capture_config.py <site_yaml> <out_yaml> legacy|capture

Both arms get `env.nemo_gym.rollout_max_attempts_to_avoid_lp_nan: 1` (the
capture arm hard-errors without it; pinning it on both keeps NaN-retry
behavior out of the A/B). The capture arm additionally gets
`token_capture.enabled: true` — the gate config injection into the Gym
policy-model server happens in code (`environments/nemo_gym.py:_spinup`),
so no other yaml change is needed.
"""

import sys

import yaml


def main() -> None:
    src, dst, arm = sys.argv[1], sys.argv[2], sys.argv[3]
    assert arm in ("legacy", "capture"), f"arm must be legacy|capture, got {arm}"
    with open(src) as f:
        config = yaml.safe_load(f)

    config.setdefault("env", {}).setdefault("nemo_gym", {})[
        "rollout_max_attempts_to_avoid_lp_nan"
    ] = 1
    if arm == "capture":
        config.setdefault("token_capture", {})["enabled"] = True

    with open(dst, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[make_capture_config] arm={arm}: {src} -> {dst}")


if __name__ == "__main__":
    main()
