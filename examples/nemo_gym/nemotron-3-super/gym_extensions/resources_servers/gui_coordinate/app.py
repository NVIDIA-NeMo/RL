# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
from typing import Any, Optional

from fastapi import FastAPI

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

POINT_PATTERN = re.compile(
    r"<point>\s*\[?\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*\]?\s*</point>"
)


class GuiCoordinateResourcesServerConfig(BaseResourcesServerConfig):
    pass


class GuiCoordinateRunRequest(BaseRunRequest):
    expected_answer: str  # "x,y" normalized 0-1 coordinates
    max_dist: float = 0.15
    metadata: Optional[dict[str, Any]] = None


class GuiCoordinateVerifyRequest(GuiCoordinateRunRequest, BaseVerifyRequest):
    pass


class GuiCoordinateVerifyResponse(BaseVerifyResponse):
    expected_answer: str
    extracted_x: Optional[float]
    extracted_y: Optional[float]
    distance: Optional[float]


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for o in body.response.output:
        if (
            getattr(o, "type", None) == "message"
            and getattr(o, "role", None) == "assistant"
        ):
            content = getattr(o, "content", None)
            if isinstance(content, list):
                for c in content:
                    t = getattr(c, "text", None)
                    if isinstance(t, str):
                        texts.append(t)
            elif isinstance(content, str):
                texts.append(content)
    return "\n".join(texts).strip()


def _parse_gt(expected_answer: str) -> Optional[tuple[float, float]]:
    parts = expected_answer.split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _parse_prediction(text: str) -> Optional[tuple[float, float]]:
    match = POINT_PATTERN.search(text)
    if not match:
        return None
    return int(match.group(1)) / 1000.0, int(match.group(2)) / 1000.0


def _compute_reward(
    gt: tuple[float, float],
    pred: tuple[float, float],
    max_dist: float,
) -> tuple[float, float]:
    dist = ((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2) ** 0.5
    if dist >= max_dist:
        return 0.0, dist
    reward = (1.0 - dist / max_dist) ** 2
    return reward, dist


class GuiCoordinateResourcesServer(SimpleResourcesServer):
    config: GuiCoordinateResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        return app

    async def verify(
        self, body: GuiCoordinateVerifyRequest
    ) -> GuiCoordinateVerifyResponse:
        text = _extract_last_assistant_text(body)
        gt = _parse_gt(body.expected_answer)
        pred = _parse_prediction(text)

        reward = 0.0
        distance = None
        extracted_x = None
        extracted_y = None

        if gt is not None and pred is not None:
            extracted_x, extracted_y = pred
            reward, distance = _compute_reward(gt, pred, body.max_dist)

        return GuiCoordinateVerifyResponse(
            **body.model_dump(
                exclude={
                    "expected_answer",
                    "extracted_x",
                    "extracted_y",
                    "distance",
                }
            ),
            reward=reward,
            expected_answer=body.expected_answer,
            extracted_x=extracted_x,
            extracted_y=extracted_y,
            distance=distance,
        )


if __name__ == "__main__":
    GuiCoordinateResourcesServer.run_webserver()
