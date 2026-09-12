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
"""Tests for the UV_CACHE_DIR warning emitted on import of nemo_rl."""

import logging
import os
from unittest.mock import patch

from nemo_rl import _warn_if_uv_cache_dir_set


class TestWarnIfUvCacheDirSet:
    """Test cases for the _warn_if_uv_cache_dir_set function."""

    def test_no_warning_when_unset(self, caplog):
        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                _warn_if_uv_cache_dir_set()

        assert caplog.records == []

    def test_no_warning_when_empty(self, caplog):
        with patch.dict(os.environ, {"UV_CACHE_DIR": ""}):
            with caplog.at_level(logging.WARNING):
                _warn_if_uv_cache_dir_set()

        assert caplog.records == []

    def test_warns_with_path_and_remedy(self, caplog):
        with patch.dict(os.environ, {"UV_CACHE_DIR": "/shared/fs/uv"}):
            with caplog.at_level(logging.WARNING):
                _warn_if_uv_cache_dir_set()

        assert len(caplog.records) == 1
        message = caplog.records[0].message
        assert "/shared/fs/uv" in message
        assert "unset UV_CACHE_DIR" in message
