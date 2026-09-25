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
import json
import os

from nemo_rl.utils.train_step_file import (
    TRAIN_STEP_FILE_NAME,
    read_train_step_file,
    train_step_file_path,
    write_train_step_file,
)


def test_write_then_read_round_trips_and_is_atomic(tmp_path):
    log_dir = tmp_path / "logs" / "nested"  # created on demand
    path = write_train_step_file(str(log_dir), 12, trainer_version=12)
    assert path == train_step_file_path(str(log_dir))
    assert os.path.basename(path) == TRAIN_STEP_FILE_NAME
    payload = json.loads(open(path, encoding="utf-8").read())
    assert payload["step"] == 12 and payload["trainer_version"] == 12
    assert payload["updated_at"] > 0
    assert read_train_step_file(str(log_dir)) == 12
    # Overwrite keeps a single file (the temp file is renamed over it).
    write_train_step_file(str(log_dir), 13)
    assert sorted(os.listdir(log_dir)) == [TRAIN_STEP_FILE_NAME]
    assert read_train_step_file(str(log_dir)) == 13


def test_missing_log_dir_or_unwritable_path_never_raises(tmp_path):
    assert write_train_step_file(None, 1) is None
    assert write_train_step_file("", 1) is None
    blocker = tmp_path / "file"
    blocker.write_text("x")
    # log_dir is a regular file: makedirs/open fail -> None, no exception.
    assert write_train_step_file(str(blocker), 1) is None
    assert read_train_step_file(str(tmp_path / "absent")) is None
