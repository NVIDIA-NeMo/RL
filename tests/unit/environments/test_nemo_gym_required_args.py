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

import json

from nemo_rl.environments.nemo_gym import _missing_required_tool_arguments

REQ = {"str_replace_editor": ["command", "path"], "execute_bash": ["command"]}


def _call(name, args):
    return {"type": "function_call", "name": name, "arguments": json.dumps(args)}


def test_flags_missing_required_argument():
    assert _missing_required_tool_arguments(
        _call("str_replace_editor", {"command": "view"}), REQ
    )
    assert _missing_required_tool_arguments(
        _call("str_replace_editor", {"command": "str_replace", "new_str": "x"}), REQ
    )
    assert _missing_required_tool_arguments(_call("execute_bash", {}), REQ)
    assert _missing_required_tool_arguments(
        _call("execute_bash", {"command": ""}), REQ
    )


def test_accepts_complete_or_unconfigured_calls():
    assert not _missing_required_tool_arguments(
        _call("str_replace_editor", {"command": "view", "path": "/a"}), REQ
    )
    assert not _missing_required_tool_arguments(_call("finish", {}), REQ)
    assert not _missing_required_tool_arguments(_call("think", {}), REQ)
    assert not _missing_required_tool_arguments(
        {"type": "message", "content": [{"text": "hi"}]}, REQ
    )
    assert not _missing_required_tool_arguments(_call("execute_bash", {}), None)


def test_unparseable_arguments_are_flagged():
    item = {"type": "function_call", "name": "execute_bash", "arguments": "{bad"}
    assert _missing_required_tool_arguments(item, REQ)
