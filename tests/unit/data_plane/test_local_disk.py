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
import pytest

from nemo_rl.data_plane import local_disk

_MOUNTS = """\
/dev/md0 / ext4 rw 0 0
tmpfs /dev/shm tmpfs rw 0 0
tmpfs /run tmpfs rw 0 0
/dev/nvme0n1 /mnt/local\\040ssd xfs rw 0 0
"""


def test_filesystem_is_the_longest_mount_prefix(tmp_path):
    assert local_disk.filesystem_of("/dev/shm/x/y", _MOUNTS) == "tmpfs"
    assert local_disk.filesystem_of("/mnt/local ssd/nrl", _MOUNTS) == "xfs"
    # A path that does not exist yet resolves through its nearest ancestor.
    assert local_disk.filesystem_of("/not/yet/created", _MOUNTS) == "ext4"
    assert local_disk.filesystem_of(str(tmp_path / "new"), "") == ""


def test_ram_backed_path_is_refused_and_names_the_setting(monkeypatch):
    monkeypatch.setattr(local_disk, "filesystem_of", lambda path, text=None: "tmpfs")
    with pytest.raises(ValueError, match=r"nixl\.ssd\.path.*tmpfs"):
        local_disk.require_disk_backed("/dev/shm/nrl", setting="data_plane.nixl.ssd.path")


def test_disk_backed_path_passes(monkeypatch):
    monkeypatch.setattr(local_disk, "filesystem_of", lambda path, text=None: "ext4")
    local_disk.require_disk_backed("/tmp/nrl", setting="data_plane.nixl.ssd.path")


@pytest.mark.parametrize(
    ("raid_line", "writable", "expected"),
    [
        ("/dev/md0 /raid ext4 rw 0 0", True, "/raid/nrl_x"),  # the DGX NVMe RAID
        ("", True, "/tmp/nrl_x"),  # no /raid mount on this node (a plain dir is not one)
        ("tmpfs /raid tmpfs rw 0 0", True, "/tmp/nrl_x"),  # a /raid that is RAM is not a disk
        ("/dev/md0 /raid ext4 rw 0 0", False, "/tmp/nrl_x"),  # root-owned /raid
    ],
)
def test_default_ssd_dir_prefers_a_writable_mounted_raid(
    monkeypatch, raid_line, writable, expected
):
    # The same device as / is fine: in an enroot container / itself lives on /raid.
    monkeypatch.setattr(local_disk.os, "access", lambda p, mode: writable)
    assert local_disk.default_ssd_dir("nrl_x", _MOUNTS + raid_line) == expected


def test_the_fallback_is_noted_once_with_what_tmp_is(monkeypatch):
    # /raid absent in this process's mounts: the note names the reason and /tmp's filesystem.
    monkeypatch.setattr(local_disk, "_noted_fallback", False)
    monkeypatch.setattr(local_disk, "_mount_of", lambda p, text=None: ("/", "ext4"))
    notes = []
    log = type("Log", (), {"warning": lambda self, msg, *a: notes.append(msg % a)})()
    local_disk.note_ssd_dir_fallback_once(log)
    local_disk.note_ssd_dir_fallback_once(log)
    (note,) = notes
    assert note.startswith("/raid is not mounted on ") and "/tmp (ext4 mounted at /)" in note


def test_no_note_when_raid_is_usable(monkeypatch):
    monkeypatch.setattr(local_disk, "_noted_fallback", False)
    monkeypatch.setattr(local_disk, "_mount_of", lambda p, text=None: ("/raid", "ext4"))
    monkeypatch.setattr(local_disk.os, "access", lambda p, mode: True)
    notes = []
    local_disk.note_ssd_dir_fallback_once(
        type("Log", (), {"warning": lambda self, *a: notes.append(a)})()
    )
    assert notes == []
