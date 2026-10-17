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
import numpy as np
import pytest

pytestmark = pytest.mark.nixl


def test_write_then_read_between_two_endpoints():
    from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint

    a = NixlEndpoint("ep-a", prog_thread=False)
    b = NixlEndpoint("ep-b", prog_thread=True)
    try:
        src = np.frombuffer(bytes(range(256)) * 16, dtype=np.uint8).copy()
        dst = np.zeros_like(src)
        back = np.zeros_like(src)
        a_src = a.register(src)
        b_dst = b.register(dst)
        a_back = a.register(back)
        a.add_remote(b.metadata())

        a.transfer("WRITE", [(a_src, src.nbytes)], [(b_dst, dst.nbytes)], b.name)
        assert np.array_equal(dst, src)

        # scatter READ: two disjoint ranges in one transfer
        half = src.nbytes // 2
        a.transfer(
            "READ",
            [(a_back, half), (a_back + half, half)],
            [(b_dst, half), (b_dst + half, half)],
            b.name,
        )
        assert np.array_equal(back, src)
    finally:
        a.close()
        b.close()


def test_descriptor_validation():
    from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint

    a = NixlEndpoint("ep-v")
    try:
        with pytest.raises(ValueError):
            a.transfer("WRITE", [(0, 8)], [(0, 16)], a.name)
        with pytest.raises(ValueError):
            a.transfer("COPY", [(0, 8)], [(0, 8)], a.name)
    finally:
        a.close()


def test_file_write_and_ranged_read_loopback(tmp_path):
    """POSIX backend: DRAM → FILE write, then two disjoint FILE → DRAM range reads."""
    import os

    from nemo_rl.data_plane.nixl.nixl_io import NixlEndpoint

    ep = NixlEndpoint("ep-file", extra_backends=["POSIX"])
    try:
        src = np.frombuffer(bytes(range(256)) * 64, dtype=np.uint8).copy()  # 16 KiB
        back = np.zeros_like(src)
        a_src = ep.register(src)
        a_back = ep.register(back)
        path = tmp_path / "blob.bin"
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        os.ftruncate(fd, src.nbytes)
        ep.register_file(fd, src.nbytes)
        ep.transfer(
            "WRITE",
            [(a_src, src.nbytes)],
            [(0, src.nbytes)],
            ep.name,
            remote_mem="FILE",
            remote_dev=fd,
        )
        assert path.read_bytes() == src.tobytes()
        q = src.nbytes // 4
        ep.transfer(
            "READ",
            [(a_back, q), (a_back + 3 * q, q)],
            [(0, q), (3 * q, q)],
            ep.name,
            remote_mem="FILE",
            remote_dev=fd,
        )
        assert np.array_equal(back[:q], src[:q]) and np.array_equal(
            back[3 * q :], src[3 * q :]
        )
        assert not back[q : 3 * q].any()
        ep.deregister_file(fd)
        os.close(fd)
    finally:
        ep.close()
