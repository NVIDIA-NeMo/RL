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

import socket


def is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


# Print header
print("Size\tRange")
print("-" * 20)

start: int | None = None
for port in range(1024, 65536):
    if is_free(port):
        if start is None:
            start = port
    else:
        if start is not None:
            if start == port - 1:
                size = 1
                print(f"{size:4d}\t{start}")
            else:
                size = port - start
                print(f"{size:4d}\t{start}-{port - 1}")
            start = None

# If it ends on a free range, print it
if start is not None:
    if start == 65535:
        size = 1
        print(f"{size:4d}\t{start}")
    else:
        size = 65536 - start
        print(f"{size:4d}\t{start}-65535")
