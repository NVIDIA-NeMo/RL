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
"""Stage the SGLang package that ships with an Inkling 2 Mini checkpoint.

The Mini release carries its own SGLang source (``sglang.zip``). The Inkling 2 Small
image installs SGLang non-editable, so serve Mini by mounting this staged package over
the image's (``SGLANG_OVERRIDE`` in ``launch_slurm.sh``). Staging does two things the
zip lacks:

* writes ``sglang/_version.py``, which a source checkout does not generate;
* relaxes the looped-Inkling2 config check in ``srt/models/tml/config.py`` (or
  ``tml/config.py``) so it only validates an explicit ``text_config``. Transformers
  builds a bare ``Inkling2Config()`` to print a repr; that default has no
  ``loop_config`` and would otherwise raise before the server starts.

    python tools/inkling2/stage_mini_sglang.py <checkpoint>/sglang.zip <out-dir>
    SGLANG_OVERRIDE=<out-dir> bash tools/inkling2/launch_slurm.sh mini
"""

import sys
import zipfile
from pathlib import Path

OLD = "        if (self.text_config.loop_config is not None) != (\n"
NEW = (
    "        # Only validate an explicit text_config: transformers builds a bare\n"
    "        # self.__class__() to print a repr, and that default has no loop_config.\n"
    "        if text_config is not None and (self.text_config.loop_config is not None) != (\n"
)


def main(zip_path: str, out_dir: str) -> None:
    out = Path(out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    prefix = None
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            parts = name.split("/")
            # <root>/python/sglang/... -> <out>/...
            if len(parts) > 3 and parts[1] == "python" and parts[2] == "sglang":
                prefix = parts[0]
                target = out.joinpath(*parts[3:])
                if name.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
    if prefix is None:
        raise SystemExit(f"no <root>/python/sglang/ in {zip_path}")

    (out / "_version.py").write_text(
        "__version__ = version = '0.0.0+inkling2.mini'\n"
        "__version_tuple__ = version_tuple = (0, 0, 0, 'inkling2.mini')\n"
        "__commit_id__ = commit_id = None\n"
    )
    configs = list(out.glob("**/tml/config.py"))
    if len(configs) != 1:
        raise SystemExit(f"expected one tml/config.py, found {configs}")
    text = configs[0].read_text()
    if text.count(OLD) != 1:
        raise SystemExit(
            f"{configs[0]}: the loop_config check was not found exactly once; "
            "the SGLang source changed, so patch the check by hand"
        )
    configs[0].write_text(text.replace(OLD, NEW))
    print(f"staged SGLang at {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
