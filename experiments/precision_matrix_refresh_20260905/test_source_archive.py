"""Exercise the launcher's publication block without submitting GPU work."""

import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


class SourceArchiveTest(unittest.TestCase):
    def test_archive_publication(self) -> None:
        launcher = Path(__file__).with_name("submit.sh").read_text()
        start = launcher.index('if [[ "${ACTION}" == submit && ! -f "${SOURCE_ARCHIVE}" ]]; then')
        end = launcher.index('\nif [[ "${ACTION}" == submit && ! -f "${SOURCE_ARCHIVE}" ]]; then', start + 1)
        block = launcher[start:end]
        for fail_copy in (False, True):
            with self.subTest(fail_copy=fail_copy), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                repo = root / "repo"
                cache = root / "cache"
                scratch = root / "scratch"
                repo.mkdir()
                scratch.mkdir()
                (repo / "payload.txt").write_text("frozen source\n")
                archive = cache / "source.tar"
                env = dict(os.environ, ACTION="submit", REPO=str(repo),
                           SOURCE_ID="test", SOURCE_ARCHIVE_ROOT=str(cache),
                           SOURCE_ARCHIVE=str(archive), TMPDIR=str(scratch))
                setup = "set -euo pipefail\ngit() { printf 'payload.txt\\0'; }\n"
                if fail_copy:
                    setup += 'cp() { head -c 32 "$1" > "$2"; return 1; }\n'
                result = subprocess.run(["bash", "-c", setup + block], env=env,
                                        capture_output=True, text=True)
                if fail_copy:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(archive.exists())
                    self.assertEqual(list(cache.iterdir()), [])
                else:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    with tarfile.open(archive) as packed:
                        payload = packed.extractfile("payload.txt")
                        assert payload is not None
                        self.assertEqual(payload.read(), b"frozen source\n")
                    self.assertEqual(list(cache.iterdir()), [archive])
                self.assertEqual(list(scratch.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
