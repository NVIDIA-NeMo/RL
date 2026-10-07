import os

from ray.dashboard.subprocesses.utils import get_socket_path


def main() -> None:
    failed_session = (
        "/raid/scratch/sna/nr-align-7776207/tmp/ray/"
        "session_2026-10-07_08-56-35_792480_551183"
    )
    failed_socket_dir = f"{failed_session}/sockets"
    failed_path = f"{failed_socket_dir}/dash_MetricsHead"
    try:
        get_socket_path(failed_socket_dir, "MetricsHead")
    except OSError as exc:
        if "AF_UNIX path length cannot exceed" not in str(exc):
            raise
        print(f"PASS: reproduced long Ray socket failure ({len(failed_path)} bytes)")
        print(exc)
    else:
        raise RuntimeError("Expected the observed Ray socket path to be rejected")

    short_socket_dir = (
        f"{os.environ['RAY_TMPDIR']}/ray/"
        "session_2026-10-07_08-56-35_792480_551183/sockets"
    )
    short_path = get_socket_path(short_socket_dir, "MetricsHead")
    print(f"PASS: short Ray socket accepted ({len(short_path)} bytes): {short_path}")


if __name__ == "__main__":
    main()
