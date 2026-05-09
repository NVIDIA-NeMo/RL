"""Subprocess helper: do the StatelessProcessGroup init in an isolated process.

Exits 0 on success, non-zero on failure. Stderr captures the failure reason.
The parent (the actor process) reads exit-code/stderr and converts to a
clean Python exception. If this helper SIGABRTs because a peer died, the
parent process is unaffected.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--suicide-after-s", type=float, default=None,
                        help="If set, this rank kills itself after N seconds.")
    args = parser.parse_args()

    sys.path.insert(0, "/opt/nemo-rl")
    sys.path.insert(0, "/work/src")

    try:
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup
        pg = StatelessProcessGroup(
            args.master_addr, args.master_port, args.rank, args.world_size
        )
        if args.suicide_after_s is not None:
            import threading, time
            def _suicide():
                time.sleep(args.suicide_after_s)
                sys.stderr.write(f"rank={args.rank} HELPER suiciding after {args.suicide_after_s}s\n")
                sys.stderr.flush()
                os._exit(137)
            threading.Thread(target=_suicide, daemon=True).start()
        pg.init_nccl_communicator(device=0)
        # Brief drain so other ranks complete their bootstrap.
        import time
        time.sleep(0.5)
        try:
            pg.destroy()
        except Exception:
            pass
        print(f"rank={args.rank} OK")
        os._exit(0)
    except BaseException as e:
        import traceback
        sys.stderr.write(f"rank={args.rank} init failed: {type(e).__name__}: {e}\n")
        traceback.print_exc(file=sys.stderr)
        os._exit(2)


if __name__ == "__main__":
    main()
