from pathlib import Path
from collections import defaultdict
from datetime import datetime

import os
import hashlib
import json
import logging
import numpy as np
from statistics import mean
from tqdm import tqdm
import copy
import signal
from nemo_rl.environments.code.testing_util import run_test

def prepare_tests(metadata):
    unittests = metadata['unittests']
    fn_name = metadata.get('fn_name', None)
    return {
        "input_output": {
            "inputs": [t['inputs'] for t in unittests],
            "outputs": [t['outputs'] for t in unittests],
            "fn_name": fn_name,
        },
    }

def calculate_string_md5(input_string: str):
    md5 = hashlib.md5()
    md5.update(input_string.encode('utf-8'))
    return md5.hexdigest()

def check_correctness(generation: str,
                       tests: dict,
                       timeout: int = 30,
                       debug: bool = False):
    """
    Pure-Python replacement for `check_correctness`.
    • No multiprocessing / threads
    • Safe to call from a Ray worker
    • Same output schema as the original
    """

    # reliability_guard() sets os.putenv = None; restore each call
    if not hasattr(os, "_orig_putenv"):
        os._orig_putenv = os.putenv
    os.putenv = os._orig_putenv

    # ---------- global wall-clock watchdog (identical envelope) ----------
    wall_time = (timeout + 1) * len(tests["input_output"]["inputs"]) + 5

    def _alarm_handler(signum, frame):
        raise TimeoutError

    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(wall_time)

    try:
        res, metadata = run_test(
            tests,
            test=generation,
            debug=debug,
            timeout=timeout,
        )

    except TimeoutError:
        if debug:
            print("global timeout")
        res      = [-1] * len(tests["input_output"]["inputs"])
        metadata = {"error_code": -3}

    finally:
        signal.alarm(0)            # always cancel

    # ---------- normalise results exactly like the forked original -------
    fixed = []
    for e in res:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        if e not in (True, False):
            e = False
        fixed.append(e)

    ispass = int(all(fixed))
    if debug:
        print("pass" if ispass else "fail")

    return {
        "ispass":   ispass,
        "md5":      calculate_string_md5(json.dumps(tests)),
        "results":  fixed,
        "metadata": metadata,
    }