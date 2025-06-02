
import os
import hashlib
import multiprocessing
import os
import sys
import traceback
from typing import Optional

import logging

from nemo_rl.environments.code.testing_util import run_test

import multiprocessing
import os
import sys
import traceback
from typing import Optional

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

def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    with open(os.devnull, "w") as devnull:
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            res, metadata = run_test(in_outs=sample, test=generation, debug=debug, timeout=timeout)
            result.append(res)
            metadata_list.append(metadata)
        except Exception:
            # print(e) # some tracebacks are extremely long.
            traceback.print_exc(10)
            result.append([-1 for i in range(len(sample["inputs"]))])
            metadata_list.append({})


def check_correctness(generation, in_outs: Optional[dict], timeout=10, debug=False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(in_outs, generation, debug, result, metadata_list, timeout))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
        # p.terminate()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print("global timeout")
    return result[0], metadata_list
