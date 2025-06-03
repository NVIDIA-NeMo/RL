import multiprocessing
from typing import Optional
import numpy as np


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

def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)

def check_correctness( generation: str, sample: dict, timeout: int = 30, debug: bool = False):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    # Use the dictionary directly without JSON parsing
    in_outs = sample["input_output"]

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()
    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(in_outs["inputs"]) + 5)
    if p.is_alive():
        p.kill()
    if not result:
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        metadata_list = [{"error_code": -3}]
        if debug:
            print(f"global timeout")

    res, metadata = result[0], metadata_list[0]
    fixed = []
    for e in res:
        if isinstance(e, np.ndarray):
            e = e.item(0)
        if isinstance(e, np.bool_):
            e = bool(e)
        if e != True and e != False:
            e = False
        fixed.append(e)
    res = fixed
    if not np.all(res):
        return dict(ispass=0, results=res, metadata=metadata)
    else:
        return dict(ispass=1, results=res, metadata=metadata)
