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
import json
import os
import time

import pytest
import ray

from nemo_rl.environments.code_environment import CodeEnvironment


@pytest.fixture(scope="module")
def code_env():
    """Create a CodeEnvironment actor for testing."""
    env = CodeEnvironment.options(
        runtime_env={
            "py_executable": CodeEnvironment.DEFAULT_PY_EXECUTABLE,
            "env_vars": dict(os.environ),
        }
    ).remote({"num_workers": 2, "timeout": 10})
    yield env
    # Clean up the actor and wait for it to be killed
    env.shutdown.remote()
    ray.kill(env)
    # Give some time for cleanup
    time.sleep(0.1)


@pytest.fixture
def chess_king_test_data():
    """Test data for chess king problem with comprehensive test cases."""
    inputs = [
        "5 7 6 11\n3\n5 3 8\n6 7 11\n5 2 5\n",
        "3 4 3 10\n3\n3 1 4\n4 5 9\n3 10 10\n",
        "1 1 2 10\n2\n1 1 3\n2 6 10\n",
        "9 8 7 8\n9\n10 6 6\n10 6 6\n7 7 8\n9 5 6\n8 9 9\n9 5 5\n9 8 8\n8 5 6\n9 10 10\n",
        "6 15 7 15\n9\n6 15 15\n7 14 14\n6 15 15\n9 14 14\n7 14 16\n6 15 15\n6 15 15\n7 14 14\n8 15 15\n",
        "13 16 20 10\n18\n13 16 16\n20 10 10\n19 10 10\n12 15 15\n20 10 10\n18 11 11\n19 10 10\n19 10 10\n20 10 10\n19 10 10\n20 10 10\n20 10 10\n19 10 10\n18 11 11\n13 16 16\n12 15 15\n19 10 10\n19 10 10\n",
        "89 29 88 30\n16\n87 31 31\n14 95 95\n98 88 89\n96 88 88\n14 97 97\n13 97 98\n100 88 88\n88 32 32\n99 88 89\n90 29 29\n87 31 31\n15 94 96\n89 29 29\n88 32 32\n97 89 89\n88 29 30\n",
        "30 14 39 19\n31\n35 7 11\n37 11 12\n32 13 13\n37 5 6\n46 13 13\n37 14 14\n31 13 13\n43 13 19\n45 15 19\n46 13 13\n32 17 17\n41 14 19\n30 14 14\n43 13 17\n34 16 18\n44 11 19\n38 13 13\n40 12 20\n37 16 18\n46 16 18\n34 10 14\n36 9 10\n36 15 19\n38 15 19\n42 13 19\n33 14 15\n35 15 19\n33 17 18\n39 12 20\n36 5 7\n45 12 12\n",
        "2 1 1 1\n2\n1 1 2\n2 1 2\n",
        "1 1 1 2\n5\n1000000000 1 10000\n19920401 1188 5566\n1000000000 1 10000\n1 1 10000\n5 100 200\n",
        "1 1 1000000000 2\n5\n1000000000 1 10000\n19920401 1188 5566\n1000000000 1 10000\n1 1 10000\n5 100 200\n",
    ]
    outputs = [
        "4\n",
        "6\n",
        "-1\n",
        "2\n",
        "1\n",
        "-1\n",
        "1\n",
        "9\n",
        "1\n",
        "1\n",
        "-1\n",
    ]

    test_list = []
    for i, o in zip(inputs, outputs):
        test_list.append({"inputs": i, "outputs": o})

    return {
        "message_log_batch": [
            [
                {
                    "role": "user",
                    "content": "Chess king problem - find minimum moves.",
                },
                {
                    "role": "assistant",
                    "content": r"""```python
from collections import deque

def min_moves(x0, y0, x1, y1, segments):
    # Create a set of allowed cells
    allowed = set()
    for r, a, b in segments:
        for c in range(a, b + 1):
            allowed.add((r, c))

    # Check if the initial and final positions are allowed
    if (x0, y0) not in allowed or (x1, y1) not in allowed:
        return -1

    # Create a graph where each node represents an allowed cell
    graph = {cell: set() for cell in allowed}
    for r, c in allowed:
        for dr, dc in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in allowed:
                graph[(r, c)].add((nr, nc))

    # Perform BFS to find the shortest path
    queue = deque([(x0, y0, 0)])
    visited = set()
    while queue:
        r, c, dist = queue.popleft()
        if (r, c) == (x1, y1):
            return dist
        if (r, c) in visited:
            continue
        visited.add((r, c))
        for nr, nc in graph[(r, c)]:
            queue.append((nr, nc, dist + 1))

    return -1

# Read input
x0, y0, x1, y1 = map(int, input().split())
n = int(input())
segments = [tuple(map(int, input().split())) for _ in range(n)]

# Print the result
print(min_moves(x0, y0, x1, y1, segments))
```""",
                },
            ]
        ],
        "metadata": [{"unittests": test_list, "fn_name": None}],
    }


@pytest.fixture
def battery_charging_test_data():
    """Test data for battery charging problem with function inputs."""
    # Test cases with (battery, charger) -> expected_output
    test_cases = [
        ([1000, 500], "2.6"),
        ([1500, 500], "3.9"),
        ([2000, 1000], "2.6"),
        ([5000, 1000], "6.5"),
        ([1000, 5000], "0.26"),
        ([3050, 2600], "1.53"),
    ]

    test_list = []
    for inputs, output in test_cases:
        # Convert array inputs to newline-separated string format for function calls
        formatted_input = "\n".join(str(arg) for arg in inputs)
        test_list.append({"inputs": formatted_input, "outputs": output})

    return {
        "message_log_batch": [
            [
                {
                    "role": "user",
                    "content": "Write a function to calculate battery charging time with three phases.",
                },
                {
                    "role": "assistant",
                    "content": """```python
def calculate_time(battery, charger):
    # Fast charge phase (0% to 85%)
    fast_charge_time = (battery * 0.85) / charger

    # Decreasing charge phase (85% to 95%)
    decreasing_charge_time = (battery * 0.10) / (charger * 0.5)

    # Trickle charge phase (95% to 100%)
    trickle_charge_time = (battery * 0.05) / (charger * 0.2)

    # Total charging time
    total_time = fast_charge_time + decreasing_charge_time + trickle_charge_time

    # Round to 2 decimal places
    total_time = round(total_time, 2)

    return total_time
```""",
                },
            ]
        ],
        "metadata": [{"unittests": test_list, "fn_name": "calculate_time"}],
    }


@pytest.fixture
def url_grouping_test_data():
    """Test data for URL grouping problem."""
    return {
        "message_log_batch": [
            [
                {
                    "role": "user",
                    "content": "Solve the following coding problem using the programming language python:\n\nThere are some websites that are accessible through several different addresses. For example, for a long time Codeforces was accessible with two hostnames codeforces.com and codeforces.ru.\n\nYou are given a list of page addresses being queried. For simplicity we consider all addresses to have the form http://<hostname>[/<path>], where:\n\n  <hostname> — server name (consists of words and maybe some dots separating them),  /<path> — optional part, where <path> consists of words separated by slashes. \n\nWe consider two <hostname> to correspond to one website if for each query to the first <hostname> there will be exactly the same query to the second one and vice versa — for each query to the second <hostname> there will be the same query to the first one. Take a look at the samples for further clarifications.\n\nYour goal is to determine the groups of server names that correspond to one website. Ignore groups consisting of the only server name.\n\nPlease note, that according to the above definition queries http://<hostname> and http://<hostname>/ are different.\n\n\n-----Input-----\n\nThe first line of the input contains a single integer n (1 ≤ n ≤ 100 000) — the number of page queries. Then follow n lines each containing exactly one address. Each address is of the form http://<hostname>[/<path>], where:\n\n  <hostname> consists of lowercase English letters and dots, there are no two consecutive dots, <hostname> doesn't start or finish with a dot. The length of <hostname> is positive and doesn't exceed 20.  <path> consists of lowercase English letters, dots and slashes. There are no two consecutive slashes, <path> doesn't start with a slash and its length doesn't exceed 20. \n\nAddresses are not guaranteed to be distinct.\n\n\n-----Output-----\n\nFirst print k — the number of groups of server names that correspond to one website. You should count only groups of size greater than one.\n\nNext k lines should contain the description of groups, one group per line. For each group print all server names separated by a single space. You are allowed to print both groups and names inside any group in arbitrary order.\n\n\n-----Examples-----\nInput\n10\nhttp://abacaba.ru/test\nhttp://abacaba.ru/\nhttp://abacaba.com\nhttp://abacaba.com/test\nhttp://abacaba.de/\nhttp://abacaba.ru/test\nhttp://abacaba.de/test\nhttp://abacaba.com/\nhttp://abacaba.com/t\nhttp://abacaba.com/test\n\nOutput\n1\nhttp://abacaba.de http://abacaba.ru \n\nInput\n14\nhttp://c\nhttp://ccc.bbbb/aba..b\nhttp://cba.com\nhttp://a.c/aba..b/a\nhttp://abc/\nhttp://a.c/\nhttp://ccc.bbbb\nhttp://ab.ac.bc.aa/\nhttp://a.a.a/\nhttp://ccc.bbbb/\nhttp://cba.com/\nhttp://cba.com/aba..b\nhttp://a.a.a/aba..b/a\nhttp://abc/aba..b/a\n\nOutput\n2\nhttp://cba.com http://ccc.bbbb \nhttp://a.a.a http://a.c http://abc\n\nThe input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code.",
                },
                {
                    "role": "assistant",
                    "content": """```python
n = int(input())
W = {}
for i in range(n):
    adr = input()
    adr = adr.split("/")
    if adr[-1] == '':
        adr[-1] = '?'
    domena = "/".join(adr[:3])
    adres = "/".join(adr[3:])
    if domena not in W:
        W[domena] = set()
    W[domena].add(adres)

E = {}
for key, ele in list(W.items()):
    lele = "#".join(sorted(list(ele)))
    if lele not in E:
        E[lele] = []
    E[lele].append(key)

res = 0
for key, ele in list(E.items()):
    if len(ele) > 1:
        res += 1

print(res)
for key, ele in list(E.items()):
    if len(ele) > 1:
        print(" ".join(ele))
```""",
                },
            ]
        ],
        "metadata": [
            {
                "unittests": [
                    {
                        "inputs": "10\nhttp://abacaba.ru/test\nhttp://abacaba.ru/\nhttp://abacaba.com\nhttp://abacaba.com/test\nhttp://abacaba.de/\nhttp://abacaba.ru/test\nhttp://abacaba.de/test\nhttp://abacaba.com/\nhttp://abacaba.com/t\nhttp://abacaba.com/test\n",
                        "outputs": "1\nhttp://abacaba.de http://abacaba.ru \n",
                    },
                    {
                        "inputs": "14\nhttp://c\nhttp://ccc.bbbb/aba..b\nhttp://cba.com\nhttp://a.c/aba..b/a\nhttp://abc/\nhttp://a.c/\nhttp://ccc.bbbb\nhttp://ab.ac.bc.aa/\nhttp://a.a.a/\nhttp://ccc.bbbb/\nhttp://cba.com/\nhttp://cba.com/aba..b\nhttp://a.a.a/aba..b/a\nhttp://abc/aba..b/a\n",
                        "outputs": "2\nhttp://cba.com http://ccc.bbbb \nhttp://a.a.a http://a.c http://abc \n",
                    },
                    {
                        "inputs": "10\nhttp://tqr.ekdb.nh/w\nhttp://p.ulz/ifw\nhttp://w.gw.dw.xn/kpe\nhttp://byt.mqii.zkv/j/xt\nhttp://ovquj.rbgrlw/k..\nhttp://bv.plu.e.dslg/j/xt\nhttp://udgci.ufgi.gwbd.s/\nhttp://l.oh.ne.o.r/.vo\nhttp://l.oh.ne.o.r/w\nhttp://tqr.ekdb.nh/.vo\n",
                        "outputs": "2\nhttp://l.oh.ne.o.r http://tqr.ekdb.nh \nhttp://bv.plu.e.dslg http://byt.mqii.zkv \n",
                    },
                    {
                        "inputs": "12\nhttp://ickght.ck/mr\nhttp://a.exhel/.b\nhttp://a.exhel/\nhttp://ti.cdm/\nhttp://ti.cdm/x/wd/lm.h.\nhttp://ickght.ck/a\nhttp://ickght.ck\nhttp://c.gcnk.d/.b\nhttp://c.gcnk.d/x/wd/lm.h.\nhttp://ti.cdm/.b\nhttp://a.exhel/x/wd/lm.h.\nhttp://c.gcnk.d/\n",
                        "outputs": "1\nhttp://a.exhel http://c.gcnk.d http://ti.cdm \n",
                    },
                    {
                        "inputs": "14\nhttp://jr/kgb\nhttp://ps.p.t.jeua.x.a.q.t\nhttp://gsqqs.n/t/\nhttp://w.afwsnuc.ff.km/cohox/u.\nhttp://u.s.wbumkuqm/\nhttp://u.s.wbumkuqm/cohox/u.\nhttp://nq.dzjkjcwv.f.s/bvm/\nhttp://zoy.shgg\nhttp://gsqqs.n\nhttp://u.s.wbumkuqm/b.pd.\nhttp://w.afwsnuc.ff.km/\nhttp://w.afwsnuc.ff.km/b.pd.\nhttp://nq.dzjkjcwv.f.s/n\nhttp://nq.dzjkjcwv.f.s/ldbw\n",
                        "outputs": "2\nhttp://ps.p.t.jeua.x.a.q.t http://zoy.shgg \nhttp://u.s.wbumkuqm http://w.afwsnuc.ff.km \n",
                    },
                    {
                        "inputs": "15\nhttp://l.edzplwqsij.rw/\nhttp://m.e.mehd.acsoinzm/s\nhttp://yg.ttahn.xin.obgez/ap/\nhttp://qqbb.pqkaqcncodxmaae\nhttp://lzi.a.flkp.lnn.k/o/qfr.cp\nhttp://lzi.a.flkp.lnn.k/f\nhttp://p.ngu.gkoq/.szinwwi\nhttp://qqbb.pqkaqcncodxmaae/od\nhttp://qqbb.pqkaqcncodxmaae\nhttp://wsxvmi.qpe.fihtgdvi/e./\nhttp://p.ngu.gkoq/zfoh\nhttp://m.e.mehd.acsoinzm/xp\nhttp://c.gy.p.h.tkrxt.jnsjt/j\nhttp://wsxvmi.qpe.fihtgdvi/grkag.z\nhttp://p.ngu.gkoq/t\n",
                        "outputs": "0\n",
                    },
                    {
                        "inputs": "15\nhttp://w.hhjvdn.mmu/.ca.p\nhttp://m.p.p.lar/\nhttp://lgmjun.r.kogpr.ijn/./t\nhttp://bapchpl.mcw.a.lob/d/ym/./g.q\nhttp://uxnjfnjp.kxr.ss.e.uu/jwo./hjl/\nhttp://fd.ezw.ykbb.xhl.t/\nhttp://i.xcb.kr/.ca.p\nhttp://jofec.ry.fht.gt\nhttp://qeo.gghwe.lcr/d/ym/./g.q\nhttp://gt\nhttp://gjvifpf.d/d/ym/./g.q\nhttp://oba\nhttp://rjs.qwd/v/hi\nhttp://fgkj/\nhttp://ivun.naumc.l/.ca.p\n",
                        "outputs": "2\nhttp://cba.com http://ccc.bbbb \nhttp://a.a.a http://a.c http://abc \n",
                    },
                    {
                        "inputs": "20\nhttp://gjwr/xsoiagp/\nhttp://gdnmu/j\nhttp://yfygudx.e.aqa.ezh/j\nhttp://mpjxue.cuvipq/\nhttp://a/\nhttp://kr/..n/c.\nhttp://a/xsoiagp/\nhttp://kr/z\nhttp://kr/v.cv/rk/k\nhttp://lvhpz\nhttp://qv.v.jqzhq\nhttp://y.no/\nhttp://kr/n\nhttp://y.no/xsoiagp/\nhttp://kr/ebe/z/\nhttp://olsvbxxw.win.n/j\nhttp://p.ct/j\nhttp://mpjxue.cuvipq/xsoiagp/\nhttp://kr/j\nhttp://gjwr/\n",
                        "outputs": "3\nhttp://lvhpz http://qv.v.jqzhq \nhttp://a http://gjwr http://mpjxue.cuvipq http://y.no \nhttp://gdnmu http://olsvbxxw.win.n http://p.ct http://yfygudx.e.aqa.ezh \n",
                    },
                    {"inputs": "1\nhttp://a\n", "outputs": "0\n"},
                    {
                        "inputs": "1\nhttp://a.a.a.f.r.f.q.e.w.a/fwe..sdfv....\n",
                        "outputs": "0\n",
                    },
                    {
                        "inputs": "3\nhttp://abacaba.com/test\nhttp://abacaba.de/test\nhttp://abacaba.de/test\n",
                        "outputs": "1\nhttp://abacaba.com http://abacaba.de \n",
                    },
                ],
                "fn_name": None,
            }
        ],
    }


@pytest.fixture
def snake_position_test_data():
    """Test data for snake final position problem."""
    # Create test cases directly without parsing JSON to avoid length issues
    test_cases = [
        ('2\n["RIGHT", "DOWN"]', "3"),
        ('3\n["DOWN", "RIGHT", "UP"]', "1"),
        ('9\n["RIGHT", "LEFT"]', "0"),  # Simplified from the long one
        (
            '5\n["RIGHT", "DOWN", "RIGHT", "DOWN", "RIGHT", "DOWN", "RIGHT", "DOWN"]',
            "24",
        ),
        ('2\n["DOWN","RIGHT","LEFT","RIGHT"]', "3"),
        ('3\n["DOWN","UP"]', "0"),
        ('4\n["RIGHT", "RIGHT", "DOWN", "DOWN", "LEFT", "LEFT", "UP", "UP"]', "0"),
        ('2\n["RIGHT","DOWN"]', "3"),
        ('2\n["DOWN","RIGHT","LEFT","UP"]', "0"),
    ]

    test_list = []
    for input_str, output_str in test_cases:
        lines = input_str.strip().split("\n")
        n = int(lines[0])
        commands_str = lines[1]
        # Parse the JSON array string
        commands = json.loads(commands_str)

        # Format for function call: n and commands as separate arguments
        formatted_input = f"{n}\n{json.dumps(commands)}"
        test_list.append({"inputs": formatted_input, "outputs": output_str})

    return {
        "message_log_batch": [
            [
                {
                    "role": "user",
                    "content": "Write a function to find the final position of a snake on a grid.",
                },
                {
                    "role": "assistant",
                    "content": """```python
from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        i, j = 0, 0
        for cmd in commands:
            if cmd == "UP":
                i -= 1
            elif cmd == "DOWN":
                i += 1
            elif cmd == "LEFT":
                j -= 1
            elif cmd == "RIGHT":
                j += 1
        return i * n + j
```""",
                },
            ]
        ],
        "metadata": [{"unittests": test_list, "fn_name": "finalPositionOfSnake"}],
    }


@pytest.fixture
def mixed_test_data():
    """Test data with a mix of correct and incorrect responses."""
    return {
        "message_log_batch": [
            [
                {
                    "role": "user",
                    "content": "Write a function that adds two numbers.",
                },
                {
                    "role": "assistant",
                    "content": """```python
def add_numbers(a, b):
    return a + b
```""",
                },
            ],
            [
                {
                    "role": "user",
                    "content": "Write a function that multiplies two numbers.",
                },
                {
                    "role": "assistant",
                    "content": """```python
def multiply_numbers(a, b):
    return a + b  # This is wrong - should be a * b
```""",
                },
            ],
        ],
        "metadata": [
            {
                "unittests": [{"inputs": "2\n3", "outputs": "5"}],
                "fn_name": "add_numbers",
            },
            {
                "unittests": [{"inputs": "3\n4", "outputs": "12"}],
                "fn_name": "multiply_numbers",
            },
        ],
    }


def test_url_grouping_problem(code_env, url_grouping_test_data):
    """Test URL grouping problem with comprehensive test cases."""
    result = ray.get(
        code_env.step.remote(
            url_grouping_test_data["message_log_batch"],
            url_grouping_test_data["metadata"],
        )
    )

    # Verify the basic structure
    assert len(result.observations) == 1
    assert result.observations[0]["role"] == "environment"
    assert result.rewards.shape == (1,)


def test_snake_position_problem(code_env, snake_position_test_data):
    """Test snake final position problem with function calls."""
    result = ray.get(
        code_env.step.remote(
            snake_position_test_data["message_log_batch"],
            snake_position_test_data["metadata"],
        )
    )

    # Verify the basic structure
    assert len(result.observations) == 1
    assert result.observations[0]["role"] == "environment"
    assert result.rewards.shape == (1,)


def test_code_env_simple_function(code_env):
    """Test basic functionality with a simple function."""
    message_log_batch = [
        [
            {
                "role": "user",
                "content": "Write a function that adds two numbers.",
            },
            {
                "role": "assistant",
                "content": "```python\ndef add_numbers(a, b):\n    return a + b\n```",
            },
        ]
    ]
    metadata = [
        {
            "unittests": [
                {"inputs": "2\n3", "outputs": "5"},
                {"inputs": "10\n5", "outputs": "15"},
            ],
            "fn_name": "add_numbers",
        }
    ]

    result = ray.get(code_env.step.remote(message_log_batch, metadata))

    # Check observations
    assert len(result.observations) == 1, "Should return observation for 1 message"
    assert result.observations[0]["role"] == "environment", (
        "Observation should be from environment"
    )
    assert result.observations[0]["content"] == "Environment: correct", (
        "Response should be correct"
    )

    # Check rewards and done flags
    assert result.rewards.shape == (1,), "Rewards should be a tensor of shape (1,)"
    assert result.rewards[0] == 1.0, "Reward should be 1.0 for correct answer"
    assert result.terminateds.shape == (1,), (
        "Terminated flags should be a tensor of shape (1,)"
    )
    assert result.terminateds[0] == 1.0, "Terminated flag should be 1.0"


def test_chess_king_problem(code_env, chess_king_test_data):
    """Test chess king problem with comprehensive test cases."""
    result = ray.get(
        code_env.step.remote(
            chess_king_test_data["message_log_batch"], chess_king_test_data["metadata"]
        )
    )

    # Test should work with your exact format
    assert len(result.observations) == 1
    assert result.observations[0]["role"] == "environment"
    assert result.rewards.shape == (1,)


def test_code_env_step_mixed(code_env, mixed_test_data):
    """Test CodeEnvironment step with a mix of correct and incorrect responses."""
    result = ray.get(
        code_env.step.remote(
            mixed_test_data["message_log_batch"], mixed_test_data["metadata"]
        )
    )

    # Check observations and rewards
    assert len(result.observations) == 2, (
        "Should return observations for all 2 messages"
    )
    assert result.observations[0]["content"] == "Environment: correct", (
        "First response should be correct"
    )
    assert result.observations[1]["content"] == "Environment: incorrect", (
        "Second response should be incorrect"
    )

    assert result.rewards.shape == (2,), "Rewards should be a tensor of shape (2,)"
    assert result.rewards[0] == 1.0, "First reward should be 1.0"
    assert result.rewards[1] == 0.0, "Second reward should be 0.0"


def test_code_env_step_empty(code_env):
    """Test CodeEnvironment step with empty input."""
    result = ray.get(code_env.step.remote([], []))

    # Check all outputs are empty
    assert len(result.observations) == 0, "Should return empty observations list"
    assert len(result.metadata) == 0, "Should return empty metadata list"
    assert result.rewards.shape == (0,), "Should return empty rewards tensor"
    assert result.terminateds.shape == (0,), "Should return empty terminateds tensor"


def test_battery_charging_problem(code_env, battery_charging_test_data):
    """Test battery charging problem with function calls."""
    result = ray.get(
        code_env.step.remote(
            battery_charging_test_data["message_log_batch"],
            battery_charging_test_data["metadata"],
        )
    )

    # Test should work with your exact format
    assert len(result.observations) == 1
    assert result.observations[0]["role"] == "environment"
    assert result.rewards.shape == (1,)


@pytest.mark.parametrize("batch_size", [1, 2, 10])
def test_code_env_various_batches(code_env, batch_size):
    """Test CodeEnvironment step with different batch sizes."""
    message_log_batch = [
        [
            {
                "role": "user",
                "content": "Write a function that returns the input unchanged.",
            },
            {
                "role": "assistant",
                "content": "```python\ndef identity(x):\n    return x\n```",
            },
        ]
    ] * batch_size
    metadata = [
        {
            "unittests": [
                {"inputs": "5", "outputs": "5"},
                {"inputs": "42", "outputs": "42"},
            ],
            "fn_name": "identity",
        }
    ] * batch_size

    result = ray.get(code_env.step.remote(message_log_batch, metadata))

    # Check outputs
    assert len(result.observations) == batch_size, (
        f"Should return observations for all {batch_size} messages"
    )
    assert all(
        obs["content"] == "Environment: correct" for obs in result.observations
    ), "All responses should be correct"
    assert result.rewards.shape == (batch_size,), (
        "Rewards should be a tensor of shape (batch_size,)"
    )
    assert all(result.rewards == 1.0), "All rewards should be 1.0"
    assert result.terminateds.shape == (batch_size,), (
        "Terminated flags should be a tensor of shape (batch_size,)"
    )
    assert all(result.terminateds == 1.0), "All terminated flags should be 1.0"


def test_code_exception_handling(code_env):
    """Test CodeEnvironment step with an exception in the verify function."""
    message_log_batch = [
        [
            {
                "role": "user",
                "content": "Write a function with syntax error.",
            },
            {
                "role": "assistant",
                "content": '```python\ndef broken_function(\n    # Missing closing parenthesis - syntax error\n    return "broken"\n```',
            },
        ]
    ]
    metadata = [
        {
            "unittests": [{"inputs": "", "outputs": "working"}],
            "fn_name": "broken_function",
        }
    ]

    result = ray.get(code_env.step.remote(message_log_batch, metadata))

    # Program should not crash, should handle exception gracefully
    assert result.rewards.shape == (1,), "Rewards should be a tensor of shape (1,)"
    assert result.rewards[0] == 0.0, "Reward should be 0.0 for failed execution"
    assert result.observations[0]["content"] == "Environment: incorrect", (
        "Should return incorrect for failed execution"
    )


def test_code_env_debug(code_env):
    """Debug test to understand what's happening in code verification."""
    message_log_batch = [
        [
            {
                "role": "user",
                "content": "Write a function that adds two numbers.",
            },
            {
                "role": "assistant",
                "content": "```python\ndef add_numbers(a, b):\n    return a + b\n```",
            },
        ]
    ]
    metadata = [
        {"unittests": [{"inputs": "2\n3", "outputs": "5"}], "fn_name": "add_numbers"}
    ]

    result = ray.get(code_env.step.remote(message_log_batch, metadata))

    # Print debug information
    print(f"Result observations: {result.observations}")
    print(f"Result rewards: {result.rewards}")

    # Just verify the basic structure works
    assert len(result.observations) == 1
    assert result.observations[0]["role"] == "environment"
    assert result.rewards.shape == (1,)
    assert result.rewards[0] in [0.0, 1.0]
