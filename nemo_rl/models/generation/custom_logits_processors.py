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

from typing import List, Optional, Any
import torch
from transformers import AutoTokenizer

try:
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
except ImportError:
    # Fallback for different vLLM versions
    try:
        from vllm.sampling_params import LogitsProcessor
    except ImportError:
        # Create a base class if neither is available
        class LogitsProcessor:
            def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
                return scores

from ast import Tuple
from sys import thread_info
from typing import Tuple
import torch


NEWLINE_TOKENS = {12291, 8199, 20487, 57354, 71693, 14350, 126989, 2064, 10260, 26644, 114710, 10263, 106525, 81952, 116768, 10274, 51235, 28710, 94246, 79914, 86059, 2092, 79917, 18479, 18481, 96306, 32819, 2100, 14390, 67639, 73782, 94265, 110650, 122935, 2116, 12357, 114758, 86087, 67656, 4169, 100423, 104519, 106570, 114759, 124997, 102479, 12368, 14416, 14417, 38997, 98389, 34903, 67673, 102490, 26720, 94306, 12390, 61546, 53355, 32876, 98414, 26740, 90228, 94325, 39031, 127095, 18554, 47229, 6273, 51330, 4227, 2180, 98437, 100485, 112779, 96397, 108686, 43151, 4241, 110737, 69780, 49301, 86173, 4256, 118944, 18595, 16548, 102565, 12455, 114859, 92334, 104623, 6320, 118960, 118963, 55476, 67766, 14519, 55479, 123067, 61631, 32960, 2241, 63685, 114885, 116935, 63691, 125131, 57551, 104655, 73942, 100571, 16606, 73954, 16616, 114925, 14576, 51440, 20723, 4342, 67830, 14588, 123134, 33024, 4353, 39170, 26885, 67846, 123142, 129286, 55561, 28938, 37131, 84236, 12560, 88337, 43282, 6422, 24866, 90404, 14630, 45352, 20777, 31020, 20783, 14642, 65846, 8503, 4410, 100670, 14657, 53569, 2373, 20806, 90445, 6478, 104785, 41301, 20822, 115029, 8538, 6491, 47456, 94561, 24931, 72038, 117098, 31086, 43374, 43378, 80242, 14708, 61813, 82293, 53623, 92535, 110964, 39293, 94592, 104833, 72068, 72071, 10632, 84360, 98696, 55692, 22926, 27022, 31120, 35215, 117138, 74136, 123293, 37278, 86430, 106910, 37281, 125346, 113060, 68007, 65960, 88489, 98729, 43438, 33203, 39347, 102839, 57784, 49593, 76218, 108983, 68030, 92606, 16832, 100798, 14790, 43462, 37321, 4554, 86475, 96716, 14797, 43472, 29141, 12758, 2519, 109013, 14810, 16858, 72156, 10717, 61918, 125407, 80352, 84451, 12776, 76264, 92650, 6637, 80365, 125422, 8691, 104947, 92665, 23034, 14843, 12795, 14845, 72189, 14847, 107007, 66050, 64006, 4615, 12807, 74247, 80390, 123401, 14863, 25104, 90641, 35346, 37395, 2580, 94736, 84503, 2591, 102946, 76323, 74283, 18988, 70189, 121388, 125483, 14896, 98865, 100915, 90676, 19002, 35390, 80447, 47680, 98879, 86596, 100932, 27207, 2634, 43597, 2638, 107087, 2640, 4688, 121426, 82515, 113236, 8789, 21078, 8791, 88661, 127578, 21090, 82531, 19044, 84578, 94822, 33383, 96867, 115307, 25197, 62062, 64112, 10869, 72310, 8824, 14971, 17020, 43646, 2687, 78464, 66177, 12930, 27266, 86659, 96894, 119428, 129666, 8841, 123531, 29324, 103055, 2706, 113300, 111257, 31391, 82592, 37541, 15014, 49833, 2731, 96940, 68269, 2734, 115377, 94903, 21181, 62142, 31423, 86720, 6849, 43715, 45763, 72388, 76485, 92867, 94919, 17100, 51919, 49872, 60113, 58067, 27350, 117462, 41690, 10971, 53979, 56028, 10974, 31454, 53982, 115419, 39653, 78566, 127727, 31473, 35571, 60147, 64244, 35574, 97015, 119546, 2812, 31484, 84733, 127743, 45825, 2820, 21253, 78597, 27399, 29448, 11017, 97033, 6923, 27404, 35596, 86796, 101126, 117520, 84754, 2838, 35608, 8985, 76569, 82713, 125721, 70433, 49958, 29479, 82729, 2861, 60206, 37679, 66357, 113462, 2871, 74551, 119606, 19261, 13118, 78657, 97091, 74566, 113483, 6991, 47952, 35669, 11095, 35674, 105306, 95068, 13149, 2913, 35683, 9060, 11108, 31588, 66404, 23400, 107366, 123751, 33645, 127854, 58226, 9075, 47988, 25461, 21366, 80758, 19323, 62332, 107387, 115580, 31624, 103304, 25482, 54158, 68494, 50068, 93078, 11159, 13207, 80791, 115607, 56219, 97181, 117662, 21407, 41890, 33699, 84898, 86947, 129960, 78768, 25525, 19383, 19384, 60343, 74681, 119736, 58303, 113599, 19394, 95172, 5062, 119750, 109514, 60363, 15308, 95179, 43982, 121809, 76754, 9171, 23507, 62421, 80855, 23513, 70623, 76772, 74726, 27624, 100335, 111595, 1010, 7154, 60403, 80888, 44025, 29690, 97278, 7168, 11270, 5127, 62473, 58380, 54285, 76816, 107537, 107539, 3092, 56343, 39960, 44059, 3100, 115739, 19486, 44065, 101411, 31782, 89128, 23593, 105512, 109611, 11308, 125994, 68655, 117811, 87093, 121909, 7223, 9273, 44090, 7227, 66620, 107577, 130110, 3135, 11330, 9283, 80962, 97347, 99397, 23626, 25676, 95309, 123980, 29775, 115792, 50259, 68692, 87126, 27736, 42076, 19551, 19553, 99425, 58467, 103527, 66666, 50283, 95338, 19565, 99437, 44143, 130154, 99442, 117876, 40058, 50298, 111739, 11389, 11390, 7295, 68735, 7297, 87168, 66691, 97412, 23685, 58502, 126084, 78984, 19594, 23691, 76944, 19602, 11411, 25748, 101522, 109715, 130199, 130200, 40089, 52377, 115869, 19614, 17572, 9381, 76968, 107688, 79020, 33966, 13487, 50352, 111791, 58547, 93363, 91318, 54455, 52408, 56504, 5306, 76984, 83135, 70848, 25799, 11471, 38097, 21714, 60625, 40151, 79063, 91352, 5338, 111832, 77021, 46302, 9439, 50400, 3297, 3298, 25828, 52454, 23784, 36080, 107760, 1267, 3318, 17655, 118010, 15611, 70909, 111870, 120061, 97536, 60673, 87298, 93441, 115976, 124171, 7437, 7438, 48403, 5396, 34069, 58643, 64790, 66835, 109848, 46366, 52511, 21794, 60707, 21796, 44325, 83235, 93475, 105766, 23849, 101673, 46380, 27949, 19758, 103726, 68912, 56626, 19767, 40247, 1338, 5439, 73024, 87362, 25923, 54597, 7498, 1355, 116042, 9549, 9551, 5457, 56657, 70993, 81235, 1365, 34133, 19802, 34138, 52572, 32093, 109916, 120155, 89445, 73064, 11625, 56683, 111979, 109940, 21877, 3448, 56697, 71032, 103804, 64895, 71040, 111999, 60802, 73091, 48516, 62853, 17798, 73093, 7562, 32140, 109964, 85394, 36243, 118171, 7580, 13724, 13725, 21916, 60830, 71069, 71071, 114080, 118177, 3493, 128422, 21927, 73127, 3500, 1456, 21942, 120246, 1468, 36285, 32194, 3523, 36297, 50634, 19920, 15829, 93653, 30168, 128472, 3546, 17882, 69082, 114140, 101854, 1512, 11753, 9706, 3563, 13803, 46570, 60907, 1520, 36336, 22004, 69108, 124407, 48632, 95738, 52733, 85501, 75265, 30211, 99843, 24071, 48647, 87559, 122375, 32268, 128535, 1561, 48668, 11810, 32291, 1572, 83494, 38439, 95783, 114219, 89644, 85551, 120367, 101940, 75319, 3640, 52797, 1600, 91712, 28230, 3655, 20039, 22087, 34378, 38471, 46665, 65102, 75342, 56913, 38482, 116306, 120408, 1626, 34395, 7772, 73307, 83549, 30303, 91740, 54881, 28258, 120413, 13926, 83558, 1640, 1641, 104042, 106092, 11886, 28272, 56950, 65142, 44664, 24185, 99959, 106108, 128638, 59009, 11906, 26244, 65159, 24202, 56977, 34450, 61074, 24212, 102038, 93852, 81566, 18081, 46753, 79523, 50852, 9893, 69285, 85665, 104104, 61100, 7854, 28335, 128687, 61105, 102066, 26293, 24246, 38582, 3768, 61112, 57018, 81592, 93884, 22205, 126654, 124608, 89793, 38594, 114377, 28364, 130764, 20174, 63183, 14032, 50898, 9940, 28378, 46811, 102109, 57054, 124643, 83684, 44775, 55015, 3817, 79594, 48875, 67307, 3824, 110322, 57075, 67316, 9973, 16117, 20213, 12024, 55031, 120568, 69371, 32508, 100092, 104190, 130808, 65280, 38659, 12039, 1801, 28426, 77579, 53004, 61198, 32530, 108306, 59158, 67350, 3864, 30489, 130842, 102173, 5920, 16161, 1826, 69411, 87840, 87842, 16166, 24362, 3885, 18225, 128818, 32563, 1844, 89907, 53047, 71482, 108346, 44861, 69437, 71488, 69441, 89922, 94017, 100167, 32584, 1877, 110423, 124760, 106331, 53084, 63325, 42846, 65374, 100191, 20321, 38754, 106341, 118631, 100202, 116588, 112493, 14190, 36719, 46958, 71537, 102254, 89974, 104312, 12156, 38781, 10110, 3971, 71556, 40838, 112518, 65416, 36745, 57226, 87946, 10127, 92048, 3989, 34710, 75670, 118680, 20377, 24474, 71578, 38816, 1953, 71586, 24483, 6052, 83877, 110496, 128930, 67499, 53165, 28590, 122797, 20400, 126895, 102323, 30645, 10166, 63413, 92089, 114617, 49083, 57277, 120765, 55234, 20419, 40900, 79811, 88002, 38858, 4043, 38859, 65483, 57294, 83917, 104396, 2002, 26579, 30679, 120791, 88025, 57311, 26592, 30689, 12260, 100324, 53223, 71657, 2030, 4078, 14320, 22512, 28656, 8179, 4084, 32753, 67568, 79855, 61432, 122875, 116734}

def suffix_prefix_overlap(a, b):
    m = min(len(a), len(b))
    for k in range(m, 0, -1):           # try longest first
        if a[-k:] == b[:k]:
            return k
    return 0

class ThinkingBudgetLogitsProcessor:
    """A logits processor that limits the number of thinking tokens."""

    def __init__(self, think_max_tokens: int = 128, think_end_token_ids: List[int] = [2]):
        """
        Initialize the thinking logits processor.

        Args:
            think_max_tokens: Maximum number of tokens allowed in thinking section
            think_end_token_id: List of Token IDs to force when think_max_tokens is reached.
        """
        self.think_max_tokens = think_max_tokens
        self.think_end_token_ids = think_end_token_ids
        self.len_think_end_ids = len(think_end_token_ids)
        self.start_of_end = False
        self.end_of_end = False
        # grace period
        grace = int(0.15 * think_max_tokens)
        if grace < 500:
            grace = 500
        if grace > 1000:
            grace = 1000
        self.grace = grace + think_max_tokens

        print(f"thinking max tokens is {self.think_max_tokens}")
        print(f"grace period is {self.grace}")
        print(f"thinking eos is is {self.think_end_token_ids}")

    def __call__(self, input_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """
        Process the logits to enforce end token when max tokens reached.

        Args:
            input_ids: List of input token IDs
            logits: Tensor of logits for the next token

        Returns:
            Processed logits tensor
        """
        if len(input_ids) > self.grace and not self.start_of_end:
            """
            start the end thinking process because we are past the grace period
            """
            print("grace period over...")
            self.start_of_end = True

        if len(input_ids) > self.think_max_tokens and input_ids[-1] in NEWLINE_TOKENS and not self.start_of_end:
            """
            start the end thinking process because we are past the budget and found a newline token
            """
            print("new line token after max thinking...")
            self.start_of_end = True

        if self.start_of_end and not self.end_of_end:
            print(f"Forcing end sequence token (ID: {self.think_end_token_ids}) after {len(input_ids)} tokens")
            last_n_inputs = list(input_ids[-self.len_think_end_ids:])
            overlap = suffix_prefix_overlap(last_n_inputs, self.think_end_token_ids)
            print(last_n_inputs, overlap)
            if overlap < self.len_think_end_ids:
                logits = torch.full_like(logits, float('-inf'))
                insert_id = self.think_end_token_ids[overlap]
                logits[insert_id] = 1.0
            else:
                self.end_of_end = True
        return logits

class TokenBanLogitsProcessor:
    """Custom logits processor that bans specific tokens by setting their logits to negative infinity."""
    
    def __init__(self, banned_token_ids: List[int]):
        """
        Initialize the token ban processor.
        
        Args:
            banned_token_ids: List of token IDs to ban during generation
        """
        self.banned_token_ids = set(banned_token_ids)
    
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """
        Apply token banning to the logits.
        
        Args:
            input_ids: List of input token IDs
            scores: Logits tensor of shape (vocab_size,)
            
        Returns:
            Modified logits tensor with banned tokens set to -inf
        """
        for token_id in self.banned_token_ids:
            if token_id < scores.shape[0]:
                scores[token_id] = float('-inf')
        return scores


class TemperatureScalingLogitsProcessor:
    """Custom logits processor that applies temperature scaling with dynamic adjustment."""
    
    def __init__(self, base_temperature: float = 1.0, scale_factor: float = 1.0):
        """
        Initialize the temperature scaling processor.
        
        Args:
            base_temperature: Base temperature for scaling
            scale_factor: Factor to adjust temperature dynamically
        """
        self.base_temperature = base_temperature
        self.scale_factor = scale_factor
    
    def __call__(self, input_ids: List[int], scores: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to the logits.
        
        Args:
            input_ids: List of input token IDs
            scores: Logits tensor of shape (vocab_size,)
            
        Returns:
            Modified logits tensor with temperature scaling applied
        """
        # Dynamic temperature based on sequence length
        dynamic_temp = self.base_temperature * (1.0 + len(input_ids) * self.scale_factor * 0.01)
        
        if dynamic_temp != 1.0:
            scores = scores / dynamic_temp
            
        return scores
