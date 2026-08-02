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

import unittest

from nemo_rl.data.datasets.preference_datasets import load_preference_dataset


class TestLoadPreferenceDataset(unittest.TestCase):
    def test_missing_dataset_name_raises_valueerror(self):
        """A missing dataset_name must raise a clear ValueError, not KeyError."""
        with self.assertRaisesRegex(ValueError, "dataset_name"):
            load_preference_dataset({})


if __name__ == "__main__":
    unittest.main()
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

import unittest

from nemo_rl.data.datasets.preference_datasets import load_preference_dataset


class TestLoadPreferenceDataset(unittest.TestCase):
    def test_missing_dataset_name_raises_valueerror(self):
        """A missing dataset_name must raise a clear ValueError, not KeyError."""
        with self.assertRaisesRegex(ValueError, "dataset_name"):
            load_preference_dataset({})


