#!/bin/bash
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

# Files ending with .py should have Copyright notice in the first line.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Move to the project root
cd $SCRIPT_DIR/..
find_files_with_missing_copyright() {
find ./nemo_rl/ ./docs/*.py ./examples/ ./tests/ ./tools/ ./3rdparty/*/*.py -type f -name '*.py' | while read path; do
    echo -en $path"\t"
    head -2 $path | grep -iv 'coding=' | head -1
done \
   | egrep -iv 'Copyright.*NVIDIA CORPORATION.*All rights reserved.' \
   | grep -iv 'BSD 3-Clause License' \
   | grep -iv 'Copyright.*Microsoft' \
   | grep -iv 'Copyright.*The Open AI Team' \
   | grep -iv 'Copyright.*The Google AI' \
   | grep -iv 'Copyright.*Facebook' | while read line; do
     echo $line | cut -d' ' -f1
   done
}


declare RESULT=($(find_files_with_missing_copyright))  # (..) = array

if [ "${#RESULT[@]}" -gt 0 ]; then
   echo "Error: Found files with missing copyright:"
   for (( i=0; i<"${#RESULT[@]}"; i++ )); do
      echo "path= ${RESULT[$i]}"
   done
   cat <<EOF
=====================
= Example Copyright =
=====================
# Copyright (c) $(date +%Y), NVIDIA CORPORATION.  All rights reserved.
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
EOF
   exit 1;
else
   echo "Ok: All files start with copyright notice"
fi
