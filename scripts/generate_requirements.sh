#!/bin/bash
# Copyright 2025 The TensorFlow Quantum Authors
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
# ==============================================================================

# Summary: produce requirements.txt using pip-compile & munging the result.
# Usage: ./generate_requirements.sh

set -eu

# Find the top of the local TFQ git tree. Do it early in case this fails.
thisdir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel 2>/dev/null || \
  echo "${thisdir}/..")

echo "Running pip-compile in ${repo_dir} …"
pip-compile -q --no-strip-extras --allow-unsafe

declare -a inplace_edit=(-i)
if [[ "$(uname -s)" == "Darwin" ]]; then
  # macOS uses BSD sed, which requires a suffix for -i.
  inplace_edit+=('')
fi

echo "Adjusting output of pip-compile …"
sed "${inplace_edit[@]}" \
  -e '/^--index-url/d' \
  -e '/^--extra-index-url/d' \
  -e 's/^pyyaml==.*/pyyaml/' \
  requirements.txt

echo "Done."
