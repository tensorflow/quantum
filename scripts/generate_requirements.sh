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
thisdir=$(CDPATH="" cd -- "$(dirname -- "${0}")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel 2>/dev/null || \
  echo "${thisdir}/..")

usage="Usage: ${0}
Generate TFQ requirements.txt file from requirements.in."

# Exit early if the user requested help.
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    echo "$usage"
    exit 0
fi

# Ensure we have pip-compile.
if ! pip show -qq pip-tools; then
    echo "Python pip-tools not found. Please run 'pip install pip-tools'."
    exit 1
fi

# Tell pip-compile to put this command in the comment header in requirements.txt
export CUSTOM_COMPILE_COMMAND="${0}"

constraint=""
constraint_file="${repo_dir}/requirements-pins.txt"
if [[ -e "${constraint_file}" ]]; then
    constraint="--constraint $(realpath --relative-to=. "${constraint_file}")"
fi

echo "Running pip-compile in ${repo_dir} …"
# shellcheck disable=SC2086
pip-compile -q \
    --rebuild \
    --allow-unsafe \
    --no-strip-extras \
    --no-emit-index-url \
     ${constraint}

declare -a inplace_edit=(-i)
if [[ "$(uname -s)" == "Darwin" ]]; then
  # macOS uses BSD sed, which requires a suffix for -i.
  inplace_edit+=('')
fi

# Pyyaml is a transitive dependency, and pinning the version (as pip-compile
# does) leads to unsatisfiable constraints on some platforms. However, we
# don't need pyyaml to be a particular version. There's no easy way to tell
# pip-compile not to constrain a particular package, so
echo "Adjusting output of pip-compile …"
sed "${inplace_edit[@]}" \
  -e 's/^pyyaml==.*/pyyaml/' \
  requirements.txt

echo "Done."
