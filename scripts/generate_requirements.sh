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
# Usage: ./scripts/generate_requirements.sh

set -eo pipefail

# Change to the top of the local TFQ git tree. Do it early in case this fails.
thisdir=$(CDPATH="" cd -- "$(dirname -- "${0}")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel 2>/dev/null || \
    echo "${thisdir}/..")
cd "${repo_dir}"

if ! pip show -qq pip-tools; then
    echo "ERROR: 'pip-compile' not found. Please install 'pip-tools'." >&2
    exit 1
fi

# Don't force the use of a constraint file, but use it if exists.
declare -a constraint=()
pins_file="$(realpath --relative-to=. "${repo_dir}/requirements-pins.txt")"
if [[ -e "${pins_file}" ]]; then
    constraint+=(--constraint "${pins_file}")
fi

# Tell pip-compile to reference this script in the requirements.txt comments.
export CUSTOM_COMPILE_COMMAND="${0}"

echo "Running pip-compile in ${repo_dir} …"
pip-compile -q \
    --rebuild \
    --allow-unsafe \
    --no-strip-extras \
    --no-emit-index-url \
    "${constraint[@]}"

echo "Done."
