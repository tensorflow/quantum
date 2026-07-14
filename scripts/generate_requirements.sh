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

# Summary: produce a version-specific requirements lock using pip-compile.
# Usage: ./scripts/generate_requirements.sh

set -eu

# Go to the top of the local TFQ git tree. Do it early in case this fails.
thisdir=$(CDPATH="" cd -- "$(dirname -- "${0}")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "${repo_dir}" ]]; then
  repo_dir=$(CDPATH="" cd -- "${thisdir}/.." && pwd -P)
fi
cd "${repo_dir}"

py_minor=$(python - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) not in {(3, 10), (3, 11), (3, 12)}:
    raise SystemExit(1)
print(f"{major}.{minor}")
PY
) || {
  echo "Error: run this script with Python 3.10, 3.11, or 3.12." >&2
  exit 1
}

lock_file="requirements_lock_${py_minor/./_}.txt"

if ! python -m pip show -qq pip-tools; then
  echo "Error: 'pip-compile' not found. Please install 'pip-tools'." >&2
  exit 1
fi

# Check for a constraints file and use it if it exists.
declare -a constraints=()
pins_file="$(realpath --relative-to=. "${repo_dir}/requirements-pins.txt")"
if [[ -e "${pins_file}" ]]; then
  constraints+=(--constraints "${pins_file}")
fi

# Have pip-compile mention this script in the requirements header it writes.
export CUSTOM_COMPILE_COMMAND="${0}"

echo "Running pip-compile in ${repo_dir} for Python ${py_minor} -> ${lock_file} ..."
python -m piptools compile -q \
  --allow-unsafe \
  --upgrade \
  --rebuild \
  --generate-hashes \
  --no-strip-extras \
  --no-emit-index-url \
  -o "${lock_file}" \
  "${constraints[@]}"

if [[ "${py_minor}" == "3.11" ]]; then
  cp "${lock_file}" requirements.txt
  echo "Updated requirements.txt from ${lock_file} (primary dev lock)."
fi

echo "Done."
