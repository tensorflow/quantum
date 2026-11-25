#!/bin/bash
# Copyright 2026 The TensorFlow Quantum Authors
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

set -eu

user_py=""
for arg in "$@"; do
  case "$arg" in
    --python=*) user_py="${arg#--python=}" ;;
    *) echo "Error: unknown arg: $arg" >&2; exit 1 ;;
  esac
done

thisdir=$(CDPATH="" cd -- "$(dirname -- "${0}")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "${repo_dir}" ]]; then
  repo_dir=$(CDPATH="" cd -- "${thisdir}/.." && pwd -P)
fi
cd "${repo_dir}"

py_bin="${user_py:-${PYTHON_BIN_PATH:-}}"
if [[ -z "${py_bin}" ]]; then
  py_bin="$(command -v python3 || command -v python || true)"
fi

if [[ -z "${py_bin}" ]]; then
  echo "Error: could not find a Python interpreter." >&2
  exit 1
fi

py_minor=$("${py_bin}" - <<'PY'
import sys
major, minor = sys.version_info[:2]
if (major, minor) not in {(3, 10), (3, 11), (3, 12)}:
    raise SystemExit(1)
print(f"{major}.{minor}")
PY
) || {
  echo "Error: Python 3.10, 3.11, or 3.12 required; found $("${py_bin}" -V 2>&1)." >&2
  exit 1
}

echo "${repo_dir}/requirements_lock_${py_minor/./_}.txt"
