#!/bin/bash
# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
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

set -e

# Pick the Python that TFQ/TensorFlow used during configure/build.
# Order: explicit env -> python3 (>= 3.10)
PY="${PYTHON_BIN_PATH:-}"
if [[ -z "${PY}" ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found. Set PYTHON_BIN_PATH to a Python 3.10+ interpreter." >&2
    exit 2
  fi

  # Require Python >= 3.10 for TFQ.
  if ! python3 - <<'PY'
import sys
sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)
PY
  then
    echo "ERROR: Python 3.10+ required for TensorFlow Quantum; found $(python3 -V 2>&1)." >&2
    exit 2
  fi

  PY="$(command -v python3)"
fi
echo "Using Python: ${PY}"

# Ensure packaging tools are present in THIS interpreter.
pip install -qq setuptools wheel build

EXPORT_DIR="bazel-bin/release/build_pip_package.runfiles/__main__"

main() {
  DEST="${1}"
  EXTRA_FLAGS="${2}"

  if [[ -z "${DEST}" ]]; then
    echo "No destination directory provided."
    exit 1
  fi

  mkdir -p "${DEST}"
  echo "=== Destination directory for wheel file: ${DEST}"

  # Build the pip package in a temporary directory.
  TMPDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  echo "$(date) : === Using tmpdir: ${TMPDIR}"
  echo "=== Copying TFQ files"

  # Copy over files necessary to run setup.py
  cp "${EXPORT_DIR}/release/setup.py" "${TMPDIR}"
  cp "${EXPORT_DIR}/release/MANIFEST.in" "${TMPDIR}"
  mkdir "${TMPDIR}/tensorflow_quantum"
  cp -r -v "${EXPORT_DIR}/tensorflow_quantum/"* "${TMPDIR}/tensorflow_quantum/"

  pushd "${TMPDIR}"
  echo "$(date) : === Building wheel"
  "${PY}" -m build -v --wheel ${EXTRA_FLAGS} > /dev/null
  cp dist/*.whl "${DEST}"
  popd
  rm -rf "${TMPDIR}"
  echo "$(date) : === Done."
}

main "$@"
