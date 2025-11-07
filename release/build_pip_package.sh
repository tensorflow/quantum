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
set -x

# Pick the Python that TFQ/TensorFlow used during configure/build.
# Order: explicit env -> 3.11 -> python3
PY="${PYTHON_BIN_PATH:-}"
if [[ -z "$PY" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PY="$(command -v python3.11)"
  elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
  else
    echo "ERROR: No suitable python found. Set PYTHON_BIN_PATH." >&2
    exit 2
  fi
fi
echo "Using Python: $PY"

# Ensure packaging tools are present in THIS interpreter
"$PY" - <<'PY' || { "$PY" -m pip install --upgrade pip setuptools wheel; }
import importlib
for m in ["setuptools","wheel"]:
    importlib.import_module(m)
PY

EXPORT_DIR="bazel-bin/release/build_pip_package.runfiles/__main__"

main() {
  DEST="$1"
  EXTRA_FLAGS="$2"

  if [[ -z "$DEST" ]]; then
    echo "No destination directory provided."
    exit 1
  fi

  mkdir -p "$DEST"
  echo "=== destination directory: $DEST"

  TMPDIR="$(mktemp -d -t tmp.XXXXXXXXXX)"
  echo "$(date) : === Using tmpdir: $TMPDIR"
  echo "=== Copy TFQ files"

  cp "${EXPORT_DIR}/release/setup.py"        "$TMPDIR"
  cp "${EXPORT_DIR}/release/MANIFEST.in"     "$TMPDIR"
  mkdir "$TMPDIR/tensorflow_quantum"
  cp -r -v "${EXPORT_DIR}/tensorflow_quantum/"* "$TMPDIR/tensorflow_quantum/"

  pushd "$TMPDIR"
  echo "$(date) : === Building wheel"
  "$PY" setup.py bdist_wheel $EXTRA_FLAGS > /dev/null
  cp dist/*.whl "$DEST"
  popd
  rm -rf "$TMPDIR"
  echo "$(date) : === Output wheel file is in: $DEST"
}

main "$@"
