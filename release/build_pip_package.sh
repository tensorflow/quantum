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

# Ensure packaging tools are present in THIS interpreter.
pip install -qq setuptools wheel build --root-user-action ignore

EXPORT_DIR="bazel-bin/release/build_pip_package.runfiles/__main__"

function main() {
  DEST=${1}
  EXTRA_FLAGS=${2}

  if [[ -z ${DEST} ]]; then
    echo "No destination directory provided."
    exit 1
  fi

  mkdir -p ${DEST}
  echo "=== destination directory: ${DEST}"

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  echo "=== Copy TFQ files"

  # Copy over files necessary to run setup.py
  cp ${EXPORT_DIR}/release/setup.py "${TMPDIR}"
  cp ${EXPORT_DIR}/release/MANIFEST.in "${TMPDIR}"

  # Copy over all files in the tensorflow_quantum/ directory that are included in the BUILD
  # rule.
  mkdir "${TMPDIR}"/tensorflow_quantum
  cp -r -v ${EXPORT_DIR}/tensorflow_quantum/* "${TMPDIR}"/tensorflow_quantum/

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"

  python3 setup.py bdist_wheel ${EXTRA_FLAGS} > /dev/null

  cp dist/*.whl "${DEST}"
  popd
  rm -rf ${TMPDIR}
  echo "$(date) : === Done."
}

main "$@"
