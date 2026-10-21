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
export TF_USE_LEGACY_KERAS=1

if [[ "$(uname -s)" == "Darwin" ]] && compgen -G "wheelhouse/*.whl" > /dev/null; then
  REPO_ROOT="$(pwd)"
  PY="${PYTHON_BIN_PATH:-python3}"
  VENV_DIR="$(mktemp -d -t tfq_venv.XXXXXXXXXX)"
  trap 'rm -rf "${VENV_DIR}"' EXIT
  "${PY}" -m venv "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install --upgrade pip
  "${VENV_DIR}/bin/pip" install tensorflow==2.19.1 "${REPO_ROOT}"/wheelhouse/*.whl
  cd ..
  cp quantum/scripts/import_test.py import_test.py
  "${VENV_DIR}/bin/python" import_test.py
else
  cd ..
  cp quantum/scripts/import_test.py import_test.py
  python import_test.py
fi