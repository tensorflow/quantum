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
#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON_BIN_PATH:-python3}"
PIP="$PY -m pip"
LOG_FILE="${LOG_FILE:-tutorials_run.log}"

export TF_CPP_MIN_LOG_LEVEL=1
export TF_USE_LEGACY_KERAS=1
export SDL_VIDEODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

# Jupyter stack
$PIP install --no-cache-dir -U \
  ipython==8.26.0 ipykernel==6.29.5 jupyter-client==8.6.0 nbclient==0.9.0

# Tutorial deps
$PIP install --no-cache-dir -U seaborn==0.12.2
$PIP install --no-cache-dir -U gym==0.26.2 shimmy==0.2.1
$PIP install --no-cache-dir -q git+https://github.com/tensorflow/docs

# Kernel for this interpreter
KERNEL_NAME="tfq-py"
echo "==[ci_validate_tutorials] Installing ipykernel '${KERNEL_NAME}'"
$PY -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python (tfq)"
KERNEL_DIR="$("$PY" - <<'PY'
import os
home=os.path.expanduser("~")
cand=[os.path.join(home,".local/share/jupyter/kernels"),
      os.path.join(home,"Library/Jupyter/kernels"),
      os.path.join("/usr/local/share/jupyter/kernels")]
print(next((p for p in cand if os.path.isdir(p)), os.getcwd()))
PY
)"
echo "==[ci_validate_tutorials] Kernel installed at: ${KERNEL_DIR}/${KERNEL_NAME}"

# More headroom just in case
export NB_KERNEL_NAME="$KERNEL_NAME"
export NBCLIENT_TIMEOUT="${NBCLIENT_TIMEOUT:-1800}"

echo "==[ci_validate_tutorials] Launching test_tutorials.py with $PY (kernel=${KERNEL_NAME})"
cd ..
( set -o pipefail; "$PY" quantum/scripts/test_tutorials.py 2>&1 | tee "${LOG_FILE}" )
status="${PIPESTATUS[0]}"

if [[ "$status" == "0" ]]; then
  echo "==[ci_validate_tutorials] Tutorials completed successfully."
  exit 0
else
  echo "==[ci_validate_tutorials] Tutorials failed. See ${LOG_FILE}"
  exit 64
fi

