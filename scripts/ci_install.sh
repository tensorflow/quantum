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

BAZEL_VERSION="6.5.0"

if [[ "$(uname -s)" == "Darwin" ]]; then
  if ! command -v bazel >/dev/null 2>&1 || [[ "$(bazel --version 2>/dev/null)" != *"${BAZEL_VERSION}"* ]]; then
    ARCH="$(uname -m)"
    if [[ "${ARCH}" == "arm64" ]]; then
      BAZEL_ARCH="arm64"
    else
      BAZEL_ARCH="x86_64"
    fi
    curl -fsSL -o /tmp/bazel \
      "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel-${BAZEL_VERSION}-darwin-${BAZEL_ARCH}"
    chmod +x /tmp/bazel
    sudo mkdir -p /usr/local/bin
    sudo mv /tmp/bazel /usr/local/bin/bazel
  fi
else
  wget "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel_${BAZEL_VERSION}-linux-x86_64.deb"
  sudo dpkg -i "bazel_${BAZEL_VERSION}-linux-x86_64.deb"
fi

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt