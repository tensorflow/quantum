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

pip install -r requirements.txt

# cd tensorflow_quantum
printf "Y\n" | ./configure.sh

arch_cxxopts=()
if [[ "$(uname -m)" == "x86_64" ]]; then
  arch_cxxopts+=(--cxxopt="-msse2" --cxxopt="-msse3" --cxxopt="-msse4")
fi

bazel build -c opt "${arch_cxxopts[@]}" release:build_pip_package
rm -f /tmp/tensorflow_quantum/* || echo ok
bazel-bin/release/build_pip_package /tmp/tensorflow_quantum/

if [[ "$(uname -s)" == "Darwin" ]]; then
  rm -rf wheelhouse
  ./release/clean_distribution.sh /tmp/tensorflow_quantum/*.whl
  expected_tag="macosx_12_0_$(uname -m)"
  repaired_wheels=(wheelhouse/*-"${expected_tag}".whl)
  if [[ ! -f "${repaired_wheels[0]}" ]]; then
    echo "ERROR: Expected repaired wheel with tag ${expected_tag} in wheelhouse/" >&2
    ls -la wheelhouse/ >&2
    exit 1
  fi
  check_dir="$(mktemp -d -t tfq_whl_check.XXXXXXXXXX)"
  unzip -q "${repaired_wheels[0]}" -d "${check_dir}"
  while IFS= read -r -d '' so_file; do
    if ! otool -l "${so_file}" | grep -q "minos 12.0"; then
      echo "ERROR: Expected minos 12.0 in ${so_file}" >&2
      otool -l "${so_file}" | grep -A 4 "LC_BUILD_VERSION" >&2 || true
      rm -rf "${check_dir}"
      exit 1
    fi
  done < <(find "${check_dir}/tensorflow_quantum" -name "*.so" -print0)
  rm -rf "${check_dir}"
  pip install -U "${repaired_wheels[0]}"
else
  pip install -U /tmp/tensorflow_quantum/*.whl
fi
