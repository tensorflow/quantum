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
echo "Testing All Bazel py_test and cc_tests.";
ENABLE_CUDA=${1}

if [[ ${ENABLE_CUDA} == "gpu" ]]; then
  echo "GPU mode. CUDA config is set."
  CUDA_CONFIG="--config=cuda"
  # Tests all including cuquantum ops.
  TAG_FILTER=""
else
  echo "CPU mode."
  CUDA_CONFIG=""
  # Tests cpu only excluding cuquantum ops.
  TAG_FILTER="--test_tag_filters=-cuquantum --build_tag_filters=-cuquantum"
fi

test_outputs=$(bazel test -c opt ${CUDA_CONFIG} ${TAG_FILTER} --experimental_repo_remote_exec --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --cxxopt="-std=c++17" --cxxopt="-msse2" --cxxopt="-msse3" --cxxopt="-msse4" --test_output=errors //tensorflow_quantum/...)
exit_code=$?
if [ "$exit_code" == "0" ]; then
	echo "Testing Complete!";
	exit 0;
else
	echo "Testing failed, please correct errors before proceeding."
	echo "{$test_outputs}"
	exit 64;
fi
