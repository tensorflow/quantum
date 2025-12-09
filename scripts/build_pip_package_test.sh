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

pip install -r requirements.txt

# cd tensorflow_quantum
echo "Y\n" | ./configure.sh

bazel build -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" --cxxopt="-msse2" --cxxopt="-msse3" --cxxopt="-msse4" release:build_pip_package
rm /tmp/tensorflow_quantum/* || echo ok
bazel-bin/release/build_pip_package /tmp/tensorflow_quantum/
pip install -U /tmp/tensorflow_quantum/*.whl
