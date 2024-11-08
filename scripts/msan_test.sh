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
echo "Testing All Bazel cc_tests with msan.";

test_outputs=$(bazel test -c dbg --cxxopt="-g" --cxxopt="-fno-omit-frame-pointer" \
  --cxxopt="-fsanitize=address" --linkopt="-fsanitize=address" \
  --color=no --show_progress_rate_limit=0 --test_summary=short \
  --announce_rc --subcommands --verbose_failures --show_timestamps \
  --keep_going --test_output=errors \
  //tensorflow_quantum/core/src:all)
exit_code=$?
if [ "$exit_code" == "0" ]; then
	echo "Testing Complete!";
	exit 0;
else
	echo "Testing failed, please correct errors before proceeding."
	echo "{$test_outputs}"
	exit 64;
fi

# test_outputs=$(bazel test -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
#   --cxxopt="-msse2" --cxxopt="-msse3" --cxxopt="-msse4" \
#   --cxxopt="-fsanitize=address" --linkopt="-fsanitize=address" \
#   --cxxopt="-g" --cxxopt="-O0" \
#   --notest_keep_going --test_output=errors \
#   //tensorflow_quantum/core/src:all && \
#   bazel test -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
#   --cxxopt="-mavx2" --cxxopt="-mavx" --cxxopt="-mfma" \
#   --cxxopt="-fsanitize=address" --linkopt="-fsanitize=address" \
#   --cxxopt="-g" --cxxopt="-O0" \
#   --notest_keep_going --test_output=errors \
#   //tensorflow_quantum/core/src:all && \
#   bazel test -c opt --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=1" \
#   --cxxopt="-fsanitize=address" --linkopt="-fsanitize=address" \
#   --cxxopt="-g" --cxxopt="-O0" \
#   --notest_keep_going --test_output=errors \
#   //tensorflow_quantum/core/src:all)
# exit_code=$?
# if [ "$exit_code" == "0" ]; then
# 	echo "Testing Complete!";
# 	exit 0;
# else
# 	echo "Testing failed, please correct errors before proceeding."
# 	echo "{$test_outputs}"
# 	exit 64;
# fi
