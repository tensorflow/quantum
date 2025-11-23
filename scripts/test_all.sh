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
test_outputs=$(bazel test -c opt --test_output=errors --cxxopt="-msse2" --cxxopt="-msse3" --cxxopt="-msse4" //tensorflow_quantum/...)
exit_code=$?
if [ "$exit_code" == "0" ]; then
	echo "Testing Complete!";
	exit 0;
else
	echo "Testing failed, please correct errors before proceeding."
	echo "{$test_outputs}"
	exit 64;
fi
