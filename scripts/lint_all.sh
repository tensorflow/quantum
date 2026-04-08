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
echo "Checking for lint in python code...";
linting_outputs=$(pylint --rcfile .pylintrc ./tensorflow_quantum);
exit_code=$?
if [ "$exit_code" == "0" ]; then
	echo "Python linting complete!";
	exit 0;
else
	echo "Linting failed, please correct errors before proceeding."
	echo "{$linting_outputs}"
	exit 64;
fi

# TODO (mbbrough/pmassey): Is there an autolinter for C++ stuff we should put in here ?
# Yes we need to put clang-tidy in here!
