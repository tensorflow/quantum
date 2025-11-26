# Copyright 2025 The TensorFlow Quantum Authors
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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_state",
        ":spin_system",
    ],
)

py_library(
    name = "cluster_state",
    srcs = ["cluster_state.py"],
    srcs_version = "PY3",
)

py_library(
    name = "spin_system",
    srcs = ["spin_system.py"],
    srcs_version = "PY3",
)

py_test(
    name = "cluster_state_test",
    srcs = ["cluster_state_test.py"],
    python_version = "PY3",
    deps = [
        ":cluster_state",
    ],
)

py_test(
    name = "spin_system_test",
    timeout = "eternal",
    srcs = ["spin_system_test.py"],
    python_version = "PY3",
    deps = [
        ":spin_system",
    ],
)
