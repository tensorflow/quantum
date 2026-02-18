# Copyright 2020 The TensorFlow Quantum Authors
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
    name = "high_level",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":controlled_pqc",
        ":noisy_controlled_pqc",
        ":pqc",
        ":noisy_pqc",
    ],
)

py_library(
    name = "controlled_pqc",
    srcs = ["controlled_pqc.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/layers/circuit_construction:elementary",
        "//tensorflow_quantum/python/layers/circuit_executors:expectation",
        "//tensorflow_quantum/python/layers/circuit_executors:sampled_expectation",
    ],
)

py_library(
    name = "noisy_controlled_pqc",
    srcs = ["noisy_controlled_pqc.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/layers/circuit_construction:elementary",
        "//tensorflow_quantum/core/ops/noise:noisy_expectation_op_py",
        "//tensorflow_quantum/core/ops/noise:noisy_sampled_expectation_op_py",
        "//tensorflow_quantum/python/differentiators:parameter_shift",
    ],
)

py_library(
    name = "pqc",
    srcs = ["pqc.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/layers/circuit_construction:elementary",
        "//tensorflow_quantum/python/layers/circuit_executors:expectation",
        "//tensorflow_quantum/python/layers/circuit_executors:sampled_expectation",
    ],
)

py_library(
    name = "noisy_pqc",
    srcs = ["noisy_pqc.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/layers/circuit_construction:elementary",
        "//tensorflow_quantum/core/ops/noise:noisy_expectation_op_py",
        "//tensorflow_quantum/core/ops/noise:noisy_sampled_expectation_op_py",
        "//tensorflow_quantum/python/differentiators:parameter_shift",
    ],
)

py_test(
    name = "controlled_pqc_test",
    srcs = ["controlled_pqc_test.py"],
    python_version = "PY3",
    deps = [
        ":controlled_pqc",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "pqc_test",
    srcs = ["pqc_test.py"],
    python_version = "PY3",
    deps = [
        ":pqc",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "noisy_controlled_pqc_test",
    srcs = ["noisy_controlled_pqc_test.py"],
    python_version = "PY3",
    deps = [
        ":noisy_controlled_pqc",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "noisy_pqc_test",
    srcs = ["noisy_pqc_test.py"],
    python_version = "PY3",
    deps = [
        ":noisy_pqc",
        "//tensorflow_quantum/python:util",
    ],
)
