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
    name = "optimizers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":rotosolve_minimizer",
        ":spsa_minimizer",
    ],
)

py_library(
    name = "rotosolve_minimizer",
    srcs = ["rotosolve_minimizer.py"],
)

py_library(
    name = "spsa_minimizer",
    srcs = ["spsa_minimizer.py"],
)

py_test(
    name = "rotosolve_minimizer_test",
    srcs = ["rotosolve_minimizer_test.py"],
    python_version = "PY3",
    deps = [
        ":rotosolve_minimizer",
        "//tensorflow_quantum/core/ops:tfq_ps_util_ops_py",
        "//tensorflow_quantum/python/layers/high_level:pqc",
    ],
)

py_test(
    name = "spsa_minimizer_test",
    srcs = ["spsa_minimizer_test.py"],
    flaky = True,
    python_version = "PY3",
    deps = [
        ":spsa_minimizer",
        "//tensorflow_quantum/core/ops:tfq_ps_util_ops_py",
        "//tensorflow_quantum/python/layers/high_level:pqc",
    ],
)
