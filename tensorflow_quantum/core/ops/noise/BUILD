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

# load op_wrapper

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])


cc_binary(
    name = "_tfq_noise_ops.so",
    srcs = [
        "tfq_noisy_expectation.cc",
        "tfq_noisy_sampled_expectation.cc",
        "tfq_noisy_samples.cc"
    ],
    # copts = select({
    #     ":windows": [
    #         "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
    #         "/D_USE_MATH_DEFINES",
    #         "/DEIGEN_MPL2_ONLY",
    #         "/DEIGEN_MAX_ALIGN_BYTES=64",
    #         "/DEIGEN_HAS_TYPE_TRAITS=0",
    #         "/DTF_USE_SNAPPY",
    #         "/showIncludes",
    #         "/MD",
    #         "/O2",
    #         "/DNDEBUG",
    #         "/w",
    #         "-DWIN32_LEAN_AND_MEAN",
    #         "-DNOGDI",
    #         "/d2ReducedOptimizeHugeFunctions",
    #         "/arch:AVX",
    #         "/std:c++17",
    #         "-DTENSORFLOW_MONOLITHIC_BUILD",
    #         "/DPLATFORM_WINDOWS",
    #         "/DEIGEN_HAS_C99_MATH",
    #         "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
    #         "/DEIGEN_AVOID_STL_ARRAY",
    #         "/Iexternal/gemmlowp",
    #         "/wd4018",
    #         "/wd4577",
    #         "/DNOGDI",
    #         "/UTF_COMPILE_LIBRARY",
    #     ],
    #     "//conditions:default": [
    #         "-pthread",
    #         "-std=c++17",
    #         "-D_GLIBCXX_USE_CXX11_ABI=1",
    #     ],
    # }),
    # features = select({
    #     ":windows": ["windows_export_all_symbols"],
    #     "//conditions:default": [],
    # }),
    linkshared = 1,
    deps = [
        # cirq cc proto
        "//tensorflow_quantum/core/ops:parse_context",
        "//tensorflow_quantum/core/ops:tfq_simulate_utils",
        "//tensorflow_quantum/core/src:circuit_parser_qsim",
        "//tensorflow_quantum/core/src:util_balance_trajectory",
        "//tensorflow_quantum/core/src:util_qsim",
        "@qsim//lib:qsim_lib",
        # tensorflow core framework
        # tensorflow core lib
        # tensorflow core protos
    ],
)

py_library(
    name = "noisy_expectation_op_py",
    srcs = ["noisy_expectation_op.py"],
    data = [":_tfq_noise_ops.so"],
    deps = [
        "//tensorflow_quantum/core/ops:load_module",
    ],
)

py_test(
    name = "noisy_expectation_op_test",
    srcs = ["noisy_expectation_op_test.py"],
    python_version = "PY3",
    deps = [
        ":noisy_expectation_op_py",
        "//tensorflow_quantum/core/ops:batch_util",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "noisy_sampled_expectation_op_py",
    srcs = ["noisy_sampled_expectation_op.py"],
    data = [":_tfq_noise_ops.so"],
    deps = [
        "//tensorflow_quantum/core/ops:load_module",
    ],
)

py_test(
    name = "noisy_sampled_expectation_op_test",
    srcs = ["noisy_sampled_expectation_op_test.py"],
    python_version = "PY3",
    deps = [
        ":noisy_sampled_expectation_op_py",
        "//tensorflow_quantum/core/ops:batch_util",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "noisy_samples_op_py",
    srcs = ["noisy_samples_op.py"],
    data = [":_tfq_noise_ops.so"],
    deps = [
        "//tensorflow_quantum/core/ops:load_module",
        "//tensorflow_quantum/core/ops:tfq_utility_ops_py",
    ],
)

py_test(
    name = "noisy_samples_op_test",
    srcs = ["noisy_samples_op_test.py"],
    python_version = "PY3",
    deps = [
        ":noisy_samples_op_py",
        "//tensorflow_quantum/core/ops:batch_util",
        "//tensorflow_quantum/python:util",
    ],
)

