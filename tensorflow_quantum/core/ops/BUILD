load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        ":windows": [],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

py_library(
    name = "ops",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":batch_util",
        ":circuit_execution_ops",
        ":cirq_ops",
        ":load_module",
        ":tfq_adj_grad_op_py",
        ":tfq_ps_util_ops_py",
        ":tfq_simulate_ops_py",
        ":tfq_unitary_op_py",
        ":tfq_utility_ops_py",
        # test addons
        "//tensorflow_quantum/core/ops/math_ops:inner_product_op_py",
        "//tensorflow_quantum/core/ops/math_ops:fidelity_op_py",
        "//tensorflow_quantum/core/ops/noise:noisy_expectation_op_py",
    ] + if_cuda_is_configured([
        ":tfq_simulate_ops_cuda_py",
        ":tfq_simulate_ops_cuquantum_py",
        ":tfq_adj_grad_op_cuquantum_py",
    ]),
)

cc_binary(
    name = "_tfq_adj_grad.so",
    srcs = [
        "tfq_adj_grad_op.cc",
    ],
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
    }),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":parse_context",
        # cirq cc proto
        # pauli sum cc proto
        # projector sum cc proto
        ":tfq_simulate_utils",
        "//tensorflow_quantum/core/src:adj_util",
        "//tensorflow_quantum/core/src:circuit_parser_qsim",
        "//tensorflow_quantum/core/src:util_qsim",
        "@qsim//lib:qsim_lib",
        # tensorflow core framework
        # tensorflow core lib
        # tensorflow core protos
    ],
)

cc_binary(
    name = "_tfq_ps_utils.so",
    srcs = [
        "tfq_ps_decompose_op.cc",
        "tfq_ps_symbol_replace_op.cc",
        "tfq_ps_weights_from_symbols_op.cc",
    ],
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
    }),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":parse_context",
        ":tfq_simulate_utils",
        # cirq cc proto
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "_tfq_simulate_ops.so",
    srcs = [
        "tfq_simulate_expectation_op.cc",
        "tfq_simulate_sampled_expectation_op.cc",
        "tfq_simulate_samples_op.cc",
        "tfq_simulate_state_op.cc",
    ],
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
    }),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":parse_context",
        ":tfq_simulate_utils",
        # cirq cc proto
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/proto:projector_sum_cc_proto",
        "//tensorflow_quantum/core/src:circuit_parser_qsim",
        "//tensorflow_quantum/core/src:program_resolution",
        "//tensorflow_quantum/core/src:util_qsim",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@qsim//lib:qsim_lib",
    ],
)

cc_binary(
    name = "_tfq_utility_ops.so",
    srcs = [
        "tfq_circuit_append_op.cc",
        "tfq_resolve_parameters_op.cc",
    ],
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
    }),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":parse_context",
        ":tfq_simulate_utils",
        # cirq cc proto
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/src:program_resolution",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "parse_context",
    srcs = ["parse_context.cc"],
    hdrs = ["parse_context.h"],
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
    }),
    deps = [
        ":tfq_simulate_utils",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/proto:projector_sum_cc_proto",
        "//tensorflow_quantum/core/src:program_resolution",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "_tfq_calculate_unitary_op.so",
    srcs = [
        "tfq_calculate_unitary_op.cc",
    ],
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
    }),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        ":parse_context",
        # cirq cc proto
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/proto:projector_sum_cc_proto",
        "//tensorflow_quantum/core/src:circuit_parser_qsim",
        "//tensorflow_quantum/core/src:util_qsim",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@qsim//lib:qsim_lib",
    ],
)

cc_library(
    name = "tfq_simulate_utils",
    srcs = ["tfq_simulate_utils.cc"],
    hdrs = ["tfq_simulate_utils.h"],
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
        ],
        "//conditions:default": [
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
        ],
    }),
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "tfq_adj_grad_op_py",
    srcs = ["tfq_adj_grad_op.py"],
    data = [":_tfq_adj_grad.so"],
    srcs_version = "PY3",
    deps = [
        ":load_module",
        # pauli sum cc proto
        # projector sum cc proto
        # tensorflow framework for wrappers
    ],
)

py_library(
    name = "tfq_unitary_op_py",
    srcs = ["tfq_unitary_op.py"],
    data = [":_tfq_calculate_unitary_op.so"],
    srcs_version = "PY3",
    deps = [
        # tensorflow framework for wrappers
        ":load_module",
        ":tfq_utility_ops_py",
        "//tensorflow_quantum/python:quantum_context",
    ],
)

py_test(
    name = "tfq_adj_grad_op_test",
    srcs = ["tfq_adj_grad_op_test.py"],
    python_version = "PY3",
    deps = [
        ":tfq_adj_grad_op_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "tfq_unitary_op_test",
    timeout = "long",
    srcs = ["tfq_unitary_op_test.py"],
    python_version = "PY3",
    deps = [
        ":tfq_unitary_op_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "tfq_simulate_ops_py",
    srcs = ["tfq_simulate_ops.py"],
    data = [":_tfq_simulate_ops.so"],
    srcs_version = "PY3",
    deps = [
        # tensorflow framework for wrappers
        ":load_module",
    ],
)

py_test(
    name = "tfq_simulate_ops_test",
    srcs = ["tfq_simulate_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":tfq_simulate_ops_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "circuit_execution_ops",
    srcs = ["circuit_execution_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":cirq_ops",
        ":tfq_simulate_ops_py",
        ":tfq_simulate_ops_cuquantum_py",
        ":tfq_utility_ops_py",
        "//tensorflow_quantum/python:quantum_context",
    ],
)

py_test(
    name = "circuit_execution_ops_test",
    timeout = "eternal",
    srcs = ["circuit_execution_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":circuit_execution_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "cirq_ops",
    srcs = ["cirq_ops.py"],
    srcs_version = "PY3",
    deps = [
        ":batch_util",
        "//tensorflow_quantum/core/proto:program_py_proto",
        "//tensorflow_quantum/core/proto:pauli_sum_py_proto",
        "//tensorflow_quantum/core/proto:projector_sum_py_proto",
        "//tensorflow_quantum/core/serialize:serializer",
    ],
)

py_test(
    name = "cirq_ops_test",
    srcs = ["cirq_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":batch_util",
        ":cirq_ops",
        "//tensorflow_quantum/core/serialize:serializer",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "batch_util",
    srcs = ["batch_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/serialize:serializer",
    ],
)

py_test(
    name = "batch_util_test",
    timeout = "eternal",
    srcs = ["batch_util_test.py"],
    python_version = "PY3",
    deps = [
        ":batch_util",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "tfq_ps_util_ops_py",
    srcs = ["tfq_ps_util_ops.py"],
    data = [":_tfq_ps_utils.so"],
    srcs_version = "PY3",
    deps = [
        # tensorflow framework for wrappers
        ":load_module",
    ],
)

py_test(
    name = "tfq_ps_util_ops_test",
    srcs = ["tfq_ps_util_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":tfq_ps_util_ops_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "tfq_utility_ops_py",
    srcs = ["tfq_utility_ops.py"],
    data = [":_tfq_utility_ops.so"],
    srcs_version = "PY3",
    deps = [
        # tensorflow framework for wrappers
        ":load_module",
    ],
)

py_test(
    name = "tfq_utility_ops_test",
    srcs = ["tfq_utility_ops_test.py"],
    python_version = "PY3",
    deps = [
        ":tfq_utility_ops_py",
        "//tensorflow_quantum/core/serialize:serializer",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "tfq_simulate_ops_cuda_py",
    srcs = ["tfq_simulate_ops_cuda.py"],
    data = [
        ":_tfq_simulate_ops_cuda.so",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow framework for wrappers
        ":load_module",
    ],
)

py_library(
    name = "tfq_simulate_ops_cuquantum_py",
    srcs = ["tfq_simulate_ops_cuquantum.py"],
    data = [
        ":_tfq_simulate_ops_cuquantum.so",
    ],
    srcs_version = "PY3",
    deps = [
        # tensorflow framework for wrappers
        ":load_module",
    ],
)

py_test(
    name = "tfq_simulate_ops_cuda_test",
    srcs = ["tfq_simulate_ops_cuda_test.py"],
    deps = [
        ":tfq_simulate_ops_cuda_py",
        ":tfq_simulate_ops_py",
        "//tensorflow_quantum/python:util",
    ],
    srcs_version = "PY3",
    tags = ["cuda"],
)

cc_binary(
    name = "_tfq_simulate_ops_cuda.so",
    srcs = [
        "tfq_simulate_expectation_op_cuda.cu.cc",
    ],
    linkshared = 1,
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
            "/D__CUDA__",
        ],
        "//conditions:default": [
            "-Iexternal/local_cuda/cuda/include",
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-O3",
            "-Iexternal/cuda_headers",
            "-DNV_CUDNN_DISABLE_EXCEPTION",
            # "-fpermissive",
        ],
    }) + if_cuda_is_configured([
        "-DTENSORFLOW_USE_NVCC=1",
        "-DGOOGLE_CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
        "-D__CUDA__",
    ]),
    deps = [
        # cirq cc proto
        "//tensorflow_quantum/core/ops:parse_context",
        "//tensorflow_quantum/core/ops:tfq_simulate_utils",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/src:circuit_parser_qsim",
        "//tensorflow_quantum/core/src:util_qsim",
        "@eigen//:eigen3",
        # "@local_cuda//:cuda_headers"
        # tensorflow core framework
        # tensorflow core lib
        # tensorflow core protos
    ] + if_cuda_is_configured([
        ":cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@qsim//lib:qsim_cuda_lib",
    ]),
    # alwayslink=1,
)

py_test(
    name = "tfq_simulate_ops_cuquantum_test",
    srcs = ["tfq_simulate_ops_cuquantum_test.py"],
    deps = [
        ":tfq_simulate_ops_cuquantum_py",
        ":tfq_simulate_ops_py",
        "//tensorflow_quantum/python:util",
    ],
    srcs_version = "PY3",
    tags = ["cuquantum"],
)

cc_binary(
    name = "_tfq_simulate_ops_cuquantum.so",
    srcs = [
        "tfq_simulate_expectation_op_cuquantum.cu.cc",
        "tfq_simulate_sampled_expectation_op_cuquantum.cu.cc",
        "tfq_simulate_samples_op_cuquantum.cu.cc",
        "tfq_simulate_state_op_cuquantum.cu.cc",
    ],
    linkshared = 1,
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
            "/D__CUSTATEVEC__",
        ],
        "//conditions:default": [
            "-Iexternal/local_cuda/cuda/include",
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-O3",
            "-Iexternal/cuda_headers",
            "-DNV_CUDNN_DISABLE_EXCEPTION",
            # "-fpermissive",
        ],
    }) + if_cuda_is_configured([
        "-DTENSORFLOW_USE_NVCC=1",
        "-DGOOGLE_CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
        "-D__CUSTATEVEC__",
    ]),
    deps = [
        # cirq cc proto
        "//tensorflow_quantum/core/ops:parse_context",
        "//tensorflow_quantum/core/ops:tfq_simulate_utils",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/src:circuit_parser_qsim",
        "//tensorflow_quantum/core/src:util_qsim",
        "@eigen//:eigen3",
        # "@local_cuda//:cuda_headers"
        # tensorflow core framework
        # tensorflow core lib
        # tensorflow core protos
    ] + if_cuda_is_configured([
        ":cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuquantum//:cuquantum_headers",
        "@local_config_cuquantum//:libcuquantum",
        "@qsim//lib:qsim_cuquantum_lib",
    ]),
    # alwayslink=1,
)

cc_binary(
    name = "_tfq_adj_grad_cuquantum.so",
    srcs = [
        "tfq_adj_grad_op_cuquantum.cu.cc",
    ],
    linkshared = 1,
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    copts = select({
        ":windows": [
            "/D__CLANG_SUPPORT_DYN_ANNOTATION__",
            "/D_USE_MATH_DEFINES",
            "/DEIGEN_MPL2_ONLY",
            "/DEIGEN_MAX_ALIGN_BYTES=64",
            "/DEIGEN_HAS_TYPE_TRAITS=0",
            "/DTF_USE_SNAPPY",
            "/showIncludes",
            "/MD",
            "/O2",
            "/DNDEBUG",
            "/w",
            "-DWIN32_LEAN_AND_MEAN",
            "-DNOGDI",
            "/d2ReducedOptimizeHugeFunctions",
            "/arch:AVX",
            "/std:c++17",
            "-DTENSORFLOW_MONOLITHIC_BUILD",
            "/DPLATFORM_WINDOWS",
            "/DEIGEN_HAS_C99_MATH",
            "/DTENSORFLOW_USE_EIGEN_THREADPOOL",
            "/DEIGEN_AVOID_STL_ARRAY",
            "/Iexternal/gemmlowp",
            "/wd4018",
            "/wd4577",
            "/DNOGDI",
            "/UTF_COMPILE_LIBRARY",
            "/D__CUSTATEVEC__",
        ],
        "//conditions:default": [
            "-Iexternal/local_cuda/cuda/include",
            "-pthread",
            "-std=c++17",
            "-D_GLIBCXX_USE_CXX11_ABI=1",
            "-O3",
            "-Iexternal/cuda_headers",
            "-DNV_CUDNN_DISABLE_EXCEPTION",
            # "-fpermissive",
        ],
    }) + if_cuda_is_configured([
        "-DTENSORFLOW_USE_NVCC=1",
        "-DGOOGLE_CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
        "-D__CUSTATEVEC__",
    ]),
    deps = [
        "//tensorflow_quantum/core/ops:parse_context",
        "//tensorflow_quantum/core/src:util_qsim",
        "//tensorflow_quantum/core/src:adj_util",
        # "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        # "//tensorflow_quantum/core/proto:program_cc_proto",
        # "//tensorflow_quantum/core/src:circuit_parser_qsim",
        # "@eigen//:eigen3",
    ] + if_cuda_is_configured([
        ":cuda",
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuquantum//:cuquantum_headers",
        "@local_config_cuquantum//:libcuquantum",
        "@qsim//lib:qsim_cuquantum_lib",
    ]),
    # alwayslink=1,
)

py_library(
    name = "tfq_adj_grad_op_cuquantum_py",
    srcs = ["tfq_adj_grad_op_cuquantum.py"],
    data = [":_tfq_adj_grad_cuquantum.so"],
    srcs_version = "PY3",
    deps = [
        ":load_module",
        # pauli sum cc proto
        # projector sum cc proto
        # tensorflow framework for wrappers
    ],
)

py_test(
    name = "tfq_adj_grad_op_cuquantum_test",
    srcs = ["tfq_adj_grad_op_cuquantum_test.py"],
    python_version = "PY3",
    deps = [
        ":tfq_adj_grad_op_cuquantum_py",
        ":tfq_adj_grad_op_py", # for testing cpu vs gpu diff
        "//tensorflow_quantum/python:util",
    ],
    srcs_version = "PY3",
    tags = ["cuquantum"],
)

py_library(
    name = "load_module",
    srcs = ["load_module.py"],
    srcs_version = "PY3",
    deps = [],
)
