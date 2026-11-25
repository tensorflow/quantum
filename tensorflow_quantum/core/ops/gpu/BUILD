load("//tensorflow:tensorflow.bzl", "tf_cuda_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

tf_cuda_library(
    name = "_tfq_simulate_ops_cuda.so",
    srcs = [
        "tfq_simulate_expectation_op_cuda.cu.cc",
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
            "/std:c++14",
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
            "-std=c++14",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "-O3",
            "-DNV_CUDNN_DISABLE_EXCEPTION",
        ],
    }),
    features = select({
        ":windows": ["windows_export_all_symbols"],
        "//conditions:default": [],
    }),
    linkshared = 1,
    deps = [
        # cirq cc proto
        "//tensorflow_quantum/core/ops:parse_context",
        "//tensorflow_quantum/core/ops:tfq_simulate_utils",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/src:circuit_parser_qsim",
        "//tensorflow_quantum/core/src:util_qsim",
        "@qsim/lib:qsim_cuda_lib",
        "@eigen//:eigen3",
        # tensorflow core framework
        # tensorflow core lib
        # tensorflow core protos
    ],
)

tf_gen_op_wrapper_py(
    name = "gen_tfq_simulate_ops_cuda_py",
    out = "gen_tfq_simulate_ops_cuda.py",
    deps = [":_tfq_simulate_ops_cuda.so"],
)

py_binary(
    name = "gpu_benchmark",
    srcs = ["gpu_benchmark.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_cuda_py",
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_py",
        "//tensorflow_quantum/core/serialize:serializer",
        "//tensorflow_quantum/python:util",
    ],
)
