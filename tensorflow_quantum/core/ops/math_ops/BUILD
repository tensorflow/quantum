# load op_wrapper

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

cc_binary(
    name = "_tfq_math_ops.so",
    srcs = [
        "tfq_inner_product.cc",
        "tfq_inner_product_grad.cc",
        "tfq_simulate_1d_expectation.cc",
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
            "-std=c++11",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
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
        "//tensorflow_quantum/core/src:adj_util",
        "//tensorflow_quantum/core/src:circuit_parser_qsim",
        "//tensorflow_quantum/core/src:util_qsim",
        "@qsim//lib:mps_simulator",
        "@qsim//lib:mps_statespace",
        "@qsim//lib:qsim_lib",
        "@eigen//:eigen3",
        # tensorflow core framework
        # tensorflow core lib
        # tensorflow core protos
    ],
)

py_library(
    name = "fidelity_op_py",
    srcs = ["fidelity_op.py"],
    deps = [
        ":inner_product_op_py",
    ],
)

py_library(
    name = "inner_product_op_py",
    srcs = ["inner_product_op.py"],
    data = [":_tfq_math_ops.so"],
    deps = [
        "//tensorflow_quantum/core/ops:load_module",
    ],
)

py_test(
    name = "inner_product_op_test",
    srcs = ["inner_product_op_test.py"],
    python_version = "PY3",
    deps = [
        ":inner_product_op_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "fidelity_op_test",
    srcs = ["fidelity_op_test.py"],
    deps = [
        ":fidelity_op_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "inner_product_grad_test",
    srcs = ["inner_product_grad_test.py"],
    python_version = "PY3",
    deps = [
        ":inner_product_op_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "simulate_mps_py",
    srcs = ["simulate_mps.py"],
    data = [":_tfq_math_ops.so"],
    deps = [
        "//tensorflow_quantum/core/ops:load_module",
    ],
)

py_test(
    name = "simulate_mps_test",
    srcs = ["simulate_mps_test.py"],
    python_version = "PY3",
    deps = [
        ":simulate_mps_py",
        "//tensorflow_quantum/python:util",
    ],
)
