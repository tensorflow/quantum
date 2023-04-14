load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "differentiators",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":adjoint",
        ":differentiator",
        ":linear_combination",
        ":parameter_shift",
        ":parameter_shift_util",
    ],
)

py_library(
    name = "adjoint",
    srcs = ["adjoint.py"],
    srcs_version = "PY3",
    deps = [
        ":differentiator",
        "//tensorflow_quantum/core/ops:tfq_adj_grad_op_py",
    ] + if_cuda_is_configured([
        "//tensorflow_quantum/core/ops:tfq_adj_grad_op_cuquantum_py",
    ]),
)

py_test(
    name = "adjoint_test",
    srcs = ["adjoint_test.py"],
    python_version = "PY3",
    deps = [
        ":adjoint",
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
    ],
)

py_library(
    name = "differentiator",
    srcs = ["differentiator.py"],
    srcs_version = "PY3",
)

py_library(
    name = "linear_combination",
    srcs = ["linear_combination.py"],
    srcs_version = "PY3",
    deps = [
        ":differentiator",
    ],
)

py_library(
    name = "parameter_shift",
    srcs = ["parameter_shift.py"],
    srcs_version = "PY3",
    deps = [
        ":differentiator",
        ":parameter_shift_util",
    ],
)

py_library(
    name = "parameter_shift_util",
    srcs = ["parameter_shift_util.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/ops:tfq_ps_util_ops_py",
    ],
)

py_test(
    name = "differentiator_test",
    srcs = ["differentiator_test.py"],
    python_version = "PY3",
    deps = [
        ":differentiator",
    ],
)

py_test(
    name = "linear_combination_test",
    srcs = ["linear_combination_test.py"],
    python_version = "PY3",
    deps = [
        ":linear_combination",
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "parameter_shift_test",
    timeout = "long",
    srcs = ["parameter_shift_test.py"],
    python_version = "PY3",
    deps = [
        ":parameter_shift",
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/core/ops:tfq_utility_ops_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "parameter_shift_util_test",
    srcs = ["parameter_shift_util_test.py"],
    python_version = "PY3",
    deps = [
        ":parameter_shift_util",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "gradient_test",
    timeout = "eternal",
    srcs = ["gradient_test.py"],
    python_version = "PY3",
    deps = [
        ":adjoint",
        ":linear_combination",
        ":parameter_shift",
        "//tensorflow_quantum/core/ops:batch_util",
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/core/ops/noise:noisy_expectation_op_py",
        "//tensorflow_quantum/core/ops/noise:noisy_sampled_expectation_op_py",
        "//tensorflow_quantum/python:util",
    ],
)
