package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "differentiator",
    srcs = ["differentiator.py"],
)

py_library(
    name = "linear_combination",
    srcs = ["linear_combination.py"],
    deps = [
        ":differentiator",
    ],
)

py_library(
    name = "parameter_shift",
    srcs = ["parameter_shift.py"],
    deps = [
        ":differentiator",
        ":parameter_shift_util",
    ],
)

py_library(
    name = "parameter_shift_util",
    srcs = ["parameter_shift_util.py"],
    deps = [
        "//tensorflow_quantum/core/ops:tfq_ps_util_ops_py",
    ],
)

py_library(
    name = "sampling_op_parameter_shift",
    srcs = ["sampling_op_parameter_shift.py"],
    deps = [
        ":parameter_shift_util",
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
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
    name = "sampling_op_parameter_shift_test",
    timeout = "eternal",
    srcs = ["sampling_op_parameter_shift_test.py"],
    python_version = "PY3",
    deps = [
        ":sampling_op_parameter_shift",
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "gradient_test",
    timeout = "eternal",
    srcs = ["gradient_test.py"],
    python_version = "PY3",
    deps = [
        ":linear_combination",
        ":parameter_shift",
        ":stochastic_differentiator",
        "//tensorflow_quantum/core/ops:batch_util",
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "stochastic_differentiator",
    srcs = ["stochastic_differentiator.py"],
    deps = [
        ":differentiator",
        ":stochastic_differentiator_util",
    ],
)

py_test(
    name = "stochastic_differentiator_test",
    srcs = ["stochastic_differentiator_test.py"],
    python_version = "PY3",
    deps = [
        ":stochastic_differentiator",
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "stochastic_differentiator_util",
    srcs = ["stochastic_differentiator_util.py"],
    deps = [
        ":parameter_shift_util",
        "//tensorflow_quantum/core/ops:cirq_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "stochastic_differentiator_util_test",
    srcs = ["stochastic_differentiator_util_test.py"],
    python_version = "PY3",
    deps = [
        ":stochastic_differentiator_util",
    ],
)
