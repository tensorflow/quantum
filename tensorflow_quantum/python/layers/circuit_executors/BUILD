package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "state",
    srcs = ["state.py"],
    deps = [
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "expectation",
    srcs = ["expectation.py"],
    deps = [
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/differentiators:differentiator",
        "//tensorflow_quantum/python/differentiators:linear_combination",
    ],
)

py_library(
    name = "sampled_expectation",
    srcs = ["sampled_expectation.py"],
    deps = [
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/differentiators:differentiator",
        "//tensorflow_quantum/python/differentiators:linear_combination",
        "//tensorflow_quantum/python/differentiators:parameter_shift",
    ],
)

py_library(
    name = "sample",
    srcs = ["sample.py"],
    deps = [
        "//tensorflow_quantum/core/ops:circuit_execution_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "input_checks",
    srcs = ["input_checks.py"],
    deps = [
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "state_test",
    srcs = ["state_test.py"],
    python_version = "PY3",
    deps = [
        ":state",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "expectation_test",
    srcs = ["expectation_test.py"],
    python_version = "PY3",
    deps = [
        ":expectation",
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/differentiators:linear_combination",
    ],
)

py_test(
    name = "sampled_expectation_test",
    timeout = "eternal",
    srcs = ["sampled_expectation_test.py"],
    python_version = "PY3",
    deps = [
        ":sampled_expectation",
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/differentiators:linear_combination",
    ],
)

py_test(
    name = "sample_test",
    srcs = ["sample_test.py"],
    python_version = "PY3",
    deps = [
        ":sample",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "input_checks_test",
    srcs = ["input_checks_test.py"],
    python_version = "PY3",
    deps = [
        ":input_checks",
        "//tensorflow_quantum/python:util",
    ],
)
