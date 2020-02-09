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

py_test(
    name = "state_test",
    srcs = ["state_test.py"],
    deps = [
        ":state",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "expectation_test",
    srcs = ["expectation_test.py"],
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
    deps = [
        ":sampled_expectation",
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/differentiators:linear_combination",
    ],
)

py_test(
    name = "sample_test",
    srcs = ["sample_test.py"],
    deps = [
        ":sample",
        "//tensorflow_quantum/python:util",
    ],
)
