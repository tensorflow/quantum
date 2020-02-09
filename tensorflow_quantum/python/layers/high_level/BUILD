package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "controlled_pqc",
    srcs = ["controlled_pqc.py"],
    deps = [
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/layers/circuit_construction:elementary",
        "//tensorflow_quantum/python/layers/circuit_executors:expectation",
        "//tensorflow_quantum/python/layers/circuit_executors:sampled_expectation",
    ],
)

py_library(
    name = "pqc",
    srcs = ["pqc.py"],
    deps = [
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/layers/circuit_construction:elementary",
        "//tensorflow_quantum/python/layers/circuit_executors:expectation",
        "//tensorflow_quantum/python/layers/circuit_executors:sampled_expectation",
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
