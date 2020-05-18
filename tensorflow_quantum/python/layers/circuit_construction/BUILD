package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "elementary",
    srcs = ["elementary.py"],
    deps = [
        "//tensorflow_quantum/core/ops:tfq_utility_ops_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "unsigned",
    srcs = ["unsigned.py"],
    deps = [
        "//tensorflow_quantum/core/ops:tfq_utility_ops_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "elementary_test",
    srcs = ["elementary_test.py"],
    python_version = "PY3",
    deps = [
        ":elementary",
        "//tensorflow_quantum/python:util",
    ],
)

py_test(
    name = "unsigned_test",
    srcs = ["unsigned_test.py"],
    python_version = "PY3",
    deps = [
        ":unsigned",
        "//tensorflow_quantum/python:util",
    ],
)
