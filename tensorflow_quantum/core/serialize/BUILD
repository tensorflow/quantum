package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "serializer",
    srcs = ["serializer.py"],
    deps = [
        "//tensorflow_quantum/core/proto:pauli_sum_py_proto",
    ],
)

py_test(
    name = "serializer_test",
    srcs = ["serializer_test.py"],
    python_version = "PY3",
    deps = [
        ":serializer",
    ],
)
