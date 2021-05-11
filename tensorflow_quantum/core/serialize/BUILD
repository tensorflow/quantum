package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "serialize",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [":serializer"],
)

py_library(
    name = "serializer",
    srcs = ["serializer.py"],
    srcs_version = "PY3",
    deps = [
        # cirq proto
        "//tensorflow_quantum/core/proto:pauli_sum_py_proto",
    ],
)

py_test(
    name = "serializer_test",
    srcs = ["serializer_test.py"],
    python_version = "PY3",
    deps = [
        ":serializer",
        # cirq proto
        "//tensorflow_quantum/core/proto:pauli_sum_py_proto",
    ],
)
