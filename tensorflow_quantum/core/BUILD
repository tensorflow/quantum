package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "core",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/ops",
        "//tensorflow_quantum/core/proto:pauli_sum_py_proto",
        "//tensorflow_quantum/core/serialize:serializer",
        "//tensorflow_quantum/core/src",
    ],
)
