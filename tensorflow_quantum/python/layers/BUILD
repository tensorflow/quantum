package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "layers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//tensorflow_quantum/python/layers/circuit_construction",
        "//tensorflow_quantum/python/layers/circuit_executors",
        "//tensorflow_quantum/python/layers/high_level",
    ],
)
