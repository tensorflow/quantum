package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "layers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow_quantum/python/layers/circuit_construction",
        "//third_party/py/tensorflow_quantum/python/layers/circuit_executors",
        "//third_party/py/tensorflow_quantum/python/layers/high_level",
    ],
)
