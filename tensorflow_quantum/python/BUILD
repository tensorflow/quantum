package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "quantum_context",
    srcs = ["quantum_context.py"]
)

py_test(
    name = "quantum_context_test",
    srcs = ["quantum_context_test.py"],
    deps = [":quantum_context"]
)

py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        "//tensorflow_quantum/core/serialize:serializer",
    ],
)

py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    python_version = "PY3",
    deps = [
        ":util",
        "//tensorflow_quantum/core/serialize:serializer",
    ],
)
