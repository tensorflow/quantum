package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_test(
    name = "convergence_test",
    srcs = ["convergence_test.py"],
    deps = [
        "//tensorflow_quantum/core/ops:batch_util",
        "//tensorflow_quantum/core/ops:cirq_ops",
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_py",
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/differentiators:stochastic_differentiator",
    ],
)
