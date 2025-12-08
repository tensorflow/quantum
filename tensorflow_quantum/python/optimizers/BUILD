package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "optimizers",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":rotosolve_minimizer",
        ":spsa_minimizer",
    ],
)

py_library(
    name = "rotosolve_minimizer",
    srcs = ["rotosolve_minimizer.py"],
)

py_library(
    name = "spsa_minimizer",
    srcs = ["spsa_minimizer.py"],
)

py_test(
    name = "rotosolve_minimizer_test",
    srcs = ["rotosolve_minimizer_test.py"],
    python_version = "PY3",
    deps = [
        ":rotosolve_minimizer",
        "//tensorflow_quantum/core/ops:tfq_ps_util_ops_py",
        "//tensorflow_quantum/python/layers/high_level:pqc",
    ],
)

py_test(
    name = "spsa_minimizer_test",
    srcs = ["spsa_minimizer_test.py"],
    flaky = True,
    python_version = "PY3",
    deps = [
        ":spsa_minimizer",
        "//tensorflow_quantum/core/ops:tfq_ps_util_ops_py",
        "//tensorflow_quantum/python/layers/high_level:pqc",
    ],
)
