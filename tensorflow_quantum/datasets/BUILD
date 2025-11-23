package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "datasets",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":cluster_state",
        ":spin_system",
    ],
)

py_library(
    name = "cluster_state",
    srcs = ["cluster_state.py"],
    srcs_version = "PY3",
    deps = ["@pypi//tensorflow"],
)

py_library(
    name = "spin_system",
    srcs = ["spin_system.py"],
    srcs_version = "PY3",
    deps = ["@pypi//tensorflow"],
)

py_test(
    name = "cluster_state_test",
    srcs = ["cluster_state_test.py"],
    python_version = "PY3",
    deps = [
        ":cluster_state",
    ],
)

py_test(
    name = "spin_system_test",
    timeout = "eternal",
    srcs = ["spin_system_test.py"],
    python_version = "PY3",
    deps = [
        ":spin_system",
    ],
)
