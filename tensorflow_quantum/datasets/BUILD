package(default_visibility = ["//visibility:public"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_library(
    name = "cluster_state",
    srcs = ["cluster_state.py"],
)

py_test(
    name = "cluster_state_test",
    srcs = ["cluster_state_test.py"],
    deps = [
        ":cluster_state",
    ],
)
