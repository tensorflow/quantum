load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda_is_configured")
package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

py_test(
    name = "benchmark_clifford_circuit",
    srcs = ["benchmark_clifford_circuit.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_py",
        "//tensorflow_quantum/core/serialize:serializer",
        "@local_config_tf//:test_log_pb2",
    ],
)

py_test(
    name = "benchmark_random_circuit",
    srcs = ["benchmark_random_circuit.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_py",
        "//tensorflow_quantum/core/serialize:serializer",
        "@local_config_tf//:test_log_pb2",
    ],
)

py_test(
    name = "benchmark_cuquantum_ops",
    srcs = ["benchmark_cuquantum_ops.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_cuquantum_py",
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_py",
        "//tensorflow_quantum/core/serialize:serializer",
        "@local_config_tf//:test_log_pb2",
        "//tensorflow_quantum/python:util",
    ],
)
py_test(
    name = "benchmark_op_gradients",
    srcs = ["benchmark_op_gradients.py"],
    python_version = "PY3",
    deps = [
        "//tensorflow_quantum/core/ops:batch_util",
        "//tensorflow_quantum/core/ops:cirq_ops",
        "//tensorflow_quantum/core/ops:tfq_simulate_ops_py",
        "//tensorflow_quantum/core/ops:tfq_utility_ops_py",
        "//tensorflow_quantum/python:util",
        "//tensorflow_quantum/python/differentiators:linear_combination",
        "//tensorflow_quantum/python/differentiators:parameter_shift",
        "@local_config_tf//:test_log_pb2",
    ],
)

py_library(
    name = "benchmark_util",
    srcs = ["benchmark_util.py"],
    deps = [
        "@local_config_tf//:test_log_pb2",
    ],
)

py_test(
    name = "benchmark_util_test",
    srcs = ["benchmark_util_test.py"],
    python_version = "PY3",
    deps = [
        ":benchmark_util",
        "@local_config_tf//:test_log_pb2",
    ],
)
