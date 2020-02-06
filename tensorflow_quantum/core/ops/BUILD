package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

cc_binary(
    name = "_tfq_ps_utils.so",
    srcs = [
        "tfq_ps_decompose_op.cc",
        "tfq_ps_symbol_replace_op.cc",
        "tfq_ps_weights_from_symbols_op.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-O3",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        ":parse_context",
        ":tfq_simulate_utils",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "_tfq_simulate_ops.so",
    srcs = [
        "tfq_simulate_expectation_op.cc",
        "tfq_simulate_state_op.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-O3",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        ":parse_context",
        ":tfq_simulate_utils",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/qsim",
        "//tensorflow_quantum/core/src:program_resolution",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_binary(
    name = "_tfq_utility_ops.so",
    srcs = [
        "tfq_circuit_append_op.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-O3",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ],
    linkshared = 1,
    deps = [
        ":parse_context",
        ":tfq_simulate_utils",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "parse_context",
    srcs = ["parse_context.cc"],
    hdrs = ["parse_context.h"],
    deps = [
        ":tfq_simulate_utils",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/src:program_resolution",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "tfq_simulate_utils",
    srcs = ["tfq_simulate_utils.cc"],
    hdrs = ["tfq_simulate_utils.h"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

py_library(
    name = "tfq_simulate_ops_py",
    srcs = ["tfq_simulate_ops.py"],
    data = [":_tfq_simulate_ops.so"],
    deps = [
        ":load_module",
    ],
)

py_test(
    name = "tfq_simulate_ops_test",
    srcs = ["tfq_simulate_ops_test.py"],
    deps = [
        ":tfq_simulate_ops_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "circuit_execution_ops",
    srcs = ["circuit_execution_ops.py"],
    deps = [
        "tfq_utility_ops_py",
        ":cirq_ops",
        ":tfq_simulate_ops_py",
    ],
)

py_test(
    name = "circuit_execution_ops_test",
    timeout = "eternal",
    srcs = ["circuit_execution_ops_test.py"],
    deps = [
        ":circuit_execution_ops",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "cirq_ops",
    srcs = ["cirq_ops.py"],
    deps = [
        ":batch_util",
        "//tensorflow_quantum/core/serialize:serializer",
    ],
)

py_test(
    name = "cirq_ops_test",
    srcs = ["cirq_ops_test.py"],
    deps = [
        ":batch_util",
        ":cirq_ops",
        "//tensorflow_quantum/core/serialize:serializer",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "batch_util",
    srcs = ["batch_util.py"],
    deps = [
        "//tensorflow_quantum/core/serialize:serializer",
    ],
)

py_test(
    name = "batch_util_test",
    timeout = "eternal",
    srcs = ["batch_util_test.py"],
    deps = [
        ":batch_util",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "tfq_ps_util_ops_py",
    srcs = ["tfq_ps_util_ops.py"],
    data = [":_tfq_ps_utils.so"],
    deps = [
        ":load_module",
    ],
)

py_test(
    name = "tfq_ps_util_ops_test",
    srcs = ["tfq_ps_util_ops_test.py"],
    deps = [
        ":tfq_ps_util_ops_py",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "tfq_utility_ops_py",
    srcs = ["tfq_utility_ops.py"],
    data = [":_tfq_utility_ops.so"],
    deps = [
        ":load_module",
    ],
)

py_test(
    name = "tfq_utility_ops_test",
    srcs = ["tfq_utility_ops_test.py"],
    deps = [
        ":tfq_utility_ops_py",
        "//tensorflow_quantum/core/serialize:serializer",
        "//tensorflow_quantum/python:util",
    ],
)

py_library(
    name = "load_module",
    srcs = ["load_module.py"],
    deps = [],
)
