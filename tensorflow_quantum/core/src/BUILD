package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

cc_library(
    name = "circuit",
    hdrs = ["circuit.h"],
    deps = [
        ":gates_def",
    ],
)

cc_test(
    name = "circuit_test",
    size = "small",
    srcs = ["circuit_test.cc"],
    linkstatic = 1,
    deps = [
        ":circuit",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "circuit_parser",
    srcs = ["circuit_parser.cc"],
    hdrs = ["circuit_parser.h"],
    deps = [
        ":circuit",
        ":gates_def",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "circuit_parser_test",
    size = "small",
    srcs = ["circuit_parser_test.cc"],
    linkstatic = 1,
    deps = [
        ":circuit_parser",
        ":gates_def",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "gates_def",
    srcs = ["gates_def.cc"],
    hdrs = ["gates_def.h"],
    deps = [
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "matrix",
    hdrs = ["matrix.h"],
    deps = [],
)

cc_test(
    name = "matrix_test",
    size = "small",
    srcs = ["matrix_test.cc"],
    deps = [
        ":matrix",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "program_resolution",
    srcs = ["program_resolution.cc"],
    hdrs = ["program_resolution.h"],
    deps = [
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "program_resolution_test",
    size = "small",
    srcs = ["program_resolution_test.cc"],
    linkstatic = 1,
    deps = [
        ":program_resolution",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:tf_header_lib",
    ],
)
