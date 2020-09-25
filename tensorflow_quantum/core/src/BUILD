package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

cc_library(
    name = "adj_util",
    srcs = ["adj_util.cc"],
    hdrs = ["adj_util.h"],
    deps = [
        "@qsim//lib:circuit",
        "@qsim//lib:gates_cirq",
        "@qsim//lib:gate",
        "@qsim//lib:fuser",
        "@qsim//lib:fuser_basic",
        "@qsim//lib:io",
        "@qsim//lib:matrix",
        ":circuit_parser_qsim",
    ],
)

cc_test(
    name = "adj_util_test",
    srcs = ["adj_util_test.cc"],
    deps = [
        ":adj_util",
        "@qsim//lib:gates_cirq",
        "@qsim//lib:matrix",
        "@com_google_googletest//:gtest_main",
        ":circuit_parser_qsim",
    ]
)

cc_library(
    name = "circuit_parser_qsim",
    srcs = ["circuit_parser_qsim.cc"],
    hdrs = ["circuit_parser_qsim.h"],
    deps = [
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@qsim//lib:circuit",
        "@qsim//lib:gates_cirq",
        "@qsim//lib:fuser",
        "@qsim//lib:fuser_basic",
        "@qsim//lib:io"  
    ],
)

cc_library(
    name = "util_qsim",
    srcs = [],
    hdrs = ["util_qsim.h"],
    deps = [
        ":circuit_parser_qsim",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@com_google_absl//absl/container:inlined_vector",  # unclear why needed.
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "@qsim//lib:qsim_lib",
    ]
)

cc_test(
    name = "util_qsim_test",
    size = "small",
    srcs = ["util_qsim_test.cc"],
    linkstatic = 0,
    deps = [
        ":util_qsim",
        "@com_google_absl//absl/container:flat_hash_map",
        "@qsim//lib:qsim_lib",
        "@com_google_googletest//:gtest_main",
    ]
)

cc_test(
    name = "circuit_parser_qsim_test",
    size = "small",
    srcs = ["circuit_parser_qsim_test.cc"],
    linkstatic = 0,
    deps = [
        ":circuit_parser_qsim",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
        "@qsim//lib:circuit",
        "@qsim//lib:gates_cirq",
        "@qsim//lib:fuser",

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
