package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

cc_library(
    name = "circuit",
    srcs = ["circuit.cc"],
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
    ],
)

cc_library(
    name = "util_qsim",
    srcs = [],
    hdrs = ["util_qsim.h"],
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
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
        "//tensorflow_quantum/core/qsim:matrix",
        "@com_google_absl//absl/container:flat_hash_map",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "gates_def_test",
    size = "small",
    srcs = ["gates_def_test.cc"],
    deps = [
        ":gates_def",
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
