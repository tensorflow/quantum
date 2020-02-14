package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

cc_library(
    name = "qsim",
    deps = [
        ":fuser_basic",
        ":mux",
        ":q_state",
        ":simulator",
        ":simulator2_avx",
        ":simulator2_slow",
        ":simulator2_sse",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/src:circuit",
        "//tensorflow_quantum/core/src:circuit_parser",
        "//tensorflow_quantum/core/src:gates_def",
        "//tensorflow_quantum/core/src:matrix",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "fuser_basic",
    srcs = ["fuser_basic.cc"],
    hdrs = ["fuser_basic.h"],
    deps = [
        "//tensorflow_quantum/core/src:circuit",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "fuser_basic_test",
    srcs = ["fuser_basic_test.cc"],
    deps = [
        ":fuser_basic",
        "//tensorflow_quantum/core/src:circuit",
        "//tensorflow_quantum/core/src:gates_def",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "mux",
    srcs = ["mux.cc"],
    hdrs = ["mux.h"],
    deps = [
        ":simulator",
        ":simulator2_avx",
        ":simulator2_slow",
        ":simulator2_sse",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "mux_test",
    srcs = ["mux_test.cc"],
    deps = [
        ":mux",
        ":simulator",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "q_state",
    srcs = ["q_state.cc"],
    hdrs = ["q_state.h"],
    deps = [
        ":fuser_basic",
        ":mux",
        ":simulator",
        ":simulator2_avx",
        ":simulator2_slow",
        ":simulator2_sse",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/src:circuit",
        "//tensorflow_quantum/core/src:circuit_parser",
        "//tensorflow_quantum/core/src:matrix",
        "@com_google_absl//absl/strings",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "q_state_test",
    srcs = ["q_state_test.cc"],
    linkstatic = 1,
    deps = [
        ":q_state",
        ":simulator",
        ":simulator2_avx",
        ":simulator2_slow",
        ":simulator2_sse",
        "//tensorflow_quantum/core/proto:program_cc_proto",
        "//tensorflow_quantum/core/src:circuit",
        "//tensorflow_quantum/core/src:circuit_parser",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest_main",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "simulator",
    srcs = ["simulator.cc"],
    hdrs = ["simulator.h"],
    deps = [
        ":util",
    ],
)

cc_library(
    name = "simulator2_avx",
    srcs = ["simulator2_avx.cc"],
    hdrs = ["simulator2_avx.h"],
    deps = [
        ":simulator",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "simulator2_sse",
    srcs = ["simulator2_sse.cc"],
    hdrs = ["simulator2_sse.h"],
    deps = [
        ":simulator",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "simulator2_slow",
    srcs = ["simulator2_slow.cc"],
    hdrs = ["simulator2_slow.h"],
    deps = [
        ":simulator",
    ],
)

cc_library(
    name = "state_space",
    srcs = ["state_space.cc"],
    hdrs = ["state_space.h"],
    deps = [
        ":fuser_basic",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/src:circuit",
        "//tensorflow_quantum/core/src:circuit_parser",
        "//tensorflow_quantum/core/src:matrix",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "state_space_slow",
    srcs = ["state_space_slow.cc"],
    hdrs = ["state_space_slow.h"],
    deps = [
        ":state_space",
        "//tensorflow_quantum/core/src:matrix",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "state_space_avx",
    srcs = ["state_space_avx.cc"],
    hdrs = ["state_space_avx.h"],
    deps = [
        ":state_space",
        ":util",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [],
)
