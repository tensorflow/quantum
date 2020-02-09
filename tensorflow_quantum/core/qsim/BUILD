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
        ":statespace",
        ":statespace_avx",
        ":statespace_slow",
        ":statespace_sse",
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
        ":statespace",
        ":statespace_avx",
        ":statespace_slow",
        ":statespace_sse",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "mux_test",
    srcs = ["mux_test.cc"],
    deps = [
        ":mux",
        ":simulator",
        ":statespace",
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
        ":statespace",
        ":statespace_avx",
        ":statespace_slow",
        ":statespace_sse",
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
        ":statespace",
        ":statespace_avx",
        ":statespace_slow",
        ":statespace_sse",
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
    hdrs = ["simulator.h"],
    deps = [
        ":statespace",
    ],
)

cc_library(
    name = "simulator2_avx",
    srcs = ["simulator2_avx.cc"],
    hdrs = ["simulator2_avx.h"],
    deps = [
        ":simulator",
        ":statespace_avx",
    ],
)

cc_library(
    name = "simulator2_sse",
    srcs = ["simulator2_sse.cc"],
    hdrs = ["simulator2_sse.h"],
    deps = [
        ":simulator",
        ":statespace_sse",
    ],
)

cc_library(
    name = "simulator2_slow",
    srcs = ["simulator2_slow.cc"],
    hdrs = ["simulator2_slow.h"],
    deps = [
        ":simulator",
        ":statespace_slow",
    ],
)

cc_library(
    name = "statespace",
    srcs = ["statespace.cc"],
    hdrs = ["statespace.h"],
    deps = [":util"],
)

cc_library(
    name = "statespace_avx",
    srcs = ["statespace_avx.cc"],
    hdrs = ["statespace_avx.h"],
    deps = [
        ":statespace",
        ":util",
    ],
)

cc_library(
    name = "statespace_sse",
    srcs = ["statespace_sse.cc"],
    hdrs = ["statespace_sse.h"],
    deps = [
        ":statespace",
    ],
)

cc_library(
    name = "statespace_slow",
    srcs = ["statespace_slow.cc"],
    hdrs = ["statespace_slow.h"],
    deps = [
        ":statespace",
    ],
)

cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [],
)
