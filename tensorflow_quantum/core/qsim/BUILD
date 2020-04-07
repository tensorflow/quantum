package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

cc_library(
    name = "qsim",
    deps = [
        ":fuser_basic",
        ":mux",
        ":state_space",
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
	"//tensorflow_quantum/core/src:gates_def",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "mux",
    srcs = ["mux.cc"],
    hdrs = ["mux.h"],
    deps = [
        ":state_space",
        ":state_space_avx",
        ":state_space_slow",
        ":state_space_sse",
        "@com_google_absl//absl/memory",
    ],
)

cc_test(
    name = "mux_test",
    srcs = ["mux_test.cc"],
    deps = [
        ":mux",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "state_space",
    srcs = ["state_space.cc"],
    hdrs = ["state_space.h"],
    deps = [
        ":fuser_basic",
        ":matrix",
        ":util",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/src:circuit",
        "//tensorflow_quantum/core/src:circuit_parser",
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_test(
    name = "state_space_test",
    srcs = ["state_space_test.cc"],
    deps = [
        ":mux",
        ":state_space",
        "//tensorflow_quantum/core/proto:pauli_sum_cc_proto",
        "//tensorflow_quantum/core/src:circuit",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "state_space_slow",
    srcs = ["state_space_slow.cc"],
    hdrs = ["state_space_slow.h"],
    deps = [
        ":matrix",
        ":state_space",
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
    name = "state_space_sse",
    srcs = ["state_space_sse.cc"],
    hdrs = ["state_space_sse.h"],
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

cc_test(
    name = "util_test",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "@com_google_googletest//:gtest_main",
    ],
)