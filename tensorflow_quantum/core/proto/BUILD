load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# Export for the PIP package.
exports_files(["__init__.py"])

proto_library(
    name = "program_proto",
    srcs = ["program.proto"]
)

py_proto_library(
    name = "program_py_proto",
    srcs = ["program.proto"],
)

cc_proto_library(
    name = "program_cc_proto",
    deps = [":program_proto"],
)

py_proto_library(
    name = "pauli_sum_py_proto",
    srcs = ["pauli_sum.proto"],
)

proto_library(
    name = "pauli_sum_proto",
    srcs = ["pauli_sum.proto"],
)

cc_proto_library(
    name = "pauli_sum_cc_proto",
    deps = [
        ":pauli_sum_proto",
    ],
)

py_proto_library(
    name = "projector_sum_py_proto",
    srcs = ["projector_sum.proto"],
)

proto_library(
    name = "projector_sum_proto",
    srcs = ["projector_sum.proto"],
)

cc_proto_library(
    name = "projector_sum_cc_proto",
    deps = [
        ":projector_sum_proto",
    ],
)
