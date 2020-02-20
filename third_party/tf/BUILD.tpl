package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = [":tf_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = [":libtensorflow_framework.so"],
    #data = ["lib/libtensorflow_framework.so"],
    visibility = ["//visibility:public"],
)

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

py_library(
	name = "test_log_pb2",
	srcs = ["test_log_pb2.py"],
)

%{TF_HEADER_GENRULE}
%{TF_SHARED_LIBRARY_GENRULE}
%{TF_PROTO_GENRULE}