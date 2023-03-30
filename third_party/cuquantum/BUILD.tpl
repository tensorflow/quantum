package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuquantum_headers",
    linkstatic = 1,
    srcs = [":cuquantum_header_include"],
    includes = ["include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libcuquantum",
    srcs = [
        ":libcustatevec.so",
    ],
    visibility = ["//visibility:public"],
)

%{CUQUANTUM_HEADER_GENRULE}
%{CUSTATEVEC_SHARED_LIBRARY_GENRULE}
