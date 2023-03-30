package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuquantum_headers",
    linkstatic = 1,
    srcs = [":cuquantum_header_include"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libcuquantum",
    srcs = [
        ":libcustatevec.so",
        ":libcutensornet.so",
    ],
    visibility = ["//visibility:public"],
)

%{CUQUANTUM_HEADER_GENRULE}
%{CUSTATEVEC_SHARED_LIBRARY_GENRULE}
%{CUTENSORNET_SHARED_LIBRARY_GENRULE}