# This file includes external dependencies that are required to compile the
# TensorFlow op.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "qsim",
    sha256 = "d39b9c48866ce4d6a095093ae8059444d649e851219497af99e937a74f1e9a45",
    strip_prefix = "qsim-0.9.2-dev-20210317",
    urls = ["https://github.com/quantumlib/qsim/archive/v0.9.2-dev+20210317.zip"],
)

http_archive(
    name = "org_tensorflow",
    sha256 = "2023a377a16e5566b8981400af9e8c8e25d3367d82824ffec2b5b6b9c7dba55d",
    strip_prefix = "tensorflow-2.5.1",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.5.1.zip",
    ],
)

load("@org_tensorflow//tensorflow:workspace3.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace2.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace1.bzl", "workspace")

workspace()

load("@org_tensorflow//tensorflow:workspace0.bzl", "workspace")

workspace()

load("//third_party/tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

http_archive(
    name = "six_archive",
    build_file = "@com_google_protobuf//:six.BUILD",
    sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
    url = "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55",
)

bind(
    name = "six",
    actual = "@six_archive//:six",
)
