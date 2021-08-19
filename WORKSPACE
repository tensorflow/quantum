# This file includes external dependencies that are required to compile the
# TensorFlow op.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

EIGEN_COMMIT = "12e8d57108c50d8a63605c6eb0144c838c128337"
EIGEN_SHA256 = "f689246e342c3955af48d26ce74ac34d21b579a00675c341721a735937919b02"


http_archive(
    name = "eigen",
    build_file_content = """
cc_library(
  name = "eigen3",
  textual_hdrs = glob(["Eigen/**", "unsupported/**"]),
  visibility = ["//visibility:public"],
)
    """,
    sha256 = EIGEN_SHA256,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
)

http_archive(
    name = "qsim",
    sha256 = "9bc32249fe83120e0d427247c4e1d4ac330bc7922fd55396b17334e1b1254f9e",
    strip_prefix = "qsim-0.10.0",
    urls = ["https://github.com/quantumlib/qsim/archive/refs/tags/v0.10.0.zip"],
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
