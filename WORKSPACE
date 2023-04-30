# This file includes external dependencies that are required to compile the
# TensorFlow op.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

EIGEN_COMMIT = "3bb6a48d8c171cf20b5f8e48bfb4e424fbd4f79e"
EIGEN_SHA256 = "eca9847b3fe6249e0234a342b78f73feec07d29f534e914ba5f920f3e09383a3"


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
    sha256 = "e4d716b945d44c6901ccc4ee4c2344e2af127b28713a0faebf0687745e0bf5e7",
    strip_prefix = "qsim-0.16.0",
    urls = ["https://github.com/quantumlib/qsim/archive/refs/tags/v0.16.0.zip"],
)

http_archive(
    name = "org_tensorflow",
    sha256 = "e52cda3bae45f0ae0fccd4055e9fa29892b414f70e2df94df9a3a10319c75fff",
    strip_prefix = "tensorflow-2.11.0",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.11.0.zip",
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

load("//third_party/cuquantum:cuquantum_configure.bzl", "cuquantum_configure")

cuquantum_configure(name = "local_config_cuquantum")
