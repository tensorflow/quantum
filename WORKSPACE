# This file includes external dependencies that are required to compile the
# TensorFlow op.



load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")



EIGEN_COMMIT = "aa6964bf3a34fd607837dd8123bc42465185c4f8"


http_archive(
    name = "eigen",
    sha256 = "35ba771e30c735a4215ed784d7e032086cf89fe6622dce4d793c45dd74373362",
    build_file_content = """
cc_library(
  name = "eigen3",
  textual_hdrs = glob(["Eigen/**", "unsupported/**"]),
  visibility = ["//visibility:public"],
)
    """,
        strip_prefix = "eigen-{commit}".format(commit = EIGEN_COMMIT),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
            "https://gitlab.com/libeigen/eigen/-/archive/{commit}/eigen-{commit}.tar.gz".format(commit = EIGEN_COMMIT),
        ],
)

http_archive(
    name = "qsim",
    sha256 = "b9c1eba09a885a938b5e73dfc2e02f5231cf3b01d899415caa24769346a731d5",
    # patches = [
    #     "//third_party/tf:qsim.patch",
    # ],
    strip_prefix = "qsim-0.13.3",
    urls = ["https://github.com/quantumlib/qsim/archive/refs/tags/v0.13.3.zip"],
)

local_repository(
    name = "python",
    path = "third_party/python_legacy",
)

http_archive(
    name = "org_tensorflow",
    patches = ["//third_party/tf:tf.patch"],
    sha256 = "c8c8936e7b6156e669e08b3c388452bb973c1f41538149fce7ed4a4849c7a012",
    strip_prefix = "tensorflow-2.16.2",
    urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.16.2.zip"],
)


load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3"); tf_workspace3()
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2"); tf_workspace2()
load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1"); tf_workspace1()
load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0"); tf_workspace0()


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
