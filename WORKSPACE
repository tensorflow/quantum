# This file includes external dependencies that are required to compile the
# TensorFlow op.

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

local_repository(
    name = "pypi_setuptools",
    path = "third_party/pypi_setuptools",
)

# TensorFlow's .bzl files, loaded later in this file, also load rules_python
# but we need a slightly newer version that is still compatible with TF's.
http_archive(
    name = "rules_python",
    sha256 = "c68bdc4fbec25de5b5493b8819cfc877c4ea299c0dcb15c244c5a00208cde311",
    strip_prefix = "rules_python-0.31.0",
    url = "https://github.com/bazel-contrib/rules_python/releases/download/0.31.0/rules_python-0.31.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

local_repository(
    name = "python",
    path = "third_party/python_legacy",
)

load("@python//:defs.bzl", "interpreter")

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    name = "pypi",
    requirements_lock = "//:requirements.txt",
    python_interpreter = interpreter,
    extra_pip_args = [
        "--index-url",
        "https://pypi.org/simple/",
    ],
)

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

# Eigen commit used by TensorFlow / TFQ.
# This commit corresponds to Eigen version 3.4.90
# (verified via Eigen/src/Core/util/Macros.h).
EIGEN_COMMIT = "c1d637433e3b3f9012b226c2c9125c494b470ae6"

http_archive(
    name = "eigen",
    sha256 = "0992b93a590c39e196a9efdb5b4919fbf3fb485e7e656c6a87b21ddadb7f6ad2",
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


http_archive(
    name = "org_tensorflow",
    # patches = ["//third_party/tf:tf.patch"],
    sha256 = "75d63eab5c4b41a831e39e2e3795e1f9dcc247e8fcca11bf3f3e1a6ac35e55d0",
    strip_prefix = "tensorflow-2.17.1",
    urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.17.1.zip"],
)


load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("//third_party/tf:tf_configure.bzl", "tf_configure")

tf_configure(name = "local_config_tf")

http_archive(
    name = "six_archive",
    build_file = "@com_google_protobuf//:six.BUILD",
    sha256 = "ff70335d468e7eb6ec65b95b99d3a2836546063f63acc5171de367e834932a81",
    url = "https://files.pythonhosted.org/packages/94/e7/b2c673351809dca68a0e064b6af791aa332cf192da575fd474ed7d6f16a2/six-1.17.0.tar.gz",
)

bind(
    name = "six",
    actual = "@six_archive//:six",
)
