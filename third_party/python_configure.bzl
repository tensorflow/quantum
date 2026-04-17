# Copyright 2026 The TensorFlow Quantum Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repository rule for Python detection and toolchain registration."""

def _python_configure_impl(repository_ctx):
    python_bin = repository_ctx.os.environ.get("PYTHON_BIN_PATH") or \
                 repository_ctx.which("python3") or \
                 repository_ctx.which("python")

    if not python_bin:
        fail("Python interpreter not found. Please provide it via --repo_env=PYTHON_BIN_PATH=/path/to/python or set the PYTHON_BIN_PATH environment variable.")

    substitutions = {"%{PYTHON_BIN_PATH}%": str(python_bin).replace("\\", "\\\\")}

    repository_ctx.template(
        "BUILD",
        Label("//third_party/python:BUILD.tpl"),
        substitutions,
    )
    repository_ctx.template(
        "defs.bzl",
        Label("//third_party/python:defs.bzl.tpl"),
        substitutions,
    )

_python_configure = repository_rule(
    implementation = _python_configure_impl,
    environ = [
        "PYTHON_BIN_PATH",
        "PATH",
    ],
)

def python_configure():
    """Configures the Python toolchain for TFQ, TF, and XLA.

    Three identical repositories are created to satisfy the naming expectations
    of various external dependencies:
    - 'local_config_python': Used by TensorFlow Quantum and its internal rules.
    - 'local_execution_config_python': Required by TensorFlow (org_tensorflow)
      and TSL for certain toolchain configurations.
    - 'python': Provided as a generic handle.

    Although redundant, this ensures compatibility across the diverse dependency
    tree without requiring extensive repo_mapping.
    """
    _python_configure(name = "local_config_python")
    _python_configure(name = "local_execution_config_python")
    _python_configure(name = "python")
