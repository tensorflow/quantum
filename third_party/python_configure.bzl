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
    python_bin = repository_ctx.os.environ.get("python-path") or \
                 repository_ctx.os.environ.get("PYTHON_BIN_PATH") or \
                 repository_ctx.which("python3") or \
                 repository_ctx.which("python")

    if not python_bin:
        fail("Python interpreter not found. Please provide it via --repo_env=python-path=/path/to/python or set PYTHON_BIN_PATH.")

    substitutions = {"%{PYTHON_BIN_PATH}%": str(python_bin)}

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

python_configure = repository_rule(
    implementation = _python_configure_impl,
    environ = [
        "python-path",
        "PYTHON_BIN_PATH",
    ],
)
