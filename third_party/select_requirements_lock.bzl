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

"""Repository rule that selects the requirements lock matching the active Python."""

def _select_requirements_lock_impl(repository_ctx):
    result = repository_ctx.execute([
        repository_ctx.attr.python_bin_path,
        "-c",
        "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
    ])
    if result.return_code:
        fail("Failed to inspect Python interpreter %r: %s" % (
            repository_ctx.attr.python_bin_path,
            result.stderr,
        ))

    version = result.stdout.strip()
    if version == "3.10":
        lock_label = repository_ctx.attr.requirements_lock_3_10
    elif version == "3.11":
        lock_label = repository_ctx.attr.requirements_lock_3_11
    elif version == "3.12":
        lock_label = repository_ctx.attr.requirements_lock_3_12
    else:
        fail("Unsupported Python version %r. Expected one of 3.10, 3.11, 3.12." % version)

    repository_ctx.file("requirements.txt", repository_ctx.read(lock_label))
    repository_ctx.file("BUILD.bazel", 'exports_files(["requirements.txt"])\n')

select_requirements_lock = repository_rule(
    implementation = _select_requirements_lock_impl,
    attrs = {
        "python_bin_path": attr.string(mandatory = True),
        "requirements_lock_3_10": attr.label(mandatory = True, allow_single_file = True),
        "requirements_lock_3_11": attr.label(mandatory = True, allow_single_file = True),
        "requirements_lock_3_12": attr.label(mandatory = True, allow_single_file = True),
    },
)
