#!/usr/bin/env bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Summary: run auditwheel on a given wheel & summarize the platform tags.

set -eu

IMAGE_TAG="quay.io/pypa/manylinux_2_34_x86_64"
AUDITWHEEL="third_party/tf/auditwheel"

# Default values for variables that can be changed via command line flags.
py_version="3.11"
verbose=""

usage="Usage: ${0} [OPTIONS] <command> <file>
Runs auditwheel <command> on the wheel in <file>.
<command> must be one of 'show' or 'repair'.
Available options:
  -h  Show this help message and exit
  -p  Python version (default: ${py_version})
  -v  Verbose output"

while getopts "hp:v" opt; do
    case "${opt}" in
        p) py_version="${OPTARG}" ;;
        v) verbose="--verbose" ;;
        h) echo "${usage}"; exit 0 ;;
        *) echo "${usage}" >&2; exit 1 ;;
    esac
done
shift $((OPTIND -1))
if [ $# -ne 2 ]; then
    echo "ERROR: insufficient arguments."
    echo "${usage}" >&2
    exit 1
fi

action="$1"
if [ "${action}" != "show" ] && [ "${action}" != "repair" ]; then
    echo "ERROR: Command must be 'show' or 'repair'." >&2
    exit 1
fi

wheel_path="$(realpath "${2}")"
wheel_name="$(basename "${2}")"

# Change to the top of the local TFQ git tree.
thisdir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel) || exit $?
cd "${repo_dir}" || exit $?

mkdir -p "${repo_dir}/wheelhouse"

if [ "${action}" == "show" ]; then
    cmd="${verbose} show"
else
    cmd="${verbose} repair --wheel-dir ${repo_dir}/wheelhouse"
fi

docker run -i --rm --network host \
    -w /tfq \
    -v "${PWD}":/tfq \
    -v "${wheel_path}":"/tmp/${wheel_name}" \
    -e py_version="${py_version}" \
    -e IMAGE_TAG="${IMAGE_TAG}" \
    "${IMAGE_TAG}" \
    bash -c "/tfq/${AUDITWHEEL} '${cmd}' '/tmp/${wheel_name}'"

echo "Done."
