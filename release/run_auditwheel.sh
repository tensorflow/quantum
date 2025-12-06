#!/bin/bash
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

# Default values for variables that can be changed via command line flags.
py_version="3.11"
action="repair"
verbose=""

usage="Usage: ${0} [OPTIONS] /path/to/wheel
Run auditwheel on the given wheel file. Available options:
  -h  Show this help message and exit
  -p  Python version to use (default: ${py_version})
  -s  Run auditwheel show, not repair (default: run auditwheel repair)
  -v  Verbose output"

while getopts "p:hsv" opt; do
    case "${opt}" in
        p) py_version="${OPTARG}" ;;
        s) action="show" ;;
        v) verbose="--verbose" ;;
        h) echo "${usage}"; exit 0 ;;
        *) echo "${usage}" >&2; exit 1 ;;
    esac
done
shift $((OPTIND -1))
if [ ! $# -ge 1 ]; then
    echo "ERROR: insufficient arguments."
    echo "${usage}" >&2
    exit 1
fi

wheel_path="$(realpath "${1}")"
wheel_name="$(basename "${1}")"

# Change to the top of the local TFQ git tree.
thisdir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel) || exit $?
cd "${repo_dir}" || exit $?

args="${action} ${verbose} --plat manylinux_2_34_x86_64"
if [ "${action}" = "repair" ]; then
    args="${args} --exclude libtensorflow_framework.so.2 -w /tfq/wheelhouse"
fi

docker run -it --rm --network host \
    -w /tfq \
    -v "${PWD}":/tfq \
    -v "${wheel_path}":"/tmp/${wheel_name}" \
    "${IMAGE_TAG}" \
    bash -c "auditwheel ${args} /tmp/${wheel_name}"

echo "Done. New wheel file written to ${repo_dir}/wheelhouse"
