#!/bin/bash
# Copyright 2025 The TensorFlow Quantum Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# Summary: bundle external shared libraries into the final TFQ wheel.
# Run this script with the option "-h" to get a usage summary.
#
# This uses auditwheel to "repair" the wheel: copy external shared libraries
# into the wheel itself and modify the RPATH entries such that these libraries
# will be picked up at runtime. This accomplishes a similar result as if the
# libraries had been statically linked.

set -eu -o pipefail

function quit() {
  printf 'Error: %b\n' "$*" >&2
  exit 1
}

# Find the top of the local TFQ git tree. Do it early in case this fails.
thisdir=$(dirname "${BASH_SOURCE[0]:?}")
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel 2> /dev/null) || \
  quit "This script must be run from inside the TFQ git tree."

# Default values for variables that can be changed via command line flags.
docker_image="quay.io/pypa/manylinux_2_34_x86_64"
platform="manylinux_2_17_x86_64"
py_version=$(python3 --version | cut -d' ' -f2 | cut -d. -f1,2)
action="repair"

usage="Usage: ${0} [OPTIONS] /path/to/wheel.whl
Run auditwheel on the given wheel file. Available options:

Configuration options:
  -m IMG  Use manylinux Docker image IMG (default: ${docker_image})
  -p X.Y  Use Python version X.Y (default: ${py_version})
  -t TAG  Pass --plat TAG to auditwheel (default: ${platform})

General options:
  -h  Show this help message and exit
  -s  Run 'auditwheel show', not repair (default: run 'auditwheel repair')"

while getopts "hm:p:st:" opt; do
  case "${opt}" in
    h) echo "${usage}"; exit 0 ;;
    m) docker_image="${OPTARG}" ;;
    p) py_version="${OPTARG}" ;;
    s) action="show" ;;
    t) platform="${OPTARG}" ;;
    *) quit "${usage}" ;;
  esac
done
shift $((OPTIND -1))
if (( $# < 1 )); then
  quit "Must provide at least one argument.\n\n${usage}"
fi

wheel_path="$(cd "$(dirname "${1}")" && pwd)/$(basename "${1}")"
wheel_name="$(basename "${1}")"

auditwheel_args=()
if [[ "${action}" == "repair" ]]; then
    auditwheel_args+=(
        "--exclude" "libtensorflow_framework.so.2"
        "--plat" "${platform}"
        "-w" "/tfq/wheelhouse"
    )
fi

echo "Running 'auditwheel ${action}' in Docker with image ${docker_image}"
docker run -it --rm --network host \
  -w /tfq \
  -v "${repo_dir}":/tfq \
  -v "${wheel_path}":"/tmp/${wheel_name}" \
  "${docker_image}" \
  auditwheel "${action}" "${auditwheel_args[@]}" "/tmp/${wheel_name}"

echo "Done. New wheel file written to ${repo_dir}/wheelhouse"
