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

# Summary: create a set of Docker images for testing TFQ distributions.
# This loops over a set of Ubuntu versions and Python versions, and builds
# Docker images using the Dockerfile in this directory.
#
# Example of how the resulting images can be run
#   docker run -it --rm --network host -v .:/tfq ubuntu22-cp39

set -e

declare -a ubuntu_versions=()
declare -a python_versions=()

ubuntu_versions+=( "22.04" "24.04" )
python_versions+=( "3.9" "3.10" "3.11" "3.12" "3.13" )

usage="Usage: ${0} [OPTIONS]

Build a set of basic Ubuntu Linux x86_64 Docker images with
Python preinstalled.

General options:
  -h     Show this help message and exit
  -v     Run Docker build with verbose progress output"

while getopts "hv" opt; do
  case "${opt}" in
        h) echo "${usage}"; exit 0 ;;
        v) export BUILDKIT_PROGRESS=plain ;;
        ?) echo "${usage}"; exit 1 ;;
    esac
done

total_items=$(( ${#ubuntu_versions[@]} * ${#python_versions[@]}))
echo "Building a total of ${total_items} Docker images."

start_time="$(date +"%Y-%m-%d-%H%M")"
for os_version in "${ubuntu_versions[@]}"; do
    for py_version in "${python_versions[@]}"; do
        echo
        echo "~~~~ Python ${py_version} on Ubuntu ${os_version}"
        # shellcheck disable=SC2086  # Lack of quotes around vars is ok here.
        docker build --no-cache --label "build-datetime=${start_time}" \
            --build-arg PYTHON_VERSION="${py_version}" \
            --build-arg UBUNTU_VERSION="${os_version}" \
            -t ubuntu${os_version%%.*}-cp${py_version//./}:latest .
    done
done

echo
echo "~~~~ Done. The following Docker images were created:"
echo
docker images --filter "label=build-datetime=${start_time}"
