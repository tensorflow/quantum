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
declare -a docker_args=()

ubuntu_versions+=( "22.04" "24.04" )
python_versions+=( "3.9" "3.10" "3.11" "3.12" )
total_items=$(( ${#ubuntu_versions[@]} * ${#python_versions[@]}))

echo "~~~~ Building a total of ${total_items} Docker images"

# Flag --no-cache forces containers to be rebuilt.
docker_args+=( "--no-cache" )

# shellcheck disable=SC2068,SC2145
for os_version in "${ubuntu_versions[@]}"; do
    for py_version in "${python_versions[@]}"; do
        echo
        echo "~~~~ Python ${py_version} on Ubuntu ${os_version}"
        docker build \
            --build-arg PYTHON_VERSION="${py_version}" \
            --build-arg UBUNTU_VERSION="${os_version}" \
            ${docker_args[@]}\
            -t ubuntu${os_version%%.*}-cp${py_version//./}:latest .
    done
done

echo
echo "~~~~ Done. Created the following Docker images:"
echo
docker images --filter "reference=ubuntu*-cp*"
