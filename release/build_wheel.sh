#!/bin/bash
# Copyright 2025 Google LLC
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

# Summary: build TFQ using a TensorFlow build Docker container. Basic steps:
#
# 1. Write to a file a small shell script that does the following:
#
#    a) pip install TFQ's requirements.txt file
#    b) run TFQ's configure.sh script
#    c) run bazel to build build_pip_package
#    d) run the resulting build_pip_package
#    e) copy the wheel created by build_pip_package to ./wheels
#
# 2. Start Docker with image tensorflow/build:${tf_version}-python${py_version}
#    and run the script written in step 1.
#
# 3. Do some basic tests on the wheel using standard Python utilities.
#
# 4. Exit.

set -eu

# Default values for variables that can be changed via command line flags.
tf_version="2.16"
py_version="3.11"
cuda_version="12"
cleanup="true"

usage="Usage: ${0} [OPTIONS]
Configuration options:
  -c N   CUDA version (default: ${cuda_version})
  -p N   Python version (default: ${py_version})
  -t N   TensorFlow version (default: ${tf_version})

General options:
  -e     Don't run bazel clean at the end (default: do)
  -h     Show this help message and exit"

while getopts "c:p:t:he" opt; do
    case "${opt}" in
        c) cuda_version="${OPTARG}" ;;
        p) py_version="${OPTARG}" ;;
        t) tf_version="${OPTARG}" ;;
        e) cleanup="false" ;;
        h) echo "${usage}"; exit 0 ;;
        *) echo "${usage}" >&2; exit 1 ;;
    esac
done
shift $((OPTIND -1))

# See https://hub.docker.com/r/tensorflow/build/tags for available containers.
image_tag="tensorflow/build:${tf_version}-python${py_version}"

# This should match what TensorFlow's .bazelrc file uses.
crosstool="@sigbuild-r${tf_version}-clang_config_cuda//crosstool:toolchain"

# Note: configure.sh is run inside the container, and it creates a .bazelrc
# file that adds other cxxopt flags. They don't need to be repeated here.
BUILD_OPTIONS="--cxxopt=-O3 --cxxopt=-msse2 --cxxopt=-msse3 --cxxopt=-msse4"

# Create a script to be run by the shell inside the Docker container.
build_script=$(mktemp /tmp/tfq_build.XXXXXX)
trap 'rm -f "${build_script}" || true' EXIT

# The printf'ed section dividers are to make it easier to search the output.
cat <<'EOF' > "${build_script}"
#!/usr/bin/env bash
set -o errexit
cd /tfq
PREFIX='[DOCKER] '
exec > >(sed "s/^/${PREFIX} /")
exec 2> >(sed "s/^/${PREFIX} /" >&2)
printf "Build configuration inside Docker container:\n"
printf "  Docker imnage:    ${image_tag}\n"
printf "  TF version:       ${tf_version}\n"
printf "  Python version:   ${py_version}\n"
printf "  CUDA version:     ${cuda_version}\n"
printf "  Docker image:     ${image_tag}\n"
printf "  Crosstool:        ${crosstool}\n"
printf "  vCPUs available:  $(nproc)\n"
printf "\n:::::::: Configuring Python environment ::::::::\n\n"
python3 -m pip install --upgrade pip
pip install -r requirements.txt
printf "\n:::::::: Configuring TensorFlow Quantum build ::::::::\n\n"
printf "Y\n" | ./configure.sh
printf "\n:::::::: Starting Bazel build ::::::::\n\n"
bazel build ${build_flags} release:build_pip_package
printf "\n:::::::: Creating Python wheel ::::::::\n\n"
bazel-bin/release/build_pip_package /build_output/
if [ "${cleanup}" == "true" ]; then
  printf "\n:::::::: Cleaning up ::::::::\n\n"
  bazel clean --async
fi
EOF

chmod +x "${build_script}"

# Change to the top of the local TFQ git tree.
thisdir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel) || exit $?
cd "${repo_dir}" || exit $?

echo "Running Docker from working directory ${repo_dir}."
docker run -it --rm --network host \
    -w /tfq \
    -v "${PWD}":/tfq \
    -v /tmp/tensorflow_quantum:/build_output \
    -v "${build_script}:/tmp/build_script.sh" \
    -e build_flags="--crosstool_top=${crosstool} ${BUILD_OPTIONS}" \
    -e cuda_version="${cuda_version}" \
    -e py_version="${py_version}" \
    -e tf_version="${tf_version}" \
    -e image_tag="${image_tag}" \
    -e crosstool="${crosstool}" \
    -e cleanup="${cleanup}" \
    -e HOST_PERMS="$(id -u):$(id -g)" \
    "${image_tag}" \
    /tmp/build_script.sh

# Run basic checks on the wheel file.
echo
echo "Doing basic checks on the wheel file."
pip install -qq check-wheel-contents
check-wheel-contents /tmp/tensorflow_quantum/*.whl

echo
echo "Done. Look for wheel in /tmp/tensorflow_quantum/."
ls -l /tmp/tensorflow_quantum/
