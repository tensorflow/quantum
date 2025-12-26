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

# Summary: build a wheel for TFQ using a TensorFlow SIG Build container.
# Run this script with the option "-h" to get a usage summary.
#
# To ensure binary compatibility with TensorFlow, TFQ distributions are built
# using TensorFlow's SIG Build containers and Crosstool C++ toolchain. This
# script encapsulates the process. The basic steps this script performs are:
#
# 1. Write to a file a small shell script that does the following:
#
#    a) pip install TFQ's requirements.txt file
#    b) run TFQ's configure.sh script
#    c) run Bazel to build build_pip_package
#    d) run the resulting build_pip_package
#    e) copy the wheel created by build_pip_package to ./wheels
#
# 2. Start Docker with image tensorflow/build:${tf_version}-python${py_version}
#    and run the script written in step 1.
#
# 3. Do some basic tests on the wheel using standard Python utilities.
#
# 4. Exit.

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
tf_version="2.16"
py_version=$(python3 --version | cut -d' ' -f2 | cut -d. -f1,2)
cuda_version="12"
cleanup="true"

usage="Usage: ${0} [OPTIONS]
Build a distribution wheel for TensorFlow Quantum.

Configuration options:
  -c X.Y  Use CUDA version X.Y (default: ${cuda_version})
  -p X.Y  Use Python version X.Y (default: ${py_version})
  -t X.Y  Use TensorFlow version X.Y (default: ${tf_version})

General options:
  -e     Don't run bazel clean at the end (default: do)
  -n     Dry run: print commands but don't execute them
  -h     Show this help message and exit"

dry_run="false"
while getopts "c:ehnp:t:" opt; do
  case "${opt}" in
    c) cuda_version="${OPTARG}" ;;
    e) cleanup="false" ;;
    h) echo "${usage}"; exit 0 ;;
    n) dry_run="true" ;;
    p) py_version=$(echo "${OPTARG}" | cut -d. -f1,2) ;;
    t) tf_version="${OPTARG}" ;;
    *) quit "${usage}" ;;
  esac
done
shift $((OPTIND -1))

# See https://hub.docker.com/r/tensorflow/build/tags for available containers.
docker_image="tensorflow/build:${tf_version}-python${py_version}"

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
#!/bin/bash
set -o errexit
cd /tfq
PREFIX='[DOCKER] '
exec > >(sed "s/^/${PREFIX} /")
exec 2> >(sed "s/^/${PREFIX} /" >&2)
printf ":::::::: Build configuration inside Docker container ::::::::\n"
printf "  Docker image:     ${docker_image}\n"
printf "  Crosstool:        ${crosstool}\n"
printf "  TF version:       ${tf_version}\n"
printf "  Python version:   ${py_version}\n"
printf "  CUDA version:     ${cuda_version}\n"
printf "  vCPUs available:  $(nproc)\n"
printf "\n\n:::::::: Configuring Python environment ::::::::\n\n"
python3 -m pip install --upgrade pip --root-user-action ignore
pip install -r requirements.txt --root-user-action ignore
printf "Y\n" | ./configure.sh
printf "\n:::::::: Starting Bazel build ::::::::\n\n"
bazel build ${build_flags} release:build_pip_package
printf "\n:::::::: Creating Python wheel ::::::::\n\n"
bazel-bin/release/build_pip_package /build_output/
if [[ "${cleanup}" == "true" ]]; then
  printf "\n:::::::: Cleaning up ::::::::\n\n"
  bazel clean --async
fi
EOF

chmod +x "${build_script}"

# Use 'set --' to build the command in the positional parameters ($1, $2, ...)
set -- docker run -it --rm --network host \
  -w /tfq \
  -v "${repo_dir}":/tfq \
  -v /tmp/tensorflow_quantum:/build_output \
  -v "${build_script}:/tmp/build_script.sh" \
  -e HOST_PERMS="$(id -u):$(id -g)" \
  -e build_flags="--crosstool_top=${crosstool} ${BUILD_OPTIONS}" \
  -e cuda_version="${cuda_version}" \
  -e py_version="${py_version}" \
  -e tf_version="${tf_version}" \
  -e docker_image="${docker_image}" \
  -e crosstool="${crosstool}" \
  -e cleanup="${cleanup}" \
  "${docker_image}" \
  /tmp/build_script.sh

if [[ "${dry_run}" == "true" ]]; then
  # Loop through the positional parameters and simply print them.
  printf "(Dry run) "
  printf '%s ' "$@"
else
  echo "Spinning up a Docker container with ${docker_image} …"
  "$@"

  echo "Done. Look for wheel in /tmp/tensorflow_quantum/."
  ls -l /tmp/tensorflow_quantum/
fi
