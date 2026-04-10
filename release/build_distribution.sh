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

# Summary: build a wheel for TFQ using a TensorFlow build container.
# Run this script with the option "-h" to get a usage summary.
#
# To ensure binary compatibility with TensorFlow, TFQ distributions are built
# using TensorFlow's build containers and crosstool C++ toolchain. This
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
# 3. Exit.

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
py_version=$(python3 --version | cut -d' ' -f2 | cut -d. -f1,2)
extra_bazel_options=""
cleanup="true"

usage="Usage: ${0} [OPTIONS]
Build a Python wheel for a distribution of TensorFlow Quantum.
Options:
  -o \"options\"  Additional options to pass to Bazel build
  -p X.Y        Use Python version X.Y (default: ${py_version})
  -e            Don't run bazel clean at the end (default: do)
  -h            Show this help message and exit"

while getopts "b:ehp:" opt; do
  case "${opt}" in
    b) extra_bazel_options="${OPTARG}" ;;
    e) cleanup="false" ;;
    h) echo "${usage}"; exit 0 ;;
    p) py_version=$(echo "${OPTARG}" | cut -d. -f1,2) ;;
    *) quit "${usage}" ;;
  esac
done
shift $((OPTIND -1))

# See https://hub.docker.com/r/tensorflow/build/tags for available containers.
docker_image="tensorflow/build:2.18-python${py_version}"

# The next values must match what is used by the TF release being targeted. Look
# in the TF .bazelrc file for the Linux CPU builds, specifically rbe_linux_cpu.
tf_compiler="/usr/lib/llvm-18/bin/clang"
tf_sysroot="/dt9"
tf_crosstool="@local_config_cuda//crosstool:toolchain"

# Create a script to be run by the shell inside the Docker container.
build_script=$(mktemp /tmp/tfq_build.XXXXXX)
trap 'rm -f "${build_script}" || true' EXIT

# The printf'ed section dividers are to make it easier to search the output.
cat <<EOF > "${build_script}"
#!/bin/bash
set -o errexit
cd /tfq
exec > >(sed "s/^/[DOCKER] /")
exec 2> >(sed "s/^/[DOCKER] /" >&2)

printf ":::::::: Configuring Python environment ::::::::\n\n"
python3 -m pip install --upgrade pip --root-user-action ignore
python3 -m pip install -r requirements.txt --root-user-action ignore

printf "\n\n:::::::: Configuring TFQ build ::::::::\n\n"
printf "Y\n" | ./configure.sh

printf "\n\n:::::::: Build configuration inside Docker container ::::::::\n"
printf "  Docker image:     ${docker_image}\n"
printf "  Crosstool:        ${tf_crosstool}\n"
printf "  Compiler:         ${tf_compiler}\n"
printf "  TF_SYSROOT:       ${tf_sysroot}\n"
printf "  Python version:   "
python3 --version | cut -d' ' -f2

printf "\n:::::::: Starting Bazel build ::::::::\n\n"
bazel build \
  --cxxopt=-O3 --cxxopt=-msse2 --cxxopt=-msse3 --cxxopt=-msse4 \
  ${extra_bazel_options} \
  --crosstool_top="${tf_crosstool}" \
  --host_crosstool_top="${tf_crosstool}" \
  --extra_toolchains="${tf_crosstool}-linux-x86_64" \
  --repo_env=CC="${tf_compiler}" \
  --repo_env=TF_SYSROOT="${tf_sysroot}" \
  release:build_pip_package

printf "\n:::::::: Creating Python wheel ::::::::\n\n"
bazel-bin/release/build_pip_package /build_output/

if [[ "${cleanup}" == "true" ]]; then
  printf "\n:::::::: Cleaning up ::::::::\n\n"
  bazel clean --async
fi
EOF

chmod +x "${build_script}"

echo "Spinning up a Docker container with ${docker_image} …"
docker run -it --rm --network host \
  -w /tfq \
  -v "${repo_dir}":/tfq \
  -v /tmp/tensorflow_quantum:/build_output \
  -v "${build_script}:/tmp/build_script.sh" \
  -e HOST_PERMS="$(id -u):$(id -g)" \
  "${docker_image}" \
  /tmp/build_script.sh

echo "Done. Look for wheel in /tmp/tensorflow_quantum/."
ls -l /tmp/tensorflow_quantum/
