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

set -eu -o pipefail

# Summary: do all the steps to generate a wheel for a TFQ release.
#
# This sets up a clean pyenv virtualenv for a given Python version, then runs
# release/build_distribution.sh and release/clean_distribution.sh, and finishes
# by printing some info about the wheel. The wheel is left in ./wheelhouse/.
# The TFQ release number is extracted from setup.py.
#
# Note: this uses build_distribution.sh, which builds the TFQ pip package inside
# a Docker container. The TFQ git directory where this script is found is mapped
# directly inside the running Docker environment. The approach makes it easy to
# iterate on changes to TFQ files in the current directory, and avoids a lot of
# frustrating "which copy of such-and-such file did it use?" questions. However,
# it also brings a risk of unexpected impact of left-overs in the current
# directory. To avoid this, it's best to make a copy of your TFQ git repository
# (or git-clone a fresh copy from GitHub) and clean it before proceeding.

# ~~~~~~~~ Helpers ~~~~~~~~

function quit() {
  echo "ERROR: $*." >&2
  exit 1
}

function have() {
  unset -v have
  type $1 &> /dev/null
}

# ~~~~~~~~ Sanity checks ~~~~~~~~

# Go to the top of the local TFQ git tree. Do it early in case this fails.
thisdir=$(CDPATH="" cd -- "$(dirname -- "$0")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel 2>/dev/null || \
  quit "this script must be run from inside the TFQ git tree")
cd "${repo_dir}"

py_version="${1}"
if ! [[ "${py_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  quit "the first argument must be a full Python version number"
fi

build_number=""
if (( $# > 1 )); then
  build_number="-${2}"
fi

setup_file="${repo_dir}/release/setup.py"
if [[ -r "${setup_file}" ]]; then
  tfq_version=$(grep -m 1 CUR_VERSION "${setup_file}" | cut -f2 -d'"')
else
  quit "cannot read ${setup_file}"
fi

for program in python3 pip pyenv jq; do
  if ! have $program; then
    quit "cannot run $program -- maybe it is not installed?"
  fi
done

version=${tfq_version}${build_number}

# ~~~~~~~~ Set up a new virtual environment ~~~~~~~~

# Since the build is done inside a Docker container, it is not really necessary
# to create a virtual Python environment for that part of the process. However,
# we do run some Python commands before and after, and we want those to be done
# in an environment we control with the same Python version being targeted for
# the build. It provides additional isolation.

echo "~~~~ Starting ${0} for TFQ release ${version}"
echo "~~~~ Current directory: $(pwd)"
echo "~~~~ (Re)creating virtual environment 'tfq-build-venv'"
echo

# Ensure pyenv is activated.
eval "$(pyenv init -)"

# Deactivate any pyenv we might be inside right now.
pyenv deactivate >& /dev/null  || true

# Ensure we have the requested version of Python.
pyenv install -s ${py_version}

# (Re)create a pyenv virtual env with an expressive name.
pyenv virtualenv-delete -f tfq-build-venv || true
pyenv virtualenv -v ${py_version} tfq-build-venv
pyenv activate tfq-build-venv

pip install --upgrade pip

# ~~~~~~~~ Build & clean the wheel ~~~~~~~~

echo
echo "~~~~ Starting build of TFQ ${version}."
echo

./release/build_distribution.sh -p "${py_version}"

# The wheel that was just created will be the most recent file.
tmp_wheel_name="$(/bin/ls -t /tmp/tensorflow_quantum | head -n 1)"
tmp_wheel="/tmp/tensorflow_quantum/${tmp_wheel_name}"

echo
echo "~~~~ Cleaning wheel ${tmp_wheel}"
echo
./release/clean_distribution.sh "${tmp_wheel}"

# ~~~~~~~~ Check the result ~~~~~~~~

echo
echo "~~~~ Inspecting the wheel."

pip install -qq wheel-inspect check-wheel-contents

final_wheel="wheelhouse/$(/bin/ls -t ./wheelhouse | head -n 1)"

echo "Check wheel contents:"
echo
check-wheel-contents "${final_wheel}"

echo
echo "Requires_python value in wheel: "
wheel2json "${final_wheel}" | jq -r '.dist_info.metadata."requires_python"'

echo
echo "Tags in wheel:"
wheel2json "${final_wheel}" | jq -r '.dist_info.wheel.tag[]'

echo
echo "~~~~ All done."
echo "${final_wheel}"
