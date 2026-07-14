#!/bin/bash
# Copyright 2020 The TensorFlow Quantum Authors. All Rights Reserved.
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
# ==============================================================================

set -e

# Use legacy tf.keras (Keras 2) with TensorFlow 2.17+.
export TF_USE_LEGACY_KERAS=1

# Tools for running notebooks non-interactively
pip install \
  "nbclient==0.6.5" \
  "jupyter-client==7.4.9" \
  "ipython>=8.10.0" \
  "ipykernel>=6.29.0"

# Gymnasium pip package needed for the quantum reinforcement learning tutorial
pip install "gymnasium[classic-control]==1.2.3"
# seaborn has also numpy dependency, it requires version >= 0.12.0.
pip install seaborn==0.12.0
# tf_docs pip package needed for noise tutorial.
pip install -q git+https://github.com/tensorflow/docs

# Leave the repository directory, otherwise errors may occur
thisdir=$(CDPATH="" cd -- "$(dirname -- "${0}")" && pwd -P)
repo_dir=$(git -C "${thisdir}" rev-parse --show-toplevel 2>/dev/null)
parent_dir=$(dirname "${repo_dir}")
repo_name=$(basename "${repo_dir}")

cd "${parent_dir}"

examples_output=$(python3 "${repo_name}/scripts/test_tutorials.py")
exit_code=$?

if [ "$exit_code" == "0" ]; then
	exit 0;
else
	echo "Tutorials failed to run to completion:"
	echo "{$examples_output}"
	exit 64;
fi
