#!/bin/bash
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"

function write_to_bazelrc() {
  echo "$1" >> .bazelrc
}

function write_action_env_to_bazelrc() {
  write_to_bazelrc "$1 --action_env $2=\"$3\""
}

function write_linkopt_dir_to_bazelrc() {
  write_to_bazelrc "$1 --linkopt -Wl,-rpath,$2" >> .bazelrc
}


function is_linux() {
  [[ "${PLATFORM}" == "linux" ]]
}

function is_macos() {
  [[ "${PLATFORM}" == "darwin" ]]
}

function is_windows() {
  # On windows, the shell script is actually running in msys
  [[ "${PLATFORM}" =~ msys_nt*|mingw*|cygwin*|uwin* ]]
}

function is_ppc64le() {
  [[ "$(uname -m)" == "ppc64le" ]]
}


# Remove .bazelrc if it already exist
[ -e .bazelrc ] && rm .bazelrc

# Check if we are building TFQ GPU or not (TODO)
while [[ "$TFQ_NEED_CUDA" == "" ]]; do
  read -p "Do you want to build TFQ against CPU?"\
" Y or enter for CPU, N for GPU. [Y/n] " INPUT
  case $INPUT in
    [Yy]* ) echo "Build with CPU ops only."; TFQ_NEED_CUDA=0;;
    [Nn]* ) echo "Build with cuQuantum support."; TFQ_NEED_CUDA=1;;
    "" ) echo "Build with CPU ops only."; TFQ_NEED_CUDA=0;;
    * ) echo "Invalid selection: " $INPUT;;
  esac
done

# Set the CUDA SDK version for TF
if [[ "$TFQ_NEED_CUDA" == "1" ]]; then
  _DEFAULT_CUDA_VERSION=11
  while [[ "$TF_CUDA_VERSION" == "" ]]; do
    read -p "Please specify the CUDA SDK major version you want to use. [Leave empty to default to CUDA $_DEFAULT_CUDA_VERSION]: " INPUT
    case $INPUT in
      "" ) echo "Build against CUDA $_DEFAULT_CUDA_VERSION."; TF_CUDA_VERSION=$_DEFAULT_CUDA_VERSION;;
      # check if the input is a number
      *[!0-9]* ) echo "Invalid selection: $INPUT";;
      * ) echo "Build against CUDA $INPUT."; TF_CUDA_VERSION=$INPUT;;
    esac
  done
fi

# If TFQ_NEED_CUDA then enforce building against TensorFlow 2.11 or newer.
IS_VALID_TF_VERSION=$(python -c "import tensorflow as tf; v = tf.__version__; print(float(v[:v.rfind('.')]) < 2.11)")
TF_VERSION=$(python -c "import tensorflow as tf; print(tf.__version__)")
if [[ $IS_VALID_TF_VERSION == "True" ]]; then
  echo "Building against TensorFlow 2.11 or newer is required."
  echo "Please upgrade your TensorFlow version."
  exit 1
elif [[ $IS_VALID_TF_VERSION == "False" ]]; then
  echo "Using TensorFlow 2.11"
else
  echo "Unable to determine TensorFlow version."
  exit 1
fi

# Check if we are building cuQuantum ops on top of CUDA.
if [[ "$TFQ_NEED_CUDA" == "1" ]]; then
  if [[ "$CUQUANTUM_ROOT" != "" ]]; then
    echo "  [*] cuQuantum library is detected here: CUQUANTUM_ROOT=$CUQUANTUM_ROOT."
  else
    # Prompt the user to enter the cuQuantum root path, do not allow empty input (pressing enter)
    # If the user enters an invalid path, prompt again.
    while true; do
      read -p "Please specify the cuQuantum root directory: " INPUT
      if [[ -z "$INPUT" ]]; then
        echo "Input cannot be empty. Please enter a valid path."
      elif [[ "$INPUT" =~ ^(/[A-Za-z0-9_-]+)+$ ]]; then
        echo "Path pattern is valid: $INPUT"
        CUQUANTUM_ROOT=$INPUT
        break
      else
        echo "Invalid path pattern: $INPUT. Please enter a valid path."
      fi
    done
  fi
  write_action_env_to_bazelrc "build:cuda" "CUQUANTUM_ROOT" ${CUQUANTUM_ROOT}
  write_linkopt_dir_to_bazelrc "build:cuda" "${CUQUANTUM_ROOT}/lib"
fi

# Check if it's installed
if [[ $(pip show tensorflow) == *tensorflow* ]] || [[ $(pip show tf-nightly) == *tf-nightly* ]]; then
  echo "Using installed tensorflow-($TF_VERSION)"
else
  echo 'Installing tensorflow 2.11 .....\n'
  pip install tensorflow==2.11.0
fi



TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS="$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"


write_to_bazelrc "build --experimental_repo_remote_exec"
write_to_bazelrc "build --spawn_strategy=standalone"
write_to_bazelrc "build --strategy=Genrule=standalone"
write_to_bazelrc "build -c opt"
write_to_bazelrc "build --cxxopt=\"-D_GLIBCXX_USE_CXX11_ABI=1\""
write_to_bazelrc "build --cxxopt=\"-std=c++17\""
write_to_bazelrc "build --cxxopt=\"-O3\""
write_to_bazelrc "build --cxxopt=\"-march=native\""

if is_windows; then
  # Use pywrap_tensorflow instead of tensorflow_framework on Windows
  SHARED_LIBRARY_DIR=${TF_CFLAGS:2:-7}"python"
else
  SHARED_LIBRARY_DIR=${TF_LFLAGS:2}
fi
SHARED_LIBRARY_NAME=$(echo $TF_LFLAGS | rev | cut -d":" -f1 | rev)
if ! [[ $TF_LFLAGS =~ .*:.* ]]; then
  if is_macos; then
    SHARED_LIBRARY_NAME="libtensorflow_framework.dylib"
  elif is_windows; then
    # Use pywrap_tensorflow's import library on Windows. It is in the same dir as the dll/pyd.
    SHARED_LIBRARY_NAME="_pywrap_tensorflow_internal.lib"
  else
    SHARED_LIBRARY_NAME="libtensorflow_framework.so"
  fi
fi

HEADER_DIR=${TF_CFLAGS:2}
if is_windows; then
  SHARED_LIBRARY_DIR=${SHARED_LIBRARY_DIR//\\//}
  SHARED_LIBRARY_NAME=${SHARED_LIBRARY_NAME//\\//}
  HEADER_DIR=${HEADER_DIR//\\//}
fi

TF_NEED_CUDA=${TFQ_NEED_CUDA}
write_action_env_to_bazelrc "build" "TF_HEADER_DIR" ${HEADER_DIR} ""
write_action_env_to_bazelrc "build" "TF_SHARED_LIBRARY_DIR" ${SHARED_LIBRARY_DIR} ""
write_action_env_to_bazelrc "build" "TF_SHARED_LIBRARY_NAME" ${SHARED_LIBRARY_NAME} ""
write_action_env_to_bazelrc "build" "TF_NEED_CUDA" ${TF_NEED_CUDA} ""

if ! is_windows; then
  write_linkopt_dir_to_bazelrc "build"  ${SHARED_LIBRARY_DIR} ""
fi

# TODO(yifeif): do not hardcode path
if [[ "$TF_NEED_CUDA" == "1" ]]; then
  write_to_bazelrc "build:cuda --experimental_repo_remote_exec"
  write_to_bazelrc "build:cuda --spawn_strategy=standalone"
  write_to_bazelrc "build:cuda --strategy=Genrule=standalone"
  write_to_bazelrc "build:cuda -c opt"
  write_to_bazelrc "build:cuda --cxxopt=\"-D_GLIBCXX_USE_CXX11_ABI=1\""
  write_to_bazelrc "build:cuda --cxxopt=\"-std=c++17\""
  write_to_bazelrc "build:cuda --cxxopt=\"-O3\""
  write_to_bazelrc "build:cuda --cxxopt=\"-march=native\""
  write_to_bazelrc "build:cuda --@local_config_cuda//:enable_cuda"
  write_to_bazelrc "build:cuda --crosstool_top=@local_config_cuda//crosstool:toolchain"

  write_action_env_to_bazelrc "build:cuda" "TF_CUDA_VERSION" ${TF_CUDA_VERSION} 
  write_action_env_to_bazelrc "build:cuda" "TF_CUDNN_VERSION" "8"
  if is_windows; then
    write_action_env_to_bazelrc "build:cuda" "CUDNN_INSTALL_PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${TF_CUDA_VERSION}"
    write_action_env_to_bazelrc "build:cuda" "CUDA_TOOLKIT_PATH" "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${TF_CUDA_VERSION}"
  else
    write_action_env_to_bazelrc "build:cuda" "CUDNN_INSTALL_PATH" "/usr/lib/x86_64-linux-gnu"
    write_action_env_to_bazelrc "build:cuda" "CUDA_TOOLKIT_PATH" "/usr/local/cuda"
  fi
  write_to_bazelrc "build --config=cuda"
  write_to_bazelrc "test --config=cuda"
fi

