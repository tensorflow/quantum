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
"""Module to load python op libraries."""

import os
from distutils.sysconfig import get_python_lib

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader


def load_module(name):
    """Loads the module with the given name.

    First attempts to load the module as though it was embedded into the binary
    using Bazel. If that fails, then it attempts to load the module as though
    it was installed in site-packages via PIP.

    Args:
        name: The name of the module, e.g. "_tfq_simulate_ops.so"

    Returns:
        A python module containing the Python wrappers for the Ops.

    Raises:
        RuntimeError: If the library cannot be found.
    """
    try:
        path = resource_loader.get_path_to_datafile(name)
        return load_library.load_op_library(path)
    except:
        path = os.path.join(get_python_lib(), "tensorflow_quantum/core/ops",
                            name)
        return load_library.load_op_library(path)
