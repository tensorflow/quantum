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


class _LazyLoader:
    """Lazily loads a TensorFlow op library on first attribute access.

    This defers the call to `load_library.load_op_library` until the module
    is actually used, preventing TensorFlow device initialization at import
    time. This allows users to configure TensorFlow devices (e.g., enabling
    memory growth) after importing tensorflow_quantum.
    """

    def __init__(self, name):
        """Initialize the lazy loader.

        Args:
            name: The name of the module, e.g. "_tfq_simulate_ops.so"
        """
        self._name = name
        self._module = None

    def _load(self):
        """Load the module if not already loaded."""
        if self._module is None:
            try:
                path = resource_loader.get_path_to_datafile(self._name)
                self._module = load_library.load_op_library(path)
            except:
                path = os.path.join(get_python_lib(),
                                    "tensorflow_quantum/core/ops", self._name)
                self._module = load_library.load_op_library(path)
        return self._module

    def __getattr__(self, name):
        """Load the module on first attribute access and delegate."""
        module = self._load()
        return getattr(module, name)


def load_module(name):
    """Returns a lazy loader for the module with the given name.

    The actual library loading is deferred until the module is first used.
    This prevents TensorFlow device initialization at import time, allowing
    users to configure TensorFlow devices after importing tensorflow_quantum.

    Args:
        name: The name of the module, e.g. "_tfq_simulate_ops.so"

    Returns:
        A lazy loader object that behaves like the loaded module but defers
        loading until first attribute access.

    Raises:
        RuntimeError: If the library cannot be found when first accessed.
    """
    return _LazyLoader(name)
