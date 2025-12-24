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
"""Tests for load_module lazy loading functionality."""
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_quantum.core.ops.load_module import load_module, _LazyLoader


class LoadModuleTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for load_module function and _LazyLoader class."""

    def test_load_module_returns_lazy_loader(self):
        """Test that load_module returns a _LazyLoader instance."""
        loader = load_module("_tfq_utility_ops.so")
        self.assertIsInstance(loader, _LazyLoader)

    def test_lazy_loader_defers_loading(self):
        """Test that _LazyLoader does not load the module on construction."""
        loader = _LazyLoader("_tfq_utility_ops.so")
        # _module should be None before any attribute access
        self.assertIsNone(loader._module)

    def test_lazy_loader_loads_on_attribute_access(self):
        """Test that _LazyLoader loads the module on attribute access."""
        loader = load_module("_tfq_utility_ops.so")
        # Access an attribute to trigger loading
        _ = loader.tfq_append_circuit
        # Now _module should be loaded
        self.assertIsNotNone(loader._module)

    def test_lazy_loader_attribute_access_works(self):
        """Test that attributes from the loaded module are accessible."""
        loader = load_module("_tfq_utility_ops.so")
        # Accessing an op should return a callable
        self.assertTrue(callable(loader.tfq_append_circuit))


if __name__ == '__main__':
    tf.test.main()
