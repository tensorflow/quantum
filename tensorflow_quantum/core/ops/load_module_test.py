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

import subprocess
import sys
import textwrap
import types

import tensorflow as tf
from tensorflow_quantum.core.ops.load_module import load_module, _LazyLoader


class LoadModuleTest(tf.test.TestCase):
    """Tests for load_module function and _LazyLoader class."""

    def test_load_module_returns_lazy_loader(self):
        """Test that load_module returns a _LazyLoader instance."""
        loader = load_module("_tfq_utility_ops.so")
        self.assertIsInstance(loader, _LazyLoader)

    def test_lazy_loader_is_module_type(self):
        """Test that _LazyLoader is a subclass of types.ModuleType."""
        loader = load_module("_tfq_utility_ops.so")
        self.assertIsInstance(loader, types.ModuleType)

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

    def test_lazy_loader_dir_introspection(self):
        """Test that dir() works on the lazy loader for introspection."""
        loader = load_module("_tfq_utility_ops.so")
        # dir() should return a list of attributes from the loaded module
        attributes = dir(loader)
        self.assertIsInstance(attributes, list)
        # Should contain the op we know exists
        self.assertIn("tfq_append_circuit", attributes)

    def test_device_config_after_import(self):
        """Test that device configuration works after importing TFQ.

        This test runs in a subprocess to ensure a fresh Python interpreter
        with no prior TensorFlow device initialization. It verifies that
        importing tensorflow_quantum does not prematurely initialize devices,
        allowing users to call set_logical_device_configuration afterward.
        """
        # Python code to run in a subprocess with a fresh interpreter
        test_code = textwrap.dedent("""
            import sys
            import tensorflow as tf

            # Import tensorflow_quantum BEFORE configuring devices.
            # This used to trigger device initialization and cause errors.
            import tensorflow_quantum as tfq

            # Get physical CPUs (always available, works on CI)
            cpus = tf.config.list_physical_devices('CPU')
            if not cpus:
                print("ERROR: No CPUs found")
                sys.exit(1)

            try:
                # Try setting logical device configuration on CPUs.
                # This would fail with RuntimeError if devices were already
                # initialized during the TFQ import.
                tf.config.set_logical_device_configuration(
                    cpus[0],
                    [tf.config.LogicalDeviceConfiguration(),
                     tf.config.LogicalDeviceConfiguration()]
                )
                print("SUCCESS")
                sys.exit(0)
            except RuntimeError as e:
                print(f"FAILED: {e}")
                sys.exit(1)
        """)

        result = subprocess.run([sys.executable, "-c", test_code],
                                capture_output=True,
                                text=True,
                                check=False)

        error_msg = (f"Device configuration after import failed.\n"
                     f"stdout: {result.stdout}\nstderr: {result.stderr}")
        self.assertEqual(result.returncode, 0, error_msg)
        self.assertIn("SUCCESS", result.stdout)


if __name__ == '__main__':
    tf.test.main()
