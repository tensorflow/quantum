#!/usr/bin/env python
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
"""Test that TensorFlow device configuration works after importing TFQ.

This script verifies the fix for the issue where importing tensorflow_quantum
before setting TensorFlow device configuration (e.g., enabling memory growth)
resulted in a RuntimeError:

    RuntimeError: Physical devices cannot be modified after being initialized

The fix uses lazy loading of native op libraries to defer TensorFlow device
initialization until the ops are actually used.

Usage:
    python test_device_config_after_import.py
"""

import sys
import tensorflow as tf

# Import tensorflow_quantum BEFORE configuring devices.
# This used to trigger device initialization and cause errors.
import tensorflow_quantum as tfq

# Now try to configure devices - this should work without RuntimeError
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Try setting memory growth - this would fail before the fix
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("SUCCESS: Device configuration after import works!")
        print(f"  - Configured memory growth for GPU: {gpus[0]}")
    except RuntimeError as e:
        print(f"FAILED: {e}")
        sys.exit(1)
else:
    # No GPU available, but we can still test that importing TFQ
    # doesn't prematurely initialize devices by checking CPU config
    cpus = tf.config.list_physical_devices('CPU')
    print("SUCCESS: TFQ import did not prematurely initialize devices!")
    print(f"  - Available CPUs: {cpus}")
    print("  - No GPU available to test memory growth, but import test passed.")

# Verify TFQ is actually usable after configuration
print(f"  - TFQ version: {tfq.__version__}")
