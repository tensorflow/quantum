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
"""Tests for quantum_context functions."""
import pathos
import tensorflow as tf
from absl.testing import parameterized

from tensorflow_quantum.python import quantum_context


class QContextTest(tf.test.TestCase, parameterized.TestCase):
    """Test that quantum context objects work."""

    def test_global_singleton(self):
        """Test that context object is a true singleton."""

        def filler(i):
            return quantum_context.q_context()

        pool = pathos.pools.ThreadPool()
        results = pool.map(filler, range(500))
        self.assertTrue(all(r is quantum_context.q_context() for r in results))

    def test_global_not_singleton(self):
        """In the case of Processes singleton objects will be reset."""

        def filler(i):
            return quantum_context.q_context()

        pool = pathos.pools.ProcessPool()
        results = pool.map(filler, range(500))
        self.assertFalse(all(r is quantum_context.q_context() for r in results))

    def test_global_engine_mode(self):
        """Test getter an setter behavior for engine_mode."""
        mode = quantum_context.get_engine_mode()
        self.assertFalse(mode)
        quantum_context.set_engine_mode(True)
        mode = quantum_context.get_engine_mode()
        self.assertTrue(mode)


if __name__ == "__main__":
    tf.test.main()
