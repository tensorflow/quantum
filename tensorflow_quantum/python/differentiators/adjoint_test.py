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
# =============================================================================
"""Tests for the differentiator abstract class."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import tensorflow as tf

from tensorflow_quantum.python.differentiators import adjoint
from tensorflow_quantum.core.ops import circuit_execution_ops


class AdjointTest(tf.test.TestCase):
    """Test that we can properly subclass differentiator."""

    def test_instantiation(self):
        """Test that adjoint can be created."""
        adjoint.Adjoint()

    def test_sample_errors(self):
        """Ensure that the adjoint method won't attach to sample ops."""

        dif = adjoint.Adjoint()
        op = circuit_execution_ops.get_sampled_expectation_op()
        with self.assertRaisesRegex(ValueError, expected_regex='not supported'):
            dif.generate_differentiable_op(sampled_op=op)

    def test_no_gradient_circuits(self):
        """Confirm the adjoint differentiator has no gradient circuits."""
        dif = adjoint.Adjoint()
        with self.assertRaisesRegex(NotImplementedError,
                                    expected_regex="no accessible "
                                    "gradient circuits"):
            _ = dif.get_gradient_circuits(None, None, None)


if __name__ == '__main__':
    tf.test.main()
