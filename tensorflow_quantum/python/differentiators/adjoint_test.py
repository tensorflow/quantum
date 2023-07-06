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
"""Tests for the differentiator abstract class."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position
from unittest import mock

from absl.testing import parameterized
import cirq
import numpy as np
import sympy
import tensorflow as tf

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import adjoint


class AdjointTest(tf.test.TestCase, parameterized.TestCase):
    """Test that we can properly subclass differentiator."""

    def test_instantiation(self):
        """Test that adjoint can be created."""
        adjoint.Adjoint()

    @parameterized.parameters(
        list(util.kwargs_cartesian_product(**{
            'use_cuquantum': [False, True],
        })))
    def test_use_cuquantum(self, use_cuquantum):
        """Ensure that use_cuquantum switches to cuquantum ops well."""
        if not circuit_execution_ops.is_gpu_configured():
            # Ignores this test if gpu is not configured.
            self.skipTest("GPU is not set. Ignoring gpu tests...")
        # Prepares a simple circuit.
        qubit = cirq.GridQubit(0, 0)
        circuit = util.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubit)**sympy.Symbol('alpha'))])
        psums = util.convert_to_tensor([[cirq.Z(qubit)]])
        symbol_values_array = np.array([[0.123]], dtype=np.float32)
        symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)

        # Mocks `Adjoint.differentiate_analytic*()` to check if
        # it's called once correctly.
        method_name = ("differentiate_analytic_cuquantum"
                       if use_cuquantum else "differentiate_analytic")
        with mock.patch.object(adjoint.Adjoint,
                               method_name,
                               return_value=None,
                               autospec=True) as mock_adj:
            dif = adjoint.Adjoint()
            op = circuit_execution_ops.get_expectation_op(
                use_cuquantum=use_cuquantum, quantum_concurrent=False)
            diff_op = dif.generate_differentiable_op(
                analytic_op=op, use_cuquantum=use_cuquantum)

            # Calculate tfq gradient.
            with tf.GradientTape() as g:
                g.watch(symbol_values_tensor)
                expectations = diff_op(circuit, tf.convert_to_tensor(['alpha']),
                                       symbol_values_tensor, psums)
            _ = g.gradient(expectations, symbol_values_tensor)
        mock_adj.assert_called_once()

    def test_sample_errors(self):
        """Ensure that the adjoint method won't attach to sample ops."""

        dif = adjoint.Adjoint()
        op = circuit_execution_ops.get_sampled_expectation_op()
        with self.assertRaisesRegex(ValueError, expected_regex='not supported'):
            dif.generate_differentiable_op(sampled_op=op)
        with self.assertRaisesRegex(ValueError, expected_regex='not supported'):
            dif.generate_differentiable_op(sampled_op=op, use_cuquantum=True)

    def test_no_gradient_circuits(self):
        """Confirm the adjoint differentiator has no gradient circuits."""
        dif = adjoint.Adjoint()
        with self.assertRaisesRegex(NotImplementedError,
                                    expected_regex="no accessible "
                                    "gradient circuits"):
            _ = dif.get_gradient_circuits(None, None, None)


if __name__ == '__main__':
    tf.test.main()
