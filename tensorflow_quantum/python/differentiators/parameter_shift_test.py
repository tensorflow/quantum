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
"""Basic tests for the ParameterShift differentiator"""
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import sympy
import cirq

from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import parameter_shift
from tensorflow_quantum.core.ops import circuit_execution_ops


def _simple_op_inputs():
    qubit = cirq.GridQubit(0, 0)
    symbol = 'alpha'
    circuit = cirq.Circuit(cirq.Y(qubit)**sympy.Symbol(symbol))
    op = cirq.X(qubit)
    value = 0.3
    n_samples = 2000

    # Return inputs prepped for expectation ops.
    # circuit, symbol_names, values, ops, n_samples
    # along with expected feedforward expectation
    # and expected gradient.
    return (util.convert_to_tensor([circuit]), tf.convert_to_tensor([symbol]),
            tf.convert_to_tensor([[value]]), util.convert_to_tensor([[op]]),
            tf.convert_to_tensor([[n_samples]]),
            tf.convert_to_tensor([[np.sin(np.pi * value)]]),
            tf.convert_to_tensor([[np.pi * np.cos(np.pi * value)]]))


class ParameterShiftTest(tf.test.TestCase, parameterized.TestCase):
    """Test the ParameterShift Differentiator will run end to end."""

    def test_no_gradient_circuits(self):
        """Confirm ParameterShift differentiator has no gradient circuits."""
        dif = linear_combination.ParameterShift()
        with self.assertRaisesRegex(NotImplementedError,
                                    expected_regex="not currently available"):
            _ = dif.get_gradient_circuits(None, None, None)

    def test_parameter_shift_analytic(self):
        """Test if ParameterShift.differentiate_analytical doesn't crash before
        running."""
        programs, names, values, ops, _, true_f, true_g = \
        _simple_op_inputs()

        ps = parameter_shift.ParameterShift()
        op = ps.generate_differentiable_op(
            analytic_op=circuit_execution_ops.get_expectation_op())

        with tf.GradientTape() as g:
            g.watch(values)
            expectations = op(programs, names, values, ops)
        grads = g.gradient(expectations, values)
        self.assertAllClose(expectations, true_f, atol=1e-2, rtol=1e-2)
        self.assertAllClose(grads, true_g, atol=1e-2, rtol=1e-2)

    def test_parameter_shift_sampled(self):
        """Test if ParameterShift.differentiate_sampled doesn't crash before
        running."""
        programs, names, values, ops, n_samples, true_f, true_g = \
        _simple_op_inputs()
        ps = parameter_shift.ParameterShift()
        op = ps.generate_differentiable_op(
            sampled_op=circuit_execution_ops.get_sampled_expectation_op())

        with tf.GradientTape() as g:
            g.watch(values)
            expectations = op(programs, names, values, ops, n_samples)
        grads = g.gradient(expectations, values)
        self.assertAllClose(expectations, true_f, atol=1e-1, rtol=1e-1)
        self.assertAllClose(grads, true_g, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    tf.test.main()
