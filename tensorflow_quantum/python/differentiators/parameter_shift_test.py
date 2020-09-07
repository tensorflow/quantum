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
import math

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

    def test_get_intermediate_logic(self):
        """Confirm get_intermediate_logic returns the expected values."""
        symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
        q0 = cirq.GridQubit(0, 0)
        q1 = cirq.GridQubit(1, 2)
        test_programs = util.convert_to_tensor([
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q1)**symbols[0],
                cirq.Z(q0)**symbols[1]),
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.Z(q1)**symbols[1]),
        ])
        test_symbol_names = tf.constant([str(s) for s in symbols])
        test_symbol_values = tf.constant([
            [1.5, -2.7],
            [-0.3, 0.9],
        ])
        test_pauli_sums = util.convert_to_tensor([
            [cirq.X(q0), cirq.Z(q1)],
            [cirq.X(q1), cirq.Y(q0)],
        ])

        test_parameter_shift = parameter_shift.ParameterShift()

        # For each program in the batch, we need to make two copies of that
        # program for each parameterized gate, where the parameter is replaced
        # by the special parameter name `_param_shift`.
        # For each symbol in each program, when there are fewer parameterized
        # gates for that symbol than the maximum number over all circuits and
        # symbols, then empty circuits are padded in to compensate.
        special = sympy.Symbol("_param_shift")
        expected_programs_0 = util.convert_to_tensor([
            cirq.Circuit(
                cirq.X(q0)**special,
                cirq.Y(q1)**symbols[0],
                cirq.Z(q0)**symbols[1]),
            cirq.Circuit(
                cirq.X(q0)**special,
                cirq.Y(q1)**symbols[0],
                cirq.Z(q0)**symbols[1]),
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q1)**special,
                cirq.Z(q0)**symbols[1]),
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q1)**special,
                cirq.Z(q0)**symbols[1]),
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q1)**symbols[0],
                cirq.Z(q0)**special),
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q1)**symbols[0],
                cirq.Z(q0)**special),
            cirq.Circuit(),
            cirq.Circuit(),
        ])
        expected_programs_1 = util.convert_to_tensor([
            cirq.Circuit(cirq.X(q0)**special,
                         cirq.Z(q1)**symbols[1]),
            cirq.Circuit(cirq.X(q0)**special,
                         cirq.Z(q1)**symbols[1]),
            cirq.Circuit(),
            cirq.Circuit(),
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.Z(q1)**special),
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.Z(q1)**special),
            cirq.Circuit(),
            cirq.Circuit(),
        ])
        expected_batch_programs = tf.concat([
            tf.expand_dims(expected_programs_0, 0),
            tf.expand_dims(expected_programs_1, 0),
        ], 0)

        # Symbol names have the special parameter appended.
        out_symbol_names = tf.constant([["s0", "s1", "_param_shift"]])
        expected_batch_symbol_names = tf.tile(out_symbol_names, [2, 1])

        # Symbol values are the original values perturbed symmetrically by an
        # amount that depends on the gate.
        # For loss function f(x) = <s|w**x|s>, where w is a Pauli matrix, we
        # have from equations 8 and 10 of https://arxiv.org/abs/1905.13311 that
        # df(x)/dx = (pi/2) * [f(x + 1/2) - f(x - 1/2)].
        # In the test circuit here all circuits are exponentials of Paulis.
        # Thus all values for the input symbols are just the input; the special
        # parameter takes on the value of the symbol it has replaced, plus 0.5
        # in the first copy and minus 0.5 in the second.
        # Unused rows have the special value filled with the original value.
        expected_batch_symbol_values = tf.constant([[[1.5, -2.7, 1.5 + 0.5],
                                                     [1.5, -2.7, 1.5 - 0.5],
                                                     [1.5, -2.7, 1.5 + 0.5],
                                                     [1.5, -2.7, 1.5 - 0.5],
                                                     [1.5, -2.7, -2.7 + 0.5],
                                                     [1.5, -2.7, -2.7 - 0.5],
                                                     [1.5, -2.7, -2.7],
                                                     [1.5, -2.7, -2.7]],
                                                    [[-0.3, 0.9, -0.3 + 0.5],
                                                     [-0.3, 0.9, -0.3 - 0.5],
                                                     [-0.3, 0.9, -0.3],
                                                     [-0.3, 0.9, -0.3],
                                                     [-0.3, 0.9, 0.9 + 0.5],
                                                     [-0.3, 0.9, 0.9 - 0.5],
                                                     [-0.3, 0.9, 0.9],
                                                     [-0.3, 0.9, 0.9]]])

        # The same paulis are measured as the input, tiled up.
        max_param_gates = 2
        n_symbols = 2
        n_shifts = 2
        expected_batch_pauli_sums = tf.tile(tf.expand_dims(
            test_pauli_sums, 1), [1, max_param_gates * n_symbols * n_shifts, 1])

        # Note that we can also write the derivative equation as
        # df(x)/dx = (pi/2) * f(x + 1/2) - (pi/2) * f(x - 1/2)],
        # so +-pi/2 are the weights we need in the linear combination
        # of measurement results defining our gradients.
        # For a given program index `i`, the tensor of values is non-zero
        # only when `j == n`, since the input pauli sums are the same as the
        # gradient pauli sums.  This we can build a 2-D map for a given
        # circuit, pad it with zeros, then concatenate the maps for the
        # individual input programs.
        single_batch_mapper_0 = tf.expand_dims(
            tf.expand_dims(
                tf.constant([[
                    math.pi / 2.0, -math.pi / 2.0, math.pi / 2.0,
                    -math.pi / 2.0, 0.0, 0.0, 0.0, 0.0
                ], [
                    0.0, 0.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0, 0.0
                ]]), 0), -1)
        single_batch_mapper_1 = tf.expand_dims(
            tf.expand_dims(
                tf.constant([[
                    math.pi / 2.0, -math.pi / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ], [
                    0.0, 0.0, 0.0, 0.0, math.pi / 2.0, -math.pi / 2.0, 0.0, 0.0
                ]]), 0), -1)
        op_mapper_0_0 = tf.pad(single_batch_mapper_0,
                               [[0, 0], [0, 0], [0, 0], [0, 1]])
        op_mapper_0_1 = tf.pad(single_batch_mapper_0,
                               [[0, 0], [0, 0], [0, 0], [1, 0]])
        op_mapper_0 = tf.concat([op_mapper_0_0, op_mapper_0_1], 0)
        op_mapper_1_0 = tf.pad(single_batch_mapper_1,
                               [[0, 0], [0, 0], [0, 0], [0, 1]])
        op_mapper_1_1 = tf.pad(single_batch_mapper_1,
                               [[0, 0], [0, 0], [0, 0], [1, 0]])
        op_mapper_1 = tf.concat([op_mapper_1_0, op_mapper_1_1], 0)
        expected_batch_mapper = tf.concat(
            [tf.expand_dims(op_mapper_0, 0),
             tf.expand_dims(op_mapper_1, 0)], 0)

        (test_batch_programs, test_batch_symbol_names, test_batch_symbol_values,
         test_batch_pauli_sums,
         test_batch_mapper) = test_parameter_shift.get_intermediate_logic(
             test_programs, test_symbol_names, test_symbol_values,
             test_pauli_sums)
        self.assertAllEqual(util.from_tensor(expected_batch_programs),
                            util.from_tensor(test_batch_programs))
        self.assertAllEqual(expected_batch_symbol_names,
                            test_batch_symbol_names)
        self.assertAllClose(expected_batch_symbol_values,
                            test_batch_symbol_values,
                            atol=1e-6)
        self.assertAllEqual(expected_batch_pauli_sums, test_batch_pauli_sums)
        self.assertAllClose(expected_batch_mapper, test_batch_mapper, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()
