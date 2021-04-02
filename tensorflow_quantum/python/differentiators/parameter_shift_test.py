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

    def test_get_gradient_circuits(self):
        """Test that the correct objects are returned."""

        diff = parameter_shift.ParameterShift()

        # Circuits to differentiate.
        symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
        q0 = cirq.GridQubit(0, 0)
        q1 = cirq.GridQubit(1, 2)
        input_programs = util.convert_to_tensor([
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.Y(q0)**symbols[0],
                         cirq.ry(symbols[1])(q1)),
            cirq.Circuit(cirq.Y(q1)**symbols[1]),
        ])
        input_symbol_names = tf.constant([str(s) for s in symbols])
        input_symbol_values = tf.constant([[1.5, -2.7], [-0.3, 0.9]])

        # First, for each symbol `s`, check how many times `s` appears in each
        # program `p`, `n_ps`. Let `n_param_gates` be the maximum of `n_ps` over
        # all symbols and programs. Then, the shape of `batch_programs` will be
        # [n_programs, n_symbols * n_param_gates * n_shifts], where `n_shifts`
        # is 2 because we decompose into gates with 2 eigenvalues. For row index
        # `p` we have for column indices between `i * n_param_gates * n_shifts`
        # and `(i + 1) * n_param_gates * n_shifts`, the first `n_pi * 2`
        # programs are parameter shifted versions of `input_programs[p]` and the
        # remaining programs are empty.
        # Here, `n_param_gates` is 2.
        impurity_symbol_name = "_param_shift"
        impurity_symbol = sympy.Symbol(impurity_symbol_name)
        expected_batch_programs_0 = util.convert_to_tensor([
            cirq.Circuit(cirq.X(q0)**impurity_symbol,
                         cirq.Y(q0)**symbols[0],
                         cirq.ry(symbols[1])(q1)),
            cirq.Circuit(cirq.X(q0)**impurity_symbol,
                         cirq.Y(q0)**symbols[0],
                         cirq.ry(symbols[1])(q1)),
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.Y(q0)**impurity_symbol,
                         cirq.ry(symbols[1])(q1)),
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.Y(q0)**impurity_symbol,
                         cirq.ry(symbols[1])(q1)),
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.Y(q0)**symbols[0],
                         cirq.ry(impurity_symbol)(q1)),
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.Y(q0)**symbols[0],
                         cirq.ry(impurity_symbol)(q1)),
            cirq.Circuit(),
            cirq.Circuit()
        ])
        expected_batch_programs_1 = util.convert_to_tensor([
            cirq.Circuit(),
            cirq.Circuit(),
            cirq.Circuit(),
            cirq.Circuit(),
            cirq.Circuit(cirq.Y(q1)**impurity_symbol),
            cirq.Circuit(cirq.Y(q1)**impurity_symbol),
            cirq.Circuit(),
            cirq.Circuit()
          ])
        expected_batch_programs = tf.stack([expected_batch_programs_0,
                                            expected_batch_programs_1])

        # The new symbols are the old ones, with an extra used for shifting.
        expected_new_symbol_names = tf.concat([
            input_symbol_names, tf.constant([impurity_symbol_name])], 0)

        # The batch symbol values are the input symbol values, tiled and with
        # shifted values appended. Locations that have empty programs should
        # also have zero for the shift.
        # The shifted values are the original value plus 1/2 divided by the
        # `exponent_scalar` of the gate.
        expected_batch_symbol_values = tf.constant(
            [[[1.5, -2.7, 1.5 + 0.5],
              [1.5, -2.7, 1.5 - 0.5],
              [1.5, -2.7, 1.5 + 0.5],
              [1.5, -2.7, 1.5 - 0.5],
              [1.5, -2.7, -2.7 + np.pi/2],
              [1.5, -2.7, -2.7 - np.pi/2],
              [1.5, -2.7, -2.7],
              [1.5, -2.7, -2.7]],
             [[-0.3, 0.9, -0.3],
              [-0.3, 0.9, -0.3],
              [-0.3, 0.9, -0.3],
              [-0.3, 0.9, -0.3],
              [-0.3, 0.9, 0.9 + 0.5],
              [-0.3, 0.9, 0.9 - 0.5],
              [-0.3, 0.9, 0.9],
              [-0.3, 0.9, 0.9]]])

        (test_batch_programs, test_new_symbol_names, test_batch_symbol_values,
         test_batch_mapper) = diff.get_gradient_circuits(
             input_programs, input_symbol_names, input_symbol_values)
        for i in range(tf.shape(input_programs)[0]):
          self.assertAllEqual(util.from_tensor(expected_batch_programs[i]),
                              util.from_tensor(test_batch_programs[i]))
        for i in range(tf.shape(expected_batch_symbol_values)[0]):
          for j in range(tf.shape(expected_batch_symbol_values[i])[0]):
            print(f"{i,j}:")
            print(expected_batch_symbol_values[i][j])
            print(test_batch_symbol_values[i][j])
            print()
        self.assertAllEqual(expected_new_symbol_names, test_new_symbol_names)
        self.assertAllClose(expected_batch_symbol_values,
                            test_batch_symbol_values,
                            atol=1e-6)
#        self.assertAllClose(expected_batch_mapper, test_batch_mapper, atol=1e-6)

    # @parameterized.parameters(
    #     list(
    #         util.kwargs_cartesian_product(
    #             **{
    #                 'differentiator': [
    #                     parameter_shift.ParameterShift(),
    #                 ],
    #                 'n_qubits': [5],
    #                 'n_programs': [3],
    #                 'n_ops': [3],
    #                 'symbol_names': [['a', 'b']]
    #             })))
    # def test_gradient_circuits_grad_comparison(self, differentiator, n_qubits,
    #                                            n_programs, n_ops, symbol_names):
    #     """Test that analytic gradient agrees with the one from grad circuits"""
    #     # Get random circuits to check.
    #     qubits = cirq.GridQubit.rect(1, n_qubits)
    #     circuit_batch, resolver_batch = \
    #         util.random_symbol_circuit_resolver_batch(
    #             cirq.GridQubit.rect(1, n_qubits), symbol_names, n_programs)
    #     psums = [
    #         util.random_pauli_sums(qubits, 1, n_ops) for _ in circuit_batch
    #     ]

    #     # Convert to tensors.
    #     symbol_names_array = np.array(symbol_names)
    #     symbol_values_array = np.array(
    #         [[resolver[symbol]
    #           for symbol in symbol_names]
    #          for resolver in resolver_batch],
    #         dtype=np.float32)
    #     symbol_names_tensor = tf.convert_to_tensor(symbol_names_array)
    #     symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
    #     programs = util.convert_to_tensor(circuit_batch)
    #     ops_tensor = util.convert_to_tensor(psums)

    #     # Get gradients using expectations of gradient circuits.
    #     (batch_programs, new_symbol_names, batch_symbol_values,
    #      batch_mapper) = differentiator.get_gradient_circuits(
    #          programs, symbol_names_tensor, symbol_values_tensor)
    #     analytic_op = circuit_execution_ops.get_expectation_op()
    #     batch_pauli_sums = tf.tile(tf.expand_dims(ops_tensor, 1),
    #                                [1, tf.shape(batch_mapper)[2], 1])
    #     n_batch_programs = tf.reduce_prod(tf.shape(batch_programs))
    #     n_symbols = len(symbol_names)
    #     batch_expectations = analytic_op(
    #         tf.reshape(batch_programs, [n_batch_programs]), new_symbol_names,
    #         tf.reshape(batch_symbol_values, [n_batch_programs, n_symbols]),
    #         tf.reshape(batch_pauli_sums, [n_batch_programs, n_ops]))
    #     batch_expectations = tf.reshape(batch_expectations,
    #                                     tf.shape(batch_pauli_sums))
    #     grad_manual = tf.reduce_sum(
    #         tf.einsum('ikm,imp->ikp', batch_mapper, batch_expectations), -1)

    #     # Get gradients using autodiff.
    #     differentiator.refresh()
    #     differentiable_op = differentiator.generate_differentiable_op(
    #         analytic_op=analytic_op)
    #     with tf.GradientTape() as g:
    #         g.watch(symbol_values_tensor)
    #         exact_outputs = differentiable_op(programs, symbol_names_tensor,
    #                                           symbol_values_tensor, ops_tensor)
    #     grad_auto = g.gradient(exact_outputs, symbol_values_tensor)
    #     self.assertAllClose(grad_manual, grad_auto)


if __name__ == "__main__":
    tf.test.main()
