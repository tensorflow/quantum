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
"""Basic tests for the LinearCombinationDifferentiator"""
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import sympy
import cirq

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import linear_combination


def _simple_op_inputs():
    qubit = cirq.GridQubit(0, 0)
    symbol = 'alpha'
    circuit = cirq.Circuit(cirq.Y(qubit)**sympy.Symbol(symbol))
    op = cirq.X(qubit)
    value = 0.3
    n_samples = 3000000

    # Return inputs prepped for expectation ops.
    # circuit, symbol_names, values, ops, n_samples
    # along with expected feedforward expectation
    # and expected gradient.
    return (util.convert_to_tensor([circuit]), tf.convert_to_tensor([symbol]),
            tf.convert_to_tensor([[value]]), util.convert_to_tensor([[op]]),
            tf.convert_to_tensor([[n_samples]]),
            tf.convert_to_tensor([[np.sin(np.pi * value)]]),
            tf.convert_to_tensor([[np.pi * np.cos(np.pi * value)]]))


class LinearCombinationTest(tf.test.TestCase, parameterized.TestCase):
    """Test the LinearCombination based Differentiators."""

    def test_linear_combination_instantiate(self):
        """Test LinearCombinationDifferentiator type checking."""
        linear_combination.LinearCombination([1, 1], [1, 0])
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="weights must be"):
            linear_combination.LinearCombination("junk", [1, 0])
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="perturbations must be"):
            linear_combination.LinearCombination([1, 1], "junk")
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="weight in weights"):
            linear_combination.LinearCombination([1, "junk"], [1, 0])
        with self.assertRaisesRegex(
                TypeError, expected_regex="perturbation in perturbations"):
            linear_combination.LinearCombination([1, 1], [1, "junk"])
        with self.assertRaisesRegex(ValueError, expected_regex="length"):
            linear_combination.LinearCombination([1, 1, 1], [1, 0])
        with self.assertRaisesRegex(ValueError, expected_regex="unique"):
            linear_combination.LinearCombination([1, 1], [1, 1])
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="at least two"):
            linear_combination.LinearCombination([1], [1])

    def test_forward_instantiate(self):
        """Test ForwardDifference type checking."""
        linear_combination.ForwardDifference()
        linear_combination.ForwardDifference(1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive integer"):
            linear_combination.ForwardDifference(0.1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive integer"):
            linear_combination.ForwardDifference(-1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive integer"):
            linear_combination.ForwardDifference(0, 0.1)
        with self.assertRaisesRegex(ValueError, expected_regex="grid_spacing"):
            linear_combination.ForwardDifference(1, -0.1)
        with self.assertRaisesRegex(ValueError, expected_regex="grid_spacing"):
            linear_combination.ForwardDifference(1, 1j)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'order_coef_perturbs': [(1, (-1, 1), (
                        0, 1)), (2, (-3 / 2, 2, -1 / 2), (0, 1, 2))],
                    'grid_spacing': [0.1, 0.01, 0.5, 1, 0.05]
                })))
    def test_forward_coeffecients(self, order_coef_perturbs, grid_spacing):
        """Test that ForwardDifference produces the right coeffecients for
        common first and second order cases."""
        order = order_coef_perturbs[0]
        expected_std_coeffs = order_coef_perturbs[1]
        expected_perturbations = order_coef_perturbs[2]
        forward = linear_combination.ForwardDifference(order, grid_spacing)
        self.assertAllClose(
            np.array(expected_std_coeffs) / grid_spacing, forward.weights)
        self.assertAllClose(
            np.array(expected_perturbations) * grid_spacing,
            forward.perturbations)

    def test_central_instantiate(self):
        """Test CentralDifference type checking."""
        linear_combination.CentralDifference()
        linear_combination.CentralDifference(2, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive, even"):
            linear_combination.CentralDifference(0.1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive, even"):
            linear_combination.CentralDifference(-1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive, even"):
            linear_combination.CentralDifference(0, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive, even"):
            linear_combination.CentralDifference(1, 0.1)
        with self.assertRaisesRegex(ValueError, expected_regex="grid_spacing"):
            linear_combination.CentralDifference(2, -0.1)
        with self.assertRaisesRegex(ValueError, expected_regex="grid_spacing"):
            linear_combination.CentralDifference(2, 1j)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'order_coef_perturbs': [(2, (-1 / 2, 1 / 2), (-1, 1)),
                                            (4, (1 / 12, -8 / 12, 8 / 12,
                                                 -1 / 12), (-2, -1, 1, 2))],
                    'grid_spacing': [0.1, 0.01, 0.5, 1, 0.05]
                })))
    def test_central_coefficients(self, order_coef_perturbs, grid_spacing):
        """Test that CentralDifference produces the right coefficients for
        common first and second order cases."""
        order = order_coef_perturbs[0]
        expected_std_coeffs = order_coef_perturbs[1]
        expected_perturbations = order_coef_perturbs[2]
        forward = linear_combination.CentralDifference(order, grid_spacing)
        self.assertAllClose(
            np.array(expected_std_coeffs) / grid_spacing, forward.weights)
        self.assertAllClose(
            np.array(expected_perturbations) * grid_spacing,
            forward.perturbations)

    @parameterized.parameters([{
        'diff': linear_combination.ForwardDifference()
    }, {
        'diff': linear_combination.CentralDifference()
    }])
    def test_analytic_functional(self, diff):
        """Test that the differentiate_analytic function WORKS."""
        differentiable_op = diff.generate_differentiable_op(
            analytic_op=circuit_execution_ops.get_expectation_op())
        circuit, names, values, ops, _, true_f, true_g = _simple_op_inputs()
        with tf.GradientTape() as g:
            g.watch(values)
            res = differentiable_op(circuit, names, values, ops)

        # Just check that it computes without failing.
        self.assertAllClose(true_f, res, atol=1e-2, rtol=1e-2)
        self.assertAllClose(true_g,
                            g.gradient(res, values),
                            atol=1e-2,
                            rtol=1e-2)

    @parameterized.parameters([{
        'diff': linear_combination.ForwardDifference()
    }, {
        'diff': linear_combination.CentralDifference()
    }])
    def test_sampled_functional(self, diff):
        """Test that the differentiate_sampled function WORKS."""
        differentiable_op = diff.generate_differentiable_op(
            sampled_op=circuit_execution_ops.get_sampled_expectation_op())
        circuit, names, values, ops, n_samples, true_f, true_g = \
            _simple_op_inputs()
        with tf.GradientTape() as g:
            g.watch(values)
            res = differentiable_op(circuit, names, values, ops, n_samples)

        # Just check that it computes without failing.
        self.assertAllClose(true_f, res, atol=1e-1, rtol=1e-1)
        self.assertAllClose(true_g,
                            g.gradient(res, values),
                            atol=1e-1,
                            rtol=1e-1)

    def test_get_gradient_circuits(self):
        """Test that the correct objects are returned."""

        # Minimal linear combination.
        input_weights = [1.0, -0.5]
        input_perturbations = [1.0, -1.5]
        diff = linear_combination.LinearCombination(
            input_weights, input_perturbations)

        # Circuits to differentiate.
        symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
        q0 = cirq.GridQubit(0, 0)
        q1 = cirq.GridQubit(1, 2)
        input_programs = util.convert_to_tensor([
            cirq.Circuit(cirq.X(q0)**symbols[0],
                         cirq.ry(symbols[1])(q1)),
            cirq.Circuit(cirq.rx(symbols[0])(q0),
                         cirq.Y(q1)**symbols[1]),
        ])
        input_symbol_names = tf.constant([str(s) for s in symbols])
        input_symbol_values = tf.constant([[1.5, -2.7], [-0.3, 0.9]])

        # For each program in the input batch: LinearCombination creates a copy
        # of that program for each symbol in the batch; then for each symbol,
        # the program is copied for each non-zero perturbation; finally, a
        # single copy is added for the zero perturbation (no zero pert here).
        expected_batch_programs = tf.stack([
            [input_programs[0]] * 4, [input_programs[1]] * 4])
        expected_new_symbol_names = input_symbol_names

        # For each program in the input batch: first, the input symbol_values
        # for the program are tiled to the number of copies in the output.
        tiled_symbol_values = tf.stack([
            [input_symbol_values[0]] * 4, [input_symbol_values[1]] * 4])
        # Then we create the tensor of perturbations to apply to these symbol
        # values: for each symbol we tile out the non-zero perturbations at that
        # symbol's index, keeping all the other symbol perturbations at zero.
        # Perturbations are the same for each program.
        single_program_perturbations = tf.stack([
            [input_perturbations[0], 0.0],
            [input_perturbations[1], 0.0],
            [0.0, input_perturbations[0]],
            [0.0, input_perturbations[1]]])
        tiled_perturbations = tf.stack([
            single_program_perturbations, single_program_perturbations])
        # Finally we add the perturbations to the original symbol values.
        expected_batch_symbol_values = tiled_symbol_values + tiled_perturbations

        # The map for LinearCombination is the same for every program.
        individual_batch_mapper = tf.stack([
            [input_weights[0], input_weights[1], 0.0, 0.0],
            [0.0, 0.0, input_weights[0], input_weights[1]]
        ])
        expected_batch_mapper = tf.stack([
            individual_batch_mapper, individual_batch_mapper])

        (
            test_batch_programs, test_new_symbol_names,
            test_batch_symbol_values, test_batch_mapper
        ) = diff.get_gradient_circuits(
             input_programs, input_symbol_names, input_symbol_values)
        self.assertAllEqual(expected_batch_programs, test_batch_programs)
        self.assertAllEqual(expected_new_symbol_names,
                            test_new_symbol_names)
        self.assertAllClose(expected_batch_symbol_values,
                            test_batch_symbol_values,
                            atol=1e-6)
        self.assertAllClose(expected_batch_mapper, test_batch_mapper, atol=1e-6)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'differentiator': [
                        linear_combination.ForwardDifference(),
                        linear_combination.CentralDifference()
                    ],
                    'n_qubits': [5],
                    'n_programs': [3],
                    'n_ops': [3],
                    'symbol_names': [['a', 'b']]
                })))
    def test_gradient_circuits_grad_comparison(self, differentiator, n_qubits, n_programs, n_ops, symbol_names):
        """Test that analytic gradient agrees with the one from grad circuits"""
        # Get random circuits to check.
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                cirq.GridQubit.rect(1, n_qubits), symbol_names, n_programs)
        psums = [
            util.random_pauli_sums(qubits, 1, n_ops) for _ in circuit_batch
        ]

        # Convert to tensors.
        symbol_names_array = np.array(symbol_names)
        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch],
            dtype=np.float32)
        symbol_names_tensor = tf.convert_to_tensor(symbol_names_array)
        symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
        programs = util.convert_to_tensor(circuit_batch)
        ops_tensor = util.convert_to_tensor(psums)

        # Get gradients using expectations of gradient circuits.
        (
            batch_programs, new_symbol_names, batch_symbol_values,
            batch_mapper
        ) = differentiator.get_gradient_circuits(
            programs, symbol_names_tensor, symbol_values_tensor)
        analytic_op = circuit_execution_ops.get_expectation_op()
        batch_pauli_sums = tf.tile(tf.expand_dims(ops_tensor, 1),
                                   [1, tf.shape(batch_mapper)[2], 1])
        n_batch_programs = tf.reduce_prod(tf.shape(batch_programs))
        n_symbols = len(symbol_names)
        batch_expectations = analytic_op(
            tf.reshape(batch_programs, [n_batch_programs]),
            new_symbol_names,
            tf.reshape(batch_symbol_values, [n_batch_programs, n_symbols]),
            tf.reshape(batch_pauli_sums, [n_batch_programs, n_ops]))
        batch_expectations = tf.reshape(
            batch_expectations, tf.shape(batch_pauli_sums))
        grad_manual = tf.reduce_sum(
            tf.einsum('ikm,imp->ikp', batch_mapper, batch_expectations), -1)

        # Get gradients using autodiff.
        differentiator.refresh()
        differentiable_op = differentiator.generate_differentiable_op(
            analytic_op=analytic_op)
        with tf.GradientTape() as g:
            g.watch(symbol_values_tensor)
            exact_outputs = differentiable_op(
                programs, symbol_names_tensor, symbol_values_tensor, ops_tensor)
        grad_auto = g.gradient(exact_outputs, symbol_values_tensor)
        self.assertAllClose(grad_manual, grad_auto)


if __name__ == "__main__":
    tf.test.main()
