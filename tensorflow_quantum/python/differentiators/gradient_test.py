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
"""Testing for gradient calculation consistency in TFQ."""
import copy

import numpy as np
import sympy
import tensorflow as tf
from absl.testing import parameterized

import cirq
from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import adjoint
from tensorflow_quantum.python.differentiators import linear_combination
from tensorflow_quantum.python.differentiators import parameter_shift
from tensorflow_quantum.core.ops import circuit_execution_ops, batch_util

ANALYTIC_DIFFS = [
    linear_combination.ForwardDifference(grid_spacing=0.0001),
    linear_combination.ForwardDifference(error_order=2, grid_spacing=0.0001),
    linear_combination.CentralDifference(grid_spacing=0.0001),
    linear_combination.CentralDifference(error_order=4, grid_spacing=0.0001),
    parameter_shift.ParameterShift(),
]

SAMPLED_DIFFS = [
    linear_combination.ForwardDifference(grid_spacing=0.05),
    linear_combination.CentralDifference(grid_spacing=0.05),
    parameter_shift.ParameterShift(),
]

SAMPLED_DIFFS_TOLS = [0.5, 0.5, 0.2]

ANALYTIC_OPS = [
    circuit_execution_ops.get_expectation_op(cirq.sim.Simulator()),  # WF
    circuit_execution_ops.get_expectation_op()  # C++
]

SAMPLED_OPS = [
    circuit_execution_ops.get_sampled_expectation_op(
        cirq.sim.Simulator()),  # WF
    circuit_execution_ops.get_sampled_expectation_op()  # C++
]


def _cirq_simple_finite_difference(circuit_batch,
                                   resolvers,
                                   symbol_names,
                                   op_batch,
                                   grid_spacing=0.0001):
    """A simple finite difference code that calculates the gradient of a
    batch of circuits using cirq."""
    simulator = cirq.sim.Simulator()

    init_vals = batch_util.batch_calculate_expectation(circuit_batch, resolvers,
                                                       op_batch, simulator)
    grad_circuits = []
    grad_resolvers = []
    grad_pauli_sums = []
    for this_program, this_pauli_sums, this_resolver in \
        zip(circuit_batch, op_batch, resolvers):
        for symbol in symbol_names:
            perturbed_resolver = copy.deepcopy(this_resolver)
            perturbed_resolver.param_dict[symbol] += grid_spacing
            grad_circuits.append(this_program)
            grad_pauli_sums.append(this_pauli_sums)
            grad_resolvers.append(perturbed_resolver)

    # shape: [n_programs * len(symbol_names), n_pauli_sums]
    results = np.array(
        batch_util.batch_calculate_expectation(circuits=grad_circuits,
                                               param_resolvers=grad_resolvers,
                                               ops=grad_pauli_sums,
                                               simulator=simulator))

    # shape: [n_pauli_sums, n_programs, len(symbol_names)]
    gradient_generator = results.transpose().reshape(
        (len(op_batch[0]), len(circuit_batch), len(symbol_names)))

    # shape: [n_pauli_sums, n_programs, len(symbol_names)]
    forward_pass_vals = np.transpose(
        np.vstack([np.expand_dims(init_vals, axis=0)] * len(symbol_names)),
        (2, 1, 0))

    return np.sum(1 / grid_spacing * (gradient_generator - forward_pass_vals),
                  axis=0)


class AnalyticGradientCorrectnessTest(tf.test.TestCase, parameterized.TestCase):
    """Test correctness of the differentiators to reference cirq algorithm."""

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'differentiator': ANALYTIC_DIFFS,
                'op': ANALYTIC_OPS
            })) + [{
                'differentiator': adjoint.Adjoint(),
                'op': circuit_execution_ops.get_expectation_op()
            }])
    def test_backprop(self, differentiator, op):
        """Test that gradients are correctly backpropagated through a quantum
        circuit via comparison to analytical results.
        """
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(analytic_op=op)

        def exact_grad(theta):
            new_theta = 2 * np.pi * theta
            return -2 * np.pi * np.sin(new_theta) * np.exp(np.cos(new_theta))

        bit = cirq.GridQubit(0, 0)
        circuits = util.convert_to_tensor(
            [cirq.Circuit(cirq.X(bit)**sympy.Symbol('rx')) for _ in range(2)])
        pstring = util.convert_to_tensor([[
            cirq.PauliSum.from_pauli_strings([cirq.PauliString({bit: cirq.Z})])
        ] for _ in circuits])
        base_rot_angles = tf.constant([[0.25], [0.125]])
        with tf.GradientTape() as g:
            g.watch(base_rot_angles)
            input_angles = 2 * base_rot_angles
            exp_res = tf.exp(
                op(circuits, tf.convert_to_tensor(['rx']), input_angles,
                   pstring))

        grad = g.gradient(exp_res, base_rot_angles)
        exact = [[exact_grad(0.25)], [exact_grad(0.125)]]

        # will this be too tight? time will tell.
        self.assertAllClose(exact, grad.numpy(), rtol=0.01, atol=0.01)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'differentiator': ANALYTIC_DIFFS,
                    'op': ANALYTIC_OPS,
                    'n_qubits': [5],
                    'n_programs': [3],
                    'n_ops': [3],
                    'symbol_names': [['a', 'b']]
                })) + [{
                    'differentiator': adjoint.Adjoint(),
                    'op': circuit_execution_ops.get_expectation_op(),
                    'n_qubits': 10,
                    'n_programs': 5,
                    'n_ops': 3,
                    'symbol_names': ['a', 'b']
                }])
    def test_gradients_vs_cirq_finite_difference(self, differentiator, op,
                                                 n_qubits, n_programs, n_ops,
                                                 symbol_names):
        """Compare TFQ differentiators to fine-grained noiseless cirq finite
        differencing.
        """
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(analytic_op=op)

        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                cirq.GridQubit.rect(1, n_qubits), symbol_names, n_programs)

        psums = [
            util.random_pauli_sums(qubits, 1, n_ops) for _ in circuit_batch
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch],
            dtype=np.float32)

        # calculate tfq gradient
        symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
        programs = util.convert_to_tensor(circuit_batch)
        ops = util.convert_to_tensor(psums)
        with tf.GradientTape() as g:
            g.watch(symbol_values_tensor)
            expectations = op(programs, tf.convert_to_tensor(symbol_names),
                              symbol_values_tensor, ops)
        tfq_grads = g.gradient(expectations, symbol_values_tensor)

        # calculate gradients in cirq using a very simple forward differencing
        # scheme
        cirq_grads = _cirq_simple_finite_difference(circuit_batch,
                                                    resolver_batch,
                                                    symbol_names, psums)

        # will this be too tight? time will tell.
        self.assertAllClose(cirq_grads, tfq_grads, rtol=2e-2, atol=2e-2)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'differentiator': ANALYTIC_DIFFS,
                'op': ANALYTIC_OPS,
            })) + [{
                'differentiator': adjoint.Adjoint(),
                'op': circuit_execution_ops.get_expectation_op(),
            }])
    def test_analytic_value_with_simple_circuit(self, differentiator, op):
        """Test the value of differentiator with simple circuit."""
        # Get an expectation op, with this differentiator attached.
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(analytic_op=op)
        qubit = cirq.GridQubit(0, 0)
        circuit = util.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubit)**sympy.Symbol('alpha'))])
        psums = util.convert_to_tensor([[cirq.Z(qubit)]])
        symbol_values_array = np.array([[0.123]], dtype=np.float32)
        # Calculate tfq gradient.
        symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
        with tf.GradientTape() as g:
            g.watch(symbol_values_tensor)
            expectations = op(circuit, tf.convert_to_tensor(['alpha']),
                              symbol_values_tensor, psums)
        grads = g.gradient(expectations, symbol_values_tensor)
        ground_truth_grads = np.array([[-1.1839752]])
        self.assertAllClose(ground_truth_grads, grads, rtol=1e-2, atol=1e-2)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'differentiator': ANALYTIC_DIFFS,
                'op': ANALYTIC_OPS,
            })) + [{
                'differentiator': adjoint.Adjoint(),
                'op': circuit_execution_ops.get_expectation_op(),
            }])
    def test_empty_circuit_grad(self, differentiator, op):
        """Test that providing no circuits will fail gracefully."""
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(analytic_op=op)
        circuit = tf.convert_to_tensor([], dtype=tf.string)
        psums = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)

        # Calculate tfq gradient.
        symbol_values_tensor = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        symbol_names_tensor = tf.convert_to_tensor([], dtype=tf.string)
        with tf.GradientTape() as g:
            g.watch(symbol_values_tensor)
            expectations = op(circuit, symbol_names_tensor,
                              symbol_values_tensor, psums)
        grads = g.gradient(expectations, symbol_values_tensor)
        self.assertShapeEqual(grads.numpy(),
                              tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32))


class SampledGradientCorrectnessTest(tf.test.TestCase, parameterized.TestCase):
    """Test approximate correctness to analytical methods."""

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'differentiator': SAMPLED_DIFFS,
                    'op': SAMPLED_OPS,
                    'num_samples': [10000]
                })))
    def test_sampled_value_with_simple_circuit(self, differentiator, op,
                                               num_samples):
        """Test the value of sampled differentiator with simple circuit."""
        # Get an expectation op, with this differentiator attached.
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(sampled_op=op)
        qubit = cirq.GridQubit(0, 0)
        circuit = util.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubit)**sympy.Symbol('alpha'))])
        psums = util.convert_to_tensor([[cirq.Z(qubit)]])
        symbol_values_array = np.array([[0.123]], dtype=np.float32)
        # Calculate tfq gradient.
        symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
        with tf.GradientTape() as g:
            g.watch(symbol_values_tensor)
            expectations = op(circuit, tf.convert_to_tensor(['alpha']),
                              symbol_values_tensor, psums,
                              tf.convert_to_tensor([[num_samples]]))
        grads = g.gradient(expectations, symbol_values_tensor)
        ground_truth_grads = np.array([[-1.1839752]])
        self.assertAllClose(ground_truth_grads, grads, rtol=0.2, atol=0.2)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'diff_and_tol': zip(SAMPLED_DIFFS, SAMPLED_DIFFS_TOLS),
                    'op': SAMPLED_OPS,
                    'n_qubits': [3],
                    'n_programs': [5],
                    'n_ops': [2],
                    'symbol_names': [['a', 'b']],
                    'num_samples': [30000]
                })))
    def test_approx_equality_shallow(self, diff_and_tol, op, n_qubits,
                                     symbol_names, n_ops, n_programs,
                                     num_samples):
        """Test small circuits with limited depth."""
        differentiator, tol = diff_and_tol
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(sampled_op=op)

        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                cirq.GridQubit.rect(1, n_qubits), symbol_names, n_programs)

        # Prepare random pauli sums and add initial superposition gates.
        psums = []
        for i in range(len(circuit_batch)):
            psums.append(util.random_pauli_sums(qubits, 1, n_ops))
            circuit_batch[i] = cirq.Circuit(
                cirq.H.on_each(qubits)) + circuit_batch[i]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch],
            dtype=np.float32)

        # calculate tfq gradient
        symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
        programs = util.convert_to_tensor(circuit_batch)
        ops = util.convert_to_tensor(psums)
        with tf.GradientTape() as g:
            g.watch(symbol_values_tensor)
            expectations = op(
                programs, tf.convert_to_tensor(symbol_names),
                symbol_values_tensor, ops,
                tf.convert_to_tensor([[num_samples] * n_ops] * n_programs))
        tfq_grads = g.gradient(expectations, symbol_values_tensor)

        # calculate gradients in cirq using a very simple forward differencing
        # scheme
        cirq_grads = _cirq_simple_finite_difference(circuit_batch,
                                                    resolver_batch,
                                                    symbol_names, psums)

        self.assertAllClose(cirq_grads, tfq_grads, rtol=tol, atol=tol)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'differentiator': SAMPLED_DIFFS,
                'op': SAMPLED_OPS,
            })))
    def test_empty_circuit_sampled_grad(self, differentiator, op):
        """Test that providing no circuits will fail gracefully."""
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(sampled_op=op)
        circuit = tf.convert_to_tensor([], dtype=tf.string)
        psums = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)

        # Calculate tfq gradient.
        symbol_values_tensor = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        symbol_names_tensor = tf.convert_to_tensor([], dtype=tf.string)
        n_samples_tensor = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.int32)
        with tf.GradientTape() as g:
            g.watch(symbol_values_tensor)
            expectations = op(circuit, symbol_names_tensor,
                              symbol_values_tensor, psums, n_samples_tensor)
        grads = g.gradient(expectations, symbol_values_tensor)
        self.assertShapeEqual(grads.numpy(),
                              tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32))


if __name__ == '__main__':
    tf.test.main()
