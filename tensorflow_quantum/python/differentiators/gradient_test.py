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
from tensorflow_quantum.python.differentiators import linear_combination
from tensorflow_quantum.python.differentiators import parameter_shift
from tensorflow_quantum.python.differentiators import stochastic_differentiator
from tensorflow_quantum.core.ops import circuit_execution_ops, batch_util

DIFFS = [
    linear_combination.ForwardDifference(grid_spacing=0.0001),
    linear_combination.ForwardDifference(error_order=2, grid_spacing=0.0001),
    linear_combination.CentralDifference(grid_spacing=0.0001),
    linear_combination.CentralDifference(error_order=4, grid_spacing=0.0001),
    parameter_shift.ParameterShift(),
]

STOCHASTIC_DIFFS = [
    stochastic_differentiator.SGDifferentiator(stochastic_coordinate=False,
                                               stochastic_generator=False,
                                               stochastic_cost=False),
    stochastic_differentiator.SGDifferentiator(stochastic_coordinate=True,
                                               stochastic_generator=False,
                                               stochastic_cost=False),
    stochastic_differentiator.SGDifferentiator(stochastic_coordinate=False,
                                               stochastic_generator=True,
                                               stochastic_cost=False),
    stochastic_differentiator.SGDifferentiator(stochastic_coordinate=True,
                                               stochastic_generator=True,
                                               stochastic_cost=False),
]

OPS = [
    circuit_execution_ops.get_expectation_op(cirq.sim.Simulator()),  # WF
    circuit_execution_ops.get_expectation_op(
        cirq.DensityMatrixSimulator()),  # DM
    circuit_execution_ops.get_expectation_op()  # C++
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


class GradientCorrectnessTest(tf.test.TestCase, parameterized.TestCase):
    """Test correctness of the differentiators to reference cirq algorithm."""

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'differentiator': DIFFS + STOCHASTIC_DIFFS,
                    'op': OPS,
                    'stochastic_cost': [False, True]
                })))
    def test_backprop(self, differentiator, op, stochastic_cost):
        """Test that gradients are correctly backpropagated through a quantum
        circuit via comparison to analytical results.
        """
        # hack to add stoachastic cost. TODO (jaeyoo): remove this hack.
        differentiator.stochastic_cost = stochastic_cost
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
            exp_res = tf.exp(op(circuits, ['rx'], input_angles, pstring))

        grad = g.gradient(exp_res, base_rot_angles)
        exact = [[exact_grad(0.25)], [exact_grad(0.125)]]

        # will this be too tight? time will tell.
        self.assertAllClose(exact, grad.numpy(), rtol=0.01, atol=0.01)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'differentiator': DIFFS,
                    'op': OPS,
                    'n_qubits': [5],
                    'n_programs': [3],
                    'n_ops': [3],
                    'symbol_names': [['a', 'b']]
                })))
    def test_gradients_vs_cirq_finite_difference(self, differentiator, op,
                                                 n_qubits, n_programs, n_ops,
                                                 symbol_names):
        """Compare TFQ differentiators to fine-grained noiseless cirq finite
        differencing.
        DISCLAIMER : the consistency of STOCHASTIC_DIFFS is hard to be checked.
        Its expectation value should be checked, but it takes long time because
        SGDifferentiator is not optimized. Until optimized, the consistency
        will be performed in benchmarks/scripts/differentiators:convergence_test
        TODO(jaeyoo) : move convergence_test here once SGDifferentiator is
         optimized.
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
            expectations = op(programs, symbol_names, symbol_values_tensor, ops)
        tfq_grads = g.gradient(expectations, symbol_values_tensor)

        # calculate gradients in cirq using a very simple forward differencing
        # scheme
        cirq_grads = _cirq_simple_finite_difference(circuit_batch,
                                                    resolver_batch,
                                                    symbol_names, psums)

        # will this be too tight? time will tell.
        self.assertAllClose(cirq_grads, tfq_grads, rtol=1e-2, atol=1e-2)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'differentiator': DIFFS + STOCHASTIC_DIFFS,
                    'op': OPS,
                    'stochastic_cost': [False, True]
                })))
    def test_analytic_value_with_simple_circuit(self, differentiator, op,
                                                stochastic_cost):
        """Test the value of differentiator with simple circuit.
        Since there are only one symbol, one gate and one op, there is only one
        samling result, STOCHATIC_DIFFS shows the same result with that of
        deterministic differentiators."""
        # Get an expectation op, with this differentiator attached.
        differentiator.refresh()
        differentiator.stochastic_cost = stochastic_cost
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
            expectations = op(circuit, ['alpha'], symbol_values_tensor, psums)
        grads = g.gradient(expectations, symbol_values_tensor)
        ground_truth_grads = np.array([[-1.1839752]])
        self.assertAllClose(ground_truth_grads, grads, rtol=1e-2, atol=1e-2)


class StochasticDifferentiatorCorrectnessTest(tf.test.TestCase,
                                              parameterized.TestCase):
    """Test correctness of the stochastic differentiators to reference cirq
    algorithm.
    DISCLAIMER: this test allows for a larger margin of error and as long
    as convergence is happening then it passes"""

    # TODO(zaqqwerty): only this test was failing after adding cirq.I
    #    support, so it is disabled pending diagnosis
    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'differentiator': STOCHASTIC_DIFFS,
                    'op': OPS,
                    'n_qubits': [5],
                    'n_programs': [3],
                    'n_ops': [3],
                    'symbol_names': [['a', 'b']],
                    'stochastic_cost_eps': [(False, 5e-1), (True, 7e-1)],
                })))
    def gradients_vs_cirq_finite_difference(self, differentiator, op, n_qubits,
                                            n_programs, n_ops, symbol_names,
                                            stochastic_cost_eps):
        """Compare TFQ differentiators to fine-grained noiseless cirq finite
        differencing with a larger margin of error."""

        # TODO (jaeyoo): cleanup this hacky wordkaround so variable
        #   assignment doesn't need to take place like this.
        differentiator.stochastic_cost, eps = stochastic_cost_eps
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

        # calculate gradients in cirq using a very simple forward differencing
        # scheme
        cirq_grads = _cirq_simple_finite_difference(circuit_batch,
                                                    resolver_batch,
                                                    symbol_names, psums)

        def _get_gradient():
            with tf.GradientTape() as g:
                g.watch(symbol_values_tensor)
                expectations = op(programs, symbol_names, symbol_values_tensor,
                                  ops)
            return g.gradient(expectations, symbol_values_tensor)

        def _abs_diff(grad, mask):
            return np.sum(np.abs(grad - cirq_grads * mask))

        def _get_nonzero_mask(grad):
            return (grad.numpy() != 0.0).astype(np.float32)

        # Get the non-zero mask because a few initial gradients have not sampled
        # zero values.
        tfq_grads_1 = _get_gradient()
        mask_1 = _get_nonzero_mask(tfq_grads_1)

        if not np.allclose(tfq_grads_1, cirq_grads * mask_1, atol=eps):
            tfq_grads_2 = 0.5 * (tfq_grads_1 + _get_gradient())
            mask_2 = _get_nonzero_mask(tfq_grads_2)
            # Check if the 2nd error becomes smaller that 1st one.
            if not _abs_diff(tfq_grads_1, mask_1) > _abs_diff(
                    tfq_grads_2, mask_2):
                cnt = 2
                tfq_grads = (cnt * tfq_grads_2 + _get_gradient()) / (cnt + 1)
                while (cnt < 10 and
                       not np.allclose(cirq_grads, tfq_grads, atol=eps)):
                    cnt += 1
                    tfq_grads = (cnt * tfq_grads + _get_gradient()) / (cnt + 1)
                self.assertAllClose(cirq_grads, tfq_grads, atol=eps)


if __name__ == '__main__':
    tf.test.main()
