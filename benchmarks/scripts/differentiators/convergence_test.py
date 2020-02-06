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
"""Testing for SGDifferentiator convergence & calculation consistency in TFQ."""
import copy
import time

import numpy as np
import tensorflow as tf
from absl.testing import parameterized

import cirq
from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import stochastic_differentiator
from tensorflow_quantum.core.ops import tfq_simulate_ops, batch_util

# DISCLAIMER: Environment : Intel(R) Xeon(R) W-2135 CPU @ 3.70GHz, 12 cores.
# The overall tests take around 1 hours.
DIFFS_NUM_RUNS = [
    # The tests without sampling cost Hamiltonian take 1.5 hours.
    # Case 1 : ParameterShift ~ 0.04 sec/shot
    (stochastic_differentiator.SGDifferentiator(stochastic_coordinate=False,
                                                stochastic_generator=False,
                                                stochastic_cost=False), 1),
    # Case 2 : coordinate ~ 42 sec (0.04 sec/shot)
    (stochastic_differentiator.SGDifferentiator(stochastic_coordinate=True,
                                                stochastic_generator=False,
                                                stochastic_cost=False), 1100),
    # Case 3 : generator ~ 350 sec (0.023 sec/shot)
    (stochastic_differentiator.SGDifferentiator(stochastic_coordinate=False,
                                                stochastic_generator=True,
                                                stochastic_cost=False), 15000),
    # Case 4 : coordinate + generator ~ 400 sec ~ (0.020 sec/shot)
    (stochastic_differentiator.SGDifferentiator(stochastic_coordinate=True,
                                                stochastic_generator=True,
                                                stochastic_cost=False), 20000),
    # The tests with sampling cost Hamiltonian takes around 3 hours
    # Case 5 : cost ~ 35 sec (0.15 sec/shot)
    (stochastic_differentiator.SGDifferentiator(stochastic_coordinate=False,
                                                stochastic_generator=False,
                                                stochastic_cost=True), 250),
    # Case 6 : cost + coordinate ~ 160 sec (0.15 sec/shot)
    (stochastic_differentiator.SGDifferentiator(stochastic_coordinate=True,
                                                stochastic_generator=False,
                                                stochastic_cost=True), 1200),
    # Case 7 : cost + generator ~ 320 sec (0.13 sec/shot)
    (stochastic_differentiator.SGDifferentiator(stochastic_coordinate=False,
                                                stochastic_generator=True,
                                                stochastic_cost=True), 2500),
    # Case 8 : All ~ 2400 sec ~ 40 m (0.12 sec/shot)
    #  Increase error margin due to numerical stability of summing up gradients
    (stochastic_differentiator.SGDifferentiator(stochastic_coordinate=True,
                                                stochastic_generator=True,
                                                stochastic_cost=True), 20000),
]


# TODO(jaeyoo): aggregate identical _cirq_simple_finite_difference functions
#  in different tests into one python file and import it.
def _cirq_simple_finite_difference(circuit_batch,
                                   resolvers,
                                   symbol_names,
                                   op_batch,
                                   grid_spacing=0.0001):
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


class StochasticGradientConvergenceTest(tf.test.TestCase,
                                        parameterized.TestCase):

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'differentiator_num_runs': DIFFS_NUM_RUNS,
                    'n_qubits': [3],
                    'n_programs': [3],
                    'n_ops': [3],
                    'symbol_names': [['a', 'b']],
                    'eps': [0.1]
                })))
    def test_gradients_vs_cirq_finite_difference(self, differentiator_num_runs,
                                                 n_qubits, n_programs, n_ops,
                                                 symbol_names, eps):
        """Convergence tests on SGDifferentiator variants."""

        # TODO(trevormccrt): remove this once I build the user-facing op
        #  interface
        differentiator, num_runs = differentiator_num_runs
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(
            analytic_op=tfq_simulate_ops.tfq_simulate_expectation)

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

        def _get_gradient():
            with tf.GradientTape() as g:
                g.watch(symbol_values_tensor)
                expectations = op(programs, symbol_names, symbol_values_tensor,
                                  ops)
            return tf.cast(g.gradient(expectations, symbol_values_tensor),
                           dtype=tf.float64)

        # warm-up & initialize tfq_grads.
        grads_sum = _get_gradient()
        tfq_grads = grads_sum

        # calculate gradients in cirq using a very simple forward differencing
        # scheme
        cirq_grads = _cirq_simple_finite_difference(circuit_batch,
                                                    resolver_batch,
                                                    symbol_names, psums)
        cnt = 1
        # Since self.assertAllClose() has more strict atol than that of
        # np.allclose(), it is required to set smaller value to np.allclose()
        total_time = 0
        while cnt < num_runs and (not np.allclose(
                tfq_grads, cirq_grads, atol=eps * 0.9)):
            cnt = cnt + 1
            s = time.time()
            grads_sum = grads_sum + _get_gradient()
            total_time += time.time() - s
            tfq_grads = grads_sum / cnt

        self.assertAllClose(cirq_grads, tfq_grads, atol=eps)
        print('Passed: count {}, total_time {} ({}sec/shot)'.format(
            cnt, total_time, total_time / cnt))


if __name__ == '__main__':
    tf.test.main()
