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
"""Test parallel Cirq simulations."""
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from scipy import stats
import cirq

from tensorflow_quantum.core.ops import batch_util
from tensorflow_quantum.python import util

BATCH_SIZE = 12
N_QUBITS = 5
PAULI_LENGTH = 3
SYMBOLS = ['alpha', 'beta', 'gamma']


def _get_mixed_batch(qubits, symbols, size):
    circuit1, resolver1 = util.random_circuit_resolver_batch(qubits, size // 2)
    circuit2, resolver2 = util.random_symbol_circuit_resolver_batch(
        qubits, symbols, size // 2)
    return circuit1 + circuit2, resolver1 + resolver2


def _pad_state(sim, state, n):
    if isinstance(sim, cirq.Simulator):
        state = state.final_state_vector
    if isinstance(sim, cirq.DensityMatrixSimulator):
        state = state.final_density_matrix
    return np.pad(state, (0, (1 << n) - state.shape[-1]),
                  'constant',
                  constant_values=-2)


def _expectation_helper(sim, circuit, params, op):
    if isinstance(sim, cirq.Simulator):
        state = sim.simulate(circuit,
                             params).final_state_vector.astype(np.complex128)
        return [
            op.expectation_from_state_vector(
                state,
                dict(
                    zip(sorted(circuit.all_qubits()),
                        (j for j in range(len(circuit.all_qubits())))))).real
        ]
    if isinstance(sim, cirq.DensityMatrixSimulator):
        state = sim.simulate(circuit, params).final_density_matrix
        return [
            sum(
                x._expectation_from_density_matrix_no_validation(
                    state,
                    dict(
                        zip(sorted(circuit.all_qubits()), (
                            j
                            for j in range(len(circuit.all_qubits()))))))
                for x in op)
        ]

    return NotImplemented


def _sample_helper(sim, state, n_qubits, n_samples):
    if isinstance(sim, cirq.Simulator):
        return cirq.sample_state_vector(state.final_state_vector,
                                        list(range(n_qubits)),
                                        repetitions=n_samples)
    if isinstance(sim, cirq.DensityMatrixSimulator):
        return cirq.sample_density_matrix(state.final_density_matrix,
                                          list(range(n_qubits)),
                                          repetitions=n_samples)

    return NotImplemented


class BatchUtilTest(tf.test.TestCase, parameterized.TestCase):
    """Test cases for BatchUtils main functions."""

    def test_batch_deserialize_programs(self):
        """Confirm that tensors are converted to Cirq correctly."""
        qubits = cirq.GridQubit.rect(1, N_QUBITS)
        (exp_circuits,
         exp_resolvers) = util.random_symbol_circuit_resolver_batch(
             qubits, SYMBOLS, BATCH_SIZE)
        programs = util.convert_to_tensor(exp_circuits)
        symbol_names = tf.constant(SYMBOLS)
        symbol_values = tf.constant(
            [[r[k] for k in SYMBOLS] for r in exp_resolvers])
        deser_circuits, deser_resolvers = batch_util.batch_deserialize_programs(
            programs, symbol_names, symbol_values)
        for exp_moment, deser_moment in zip(exp_circuits, deser_circuits):
            for exp_op, deser_op in zip(exp_moment, deser_moment):
                self.assertTrue(
                    util.is_gate_approx_eq(exp_op.gate, deser_op.gate))
        for e_r, d_r in zip(exp_resolvers, deser_resolvers):
            self.assertTrue(set(e_r) == set(d_r))
            for k in e_r:
                self.assertTrue(
                    util.is_expression_approx_eq(e_r[k], d_r[k], 1e-5))

    def test_batch_deserialize_operators(self):
        """Confirm that tensors are converted to PauliSums correctly."""
        qubits = cirq.GridQubit.rect(1, N_QUBITS)
        psums = []
        for _ in range(BATCH_SIZE):
            psums.append(
                util.random_pauli_sums(qubits, PAULI_LENGTH, BATCH_SIZE))
        psums_tf = util.convert_to_tensor(psums)
        psums_deser = batch_util.batch_deserialize_operators(psums_tf)
        cirq.protocols.approx_eq(psums, psums_deser)

    @parameterized.parameters([{
        'sim': cirq.DensityMatrixSimulator()
    }, {
        'sim': cirq.Simulator()
    }])
    def test_batch_simulate_state_vector(self, sim):
        """Test variable sized state vector output."""
        circuit_batch, resolver_batch = _get_mixed_batch(
            cirq.GridQubit.rect(1, N_QUBITS), SYMBOLS, BATCH_SIZE)
        results = batch_util.batch_calculate_state(circuit_batch,
                                                   resolver_batch, sim)

        for circuit, resolver, result in zip(circuit_batch, resolver_batch,
                                             results):
            r = _pad_state(sim, sim.simulate(circuit, resolver), N_QUBITS)
            self.assertAllClose(r, result, rtol=1e-5, atol=1e-5)

        self.assertDTypeEqual(results, np.complex64)

    @parameterized.parameters([{
        'sim': cirq.DensityMatrixSimulator()
    }, {
        'sim': cirq.Simulator()
    }])
    def test_batch_expectation(self, sim):
        """Test expectation."""
        qubits = cirq.GridQubit.rect(1, N_QUBITS)
        circuit_batch, resolver_batch = _get_mixed_batch(
            qubits + [cirq.GridQubit(9, 9)], SYMBOLS, BATCH_SIZE)
        ops = util.random_pauli_sums(qubits, PAULI_LENGTH, BATCH_SIZE)

        results = batch_util.batch_calculate_expectation(
            circuit_batch, resolver_batch, [[x] for x in ops], sim)

        for circuit, resolver, result, op in zip(circuit_batch, resolver_batch,
                                                 results, ops):
            r = _expectation_helper(sim, circuit, resolver, op)
            self.assertAllClose(r, result, rtol=1e-5, atol=1e-5)

        self.assertDTypeEqual(results, np.float32)

    @parameterized.parameters([{
        'sim': cirq.DensityMatrixSimulator()
    }, {
        'sim': cirq.Simulator()
    }])
    def test_batch_sampled_expectation(self, sim):
        """Test expectation."""
        qubits = cirq.GridQubit.rect(1, N_QUBITS)
        circuit_batch, resolver_batch = _get_mixed_batch(
            qubits + [cirq.GridQubit(9, 9)], SYMBOLS, BATCH_SIZE)

        ops = util.random_pauli_sums(qubits, PAULI_LENGTH, BATCH_SIZE)
        n_samples = [[1000] for _ in range(len(ops))]

        results = batch_util.batch_calculate_sampled_expectation(
            circuit_batch, resolver_batch, [[x] for x in ops], n_samples, sim)

        for circuit, resolver, result, op in zip(circuit_batch, resolver_batch,
                                                 results, ops):
            r = _expectation_helper(sim, circuit, resolver, op)
            self.assertAllClose(r, result, rtol=1.0, atol=1e-1)

        self.assertDTypeEqual(results, np.float32)

    @parameterized.parameters([{
        'sim': cirq.DensityMatrixSimulator()
    }, {
        'sim': cirq.Simulator()
    }])
    def test_batch_sample_basic(self, sim):
        """Test sampling."""
        n_samples = 1
        n_qubits = 8
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit = cirq.Circuit(*cirq.Z.on_each(*qubits[:n_qubits // 2]),
                               *cirq.X.on_each(*qubits[n_qubits // 2:]))

        test_results = batch_util.batch_sample([circuit],
                                               [cirq.ParamResolver({})],
                                               n_samples, sim)

        state = sim.simulate(circuit, cirq.ParamResolver({}))
        expected_results = _sample_helper(sim, state, len(qubits), n_samples)

        self.assertAllEqual(expected_results, test_results[0])
        self.assertDTypeEqual(test_results, np.int32)

    @parameterized.parameters([{
        'sim': cirq.DensityMatrixSimulator()
    }, {
        'sim': cirq.Simulator()
    }])
    def test_batch_sample(self, sim):
        """Test sampling."""
        n_samples = 2000 * (2**N_QUBITS)

        circuit_batch, resolver_batch = _get_mixed_batch(
            cirq.GridQubit.rect(1, N_QUBITS), SYMBOLS, BATCH_SIZE)

        results = batch_util.batch_sample(circuit_batch, resolver_batch,
                                          n_samples, sim)

        tfq_histograms = []
        for r in results:
            tfq_histograms.append(
                np.histogram(r.dot(1 << np.arange(r.shape[-1] - 1, -1, -1)),
                             range=(0, 2**N_QUBITS),
                             bins=2**N_QUBITS)[0])

        cirq_histograms = []
        for circuit, resolver in zip(circuit_batch, resolver_batch):
            state = sim.simulate(circuit, resolver)
            r = _sample_helper(sim, state, len(circuit.all_qubits()), n_samples)
            cirq_histograms.append(
                np.histogram(r.dot(1 << np.arange(r.shape[-1] - 1, -1, -1)),
                             range=(0, 2**N_QUBITS),
                             bins=2**N_QUBITS)[0])

        for a, b in zip(tfq_histograms, cirq_histograms):
            self.assertLess(stats.entropy(a + 1e-8, b + 1e-8), 0.005)

        self.assertDTypeEqual(results, np.int32)

    @parameterized.parameters([{
        'sim': cirq.DensityMatrixSimulator()
    }, {
        'sim': cirq.Simulator()
    }])
    def test_empty_circuits(self, sim):
        """Test functions with empty circuits."""
        # Common preparation
        resolver_batch = [cirq.ParamResolver({}) for _ in range(BATCH_SIZE)]
        circuit_batch = [cirq.Circuit() for _ in range(BATCH_SIZE)]
        qubits = cirq.GridQubit.rect(1, N_QUBITS)
        ops = util.random_pauli_sums(qubits, PAULI_LENGTH, BATCH_SIZE)
        n_samples = [[1000] for _ in range(len(ops))]
        # If there is no op on a qubit, the expectation answer is -2.0
        true_expectation = (-2.0,)

        # (1) Test expectation
        results = batch_util.batch_calculate_expectation(
            circuit_batch, resolver_batch, [[x] for x in ops], sim)

        for _, _, result, _ in zip(circuit_batch, resolver_batch, results, ops):
            self.assertAllClose(true_expectation, result, rtol=1e-5, atol=1e-5)

        self.assertDTypeEqual(results, np.float32)

        # (2) Test sampled_expectation
        results = batch_util.batch_calculate_sampled_expectation(
            circuit_batch, resolver_batch, [[x] for x in ops], n_samples, sim)

        for _, _, result, _ in zip(circuit_batch, resolver_batch, results, ops):
            self.assertAllClose(true_expectation, result, rtol=1.0, atol=1e-1)

        self.assertDTypeEqual(results, np.float32)

        # (3) Test state
        results = batch_util.batch_calculate_state(circuit_batch,
                                                   resolver_batch, sim)

        for circuit, resolver, result in zip(circuit_batch, resolver_batch,
                                             results):
            r = _pad_state(sim, sim.simulate(circuit, resolver), 0)
            self.assertAllClose(r, result, rtol=1e-5, atol=1e-5)

        self.assertDTypeEqual(results, np.complex64)

        # (4) Test sampling
        n_samples = 2000 * (2**N_QUBITS)
        results = batch_util.batch_sample(circuit_batch, resolver_batch,
                                          n_samples, sim)

        for circuit, resolver, a in zip(circuit_batch, resolver_batch, results):
            state = sim.simulate(circuit, resolver)
            r = _sample_helper(sim, state, len(circuit.all_qubits()), n_samples)
            self.assertAllClose(r, a, atol=1e-5)

        self.assertDTypeEqual(results, np.int32)


if __name__ == '__main__':
    tf.test.main()
