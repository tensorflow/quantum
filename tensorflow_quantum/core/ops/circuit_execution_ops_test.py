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
"""Module to test consistency between Cirq and TFQ circuit execution ops."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from scipy import stats
import cirq
import cirq_google

from tensorflow_quantum.core.ops import batch_util, circuit_execution_ops
from tensorflow_quantum.python import util

# Number of random circuits to use in a test batch.
BATCH_SIZE = 15

# These get used everywhere
WF_SIM = cirq.sim.sparse_simulator.Simulator()
DM_SIM = cirq.sim.density_matrix_simulator.DensityMatrixSimulator()

EXPECTATION_OPS = [
    circuit_execution_ops.get_expectation_op(backend=None,
                                             quantum_concurrent=True),
    circuit_execution_ops.get_expectation_op(backend=WF_SIM,
                                             quantum_concurrent=True),
    circuit_execution_ops.get_expectation_op(backend=DM_SIM,
                                             quantum_concurrent=True),
    # For timing interests C++ backend is tested in quantum_concurrent mode.
    circuit_execution_ops.get_expectation_op(backend=None,
                                             quantum_concurrent=False)
]

SAMPLING_OPS = [
    circuit_execution_ops.get_sampling_op(backend=None,
                                          quantum_concurrent=True),
    circuit_execution_ops.get_sampling_op(backend=WF_SIM,
                                          quantum_concurrent=True),
    circuit_execution_ops.get_sampling_op(backend=DM_SIM,
                                          quantum_concurrent=True),
    # For timing interests C++ backend is tested in quantum_concurrent mode.
    circuit_execution_ops.get_sampling_op(backend=None,
                                          quantum_concurrent=False)
]

STATE_OPS = [
    circuit_execution_ops.get_state_op(backend=None, quantum_concurrent=True),
    circuit_execution_ops.get_state_op(backend=WF_SIM, quantum_concurrent=True),
    circuit_execution_ops.get_state_op(backend=DM_SIM, quantum_concurrent=True),
    # For timing interests C++ backend is tested in quantum_concurrent mode.
    circuit_execution_ops.get_state_op(backend=None, quantum_concurrent=False)
]

SAMPLED_EXPECTATION_OPS = [
    circuit_execution_ops.get_sampled_expectation_op(backend=None,
                                                     quantum_concurrent=True),
    circuit_execution_ops.get_sampled_expectation_op(backend=WF_SIM,
                                                     quantum_concurrent=True),
    circuit_execution_ops.get_sampled_expectation_op(backend=DM_SIM,
                                                     quantum_concurrent=True),
    # For timing interests C++ backend is tested in quantum_concurrent mode.
    circuit_execution_ops.get_sampled_expectation_op(backend=None,
                                                     quantum_concurrent=False),
]

SIMS = [WF_SIM, WF_SIM, DM_SIM, WF_SIM]


class OpGetterInputChecks(tf.test.TestCase):
    """Check that the op getters handle inputs correctly."""

    def test_get_expectation_inputs(self):
        """Test that get expectation only accepts inputs it should."""
        circuit_execution_ops.get_expectation_op()
        circuit_execution_ops.get_expectation_op(backend=cirq.Simulator())
        circuit_execution_ops.get_expectation_op(
            backend=cirq.DensityMatrixSimulator())
        circuit_execution_ops.get_expectation_op()
        with self.assertRaisesRegex(NotImplementedError,
                                    expected_regex='Sample-based'):
            circuit_execution_ops.get_expectation_op(
                cirq_google.engine.ProcessorSampler(processor='test'))
        with self.assertRaisesRegex(
                TypeError,
                expected_regex="cirq.sim.simulator.SimulatesExpectationValues"):
            circuit_execution_ops.get_expectation_op(backend="junk")

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must be type bool."):
            circuit_execution_ops.get_expectation_op(quantum_concurrent='junk')

    def test_get_sampled_expectation_inputs(self):
        """Test that get expectation only accepts inputs it should."""
        circuit_execution_ops.get_sampled_expectation_op()
        circuit_execution_ops.get_sampled_expectation_op(
            backend=cirq.Simulator())
        circuit_execution_ops.get_sampled_expectation_op(
            backend=cirq.DensityMatrixSimulator())
        circuit_execution_ops.get_sampled_expectation_op(
            cirq_google.engine.ProcessorSampler(processor='test'))
        with self.assertRaisesRegex(TypeError, expected_regex="a Cirq.Sampler"):
            circuit_execution_ops.get_sampled_expectation_op(backend="junk")

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must be type bool."):
            circuit_execution_ops.get_sampled_expectation_op(
                quantum_concurrent='junk')

    def test_get_samples_inputs(self):
        """Test that get_samples only accepts inputs it should."""
        circuit_execution_ops.get_sampling_op()
        circuit_execution_ops.get_sampling_op(backend=cirq.Simulator())
        circuit_execution_ops.get_sampling_op(
            backend=cirq.DensityMatrixSimulator())
        circuit_execution_ops.get_sampling_op(
            backend=cirq_google.engine.ProcessorSampler(processor='test'))
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="Expected a Cirq.Sampler"):
            circuit_execution_ops.get_sampling_op(backend="junk")

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must be type bool."):
            circuit_execution_ops.get_sampling_op(quantum_concurrent='junk')

    def test_get_state_inputs(self):
        """Test that get_states only accepts inputs it should."""
        circuit_execution_ops.get_state_op()
        circuit_execution_ops.get_state_op(backend=cirq.Simulator())
        circuit_execution_ops.get_state_op(
            backend=cirq.DensityMatrixSimulator())
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="Cirq.SimulatesFinalState"):
            circuit_execution_ops.get_state_op(backend="junk")
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="Cirq.SimulatesFinalState"):
            circuit_execution_ops.get_state_op(
                cirq_google.engine.ProcessorSampler(processor='test'))

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must be type bool."):
            circuit_execution_ops.get_state_op(quantum_concurrent='junk')


class ExecutionOpsConsistentyTest(tf.test.TestCase, parameterized.TestCase):
    """Test all ops produce equivalent output to one another."""

    @parameterized.parameters([{
        'op_and_sim': (op, sim)
    } for (op, sim) in zip(STATE_OPS, SIMS)])
    def test_supported_gates_consistent(self, op_and_sim):
        """Ensure that supported gates are consistent across backends."""
        op = op_and_sim[0]
        sim = op_and_sim[1]
        # mix qubit types.
        qubits = cirq.GridQubit.rect(1, 4) + [cirq.LineQubit(10)]
        circuit_batch = []

        gate_ref = util.get_supported_gates()
        for gate in gate_ref:
            # Create a circuit with non zero entries on real
            # and imaginary values.
            c = cirq.Circuit()
            for qubit in qubits:
                c += cirq.Circuit(cirq.Y(qubit)**0.125)

            if gate_ref[gate] == 2:
                op_qubits = np.random.choice(qubits, size=2, replace=False)
                c += cirq.Circuit(gate(*op_qubits))
            elif gate_ref[gate] == 1:
                op_qubits = np.random.choice(qubits, size=1, replace=False)
                c += cirq.Circuit(gate(*op_qubits))
            else:
                raise ValueError(
                    "Unable to test supported gates across all ops."
                    "please update circuit_execution_ops_test.py")

            circuit_batch.append(c)

        op_states = op(util.convert_to_tensor(circuit_batch), [],
                       [[]] * len(circuit_batch)).to_list()
        cirq_states = batch_util.batch_calculate_state(
            circuit_batch, [cirq.ParamResolver({}) for _ in circuit_batch], sim)

        self.assertAllClose(cirq_states, op_states, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim)
                                   for (op, sim) in zip(STATE_OPS, SIMS)],
                    'n_qubits': [3, 7]
                })))
    def test_simulate_state_no_symbols(self, op_and_sim, n_qubits):
        """Compute states using cirq and tfq without symbols."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        circuit_batch, resolver_batch = util.random_circuit_resolver_batch(
            cirq.GridQubit.rect(1, n_qubits), BATCH_SIZE)

        op_states = op(util.convert_to_tensor(circuit_batch), [],
                       [[]] * BATCH_SIZE).to_list()
        cirq_states = batch_util.batch_calculate_state(circuit_batch,
                                                       resolver_batch, sim)

        self.assertAllClose(cirq_states, op_states, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim)
                                   for (op, sim) in zip(STATE_OPS, SIMS)],
                    'n_qubits': [3, 7],
                    'symbol_names': [['a'], ['a', 'b'],
                                     ['a', 'b', 'c', 'd', 'e']]
                })))
    def test_simulate_state_with_symbols(self, op_and_sim, n_qubits,
                                         symbol_names):
        """Compute states using cirq and tfq with symbols."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                cirq.GridQubit.rect(1, n_qubits), symbol_names, BATCH_SIZE)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        op_states = op(util.convert_to_tensor(circuit_batch), symbol_names,
                       symbol_values_array).to_list()

        cirq_states = batch_util.batch_calculate_state(circuit_batch,
                                                       resolver_batch, sim)

        self.assertAllClose(cirq_states, op_states, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim) for (
                        op,
                        sim) in zip(STATE_OPS[:-2] +
                                    [STATE_OPS[-1]], SIMS[:-2] + [SIMS[-1]])],
                })))
    def test_simulate_state_large(self, op_and_sim):
        """Test a reasonably large and complex circuit."""
        op, sim = op_and_sim
        symbol_names = []
        circuit_batch, resolver_batch = \
            util.random_circuit_resolver_batch(
                cirq.GridQubit.rect(4, 4), 5)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch]).astype(np.float32)

        op_states = op(util.convert_to_tensor(circuit_batch), symbol_names,
                       symbol_values_array).to_list()

        cirq_states = batch_util.batch_calculate_state(circuit_batch,
                                                       resolver_batch, sim)

        self.assertAllClose(cirq_states, op_states, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'op_and_sim': [(op, sim) for (op, sim) in zip(STATE_OPS, SIMS)],
            })))
    def test_simulate_state_empty(self, op_and_sim):
        """Test empty circuits for states using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        circuit_batch = [cirq.Circuit() for _ in range(BATCH_SIZE)]
        resolver_batch = [cirq.ParamResolver({}) for _ in range(BATCH_SIZE)]

        op_states = op(util.convert_to_tensor(circuit_batch), [],
                       [[]] * BATCH_SIZE).to_list()
        cirq_states = batch_util.batch_calculate_state(circuit_batch,
                                                       resolver_batch, sim)

        self.assertAllClose(cirq_states, op_states, atol=1e-5, rtol=1e-5)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'op_and_sim': [(op, sim) for (op, sim) in zip(STATE_OPS, SIMS)]
            })))
    def test_simulate_state_no_circuits(self, op_and_sim):
        """Test no circuits for states using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        circuit_batch = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_params = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)

        op_states = op(circuit_batch, [], empty_params).numpy()
        cirq_states = batch_util.batch_calculate_state([], [], sim)
        self.assertEqual(op_states.shape, cirq_states.shape)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim)
                                   for (op, sim) in zip(EXPECTATION_OPS, SIMS)],
                    'n_qubits': [3, 7],
                    'symbol_names': [['a', 'b', 'c', 'd', 'e']],
                    'max_paulisum_length': [6]
                })))
    def test_analytical_expectation(self, op_and_sim, n_qubits, symbol_names,
                                    max_paulisum_length):
        """Compute expectations using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        qubits = cirq.LineQubit.range(n_qubits - 1) + [cirq.GridQubit(0, 0)]
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, BATCH_SIZE)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, max_paulisum_length,
                                            BATCH_SIZE)

        op_expectations = op(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array,
            util.convert_to_tensor([[psum] for psum in pauli_sums]))

        cirq_expectations = batch_util.batch_calculate_expectation(
            circuit_batch, resolver_batch, [[x] for x in pauli_sums], sim)

        self.assertAllClose(op_expectations.numpy().flatten(),
                            cirq_expectations.flatten(),
                            rtol=1e-5,
                            atol=1e-5)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim)
                                   for (op, sim) in zip(EXPECTATION_OPS, SIMS)],
                    'n_qubits': [3],
                    'symbol_names': [['a', 'b', 'c', 'd', 'e']],
                    'max_paulisum_length': [6]
                })))
    def test_analytical_expectation_empty(self, op_and_sim, n_qubits,
                                          symbol_names, max_paulisum_length):
        """Test empty circuits for analytical expectation using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch = [cirq.Circuit() for _ in range(BATCH_SIZE)]
        resolver_batch = [cirq.ParamResolver({}) for _ in range(BATCH_SIZE)]

        symbol_values_array = np.array(
            [[0.0 for _ in symbol_names] for _ in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, max_paulisum_length,
                                            BATCH_SIZE)

        op_expectations = op(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array,
            util.convert_to_tensor([[psum] for psum in pauli_sums]))

        cirq_expectations = batch_util.batch_calculate_expectation(
            circuit_batch, resolver_batch, [[x] for x in pauli_sums], sim)

        self.assertAllClose(op_expectations.numpy().flatten(),
                            cirq_expectations.flatten(),
                            rtol=1e-5,
                            atol=1e-5)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim)
                                   for (op, sim) in zip(EXPECTATION_OPS, SIMS)]
                })))
    def test_analytical_expectation_no_circuits(self, op_and_sim):
        """Test no circuits for states using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        circuit_batch = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_params = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        empty_ops = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)

        op_exp = op(circuit_batch, [], empty_params, empty_ops).numpy()
        cirq_exp = batch_util.batch_calculate_expectation([], [], [[]], sim)
        self.assertEqual(op_exp.shape, cirq_exp.shape)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim) for (
                        op, sim) in zip(SAMPLED_EXPECTATION_OPS, SIMS)],
                    'n_qubits': [3, 7],
                    'symbol_names': [['a', 'b', 'c', 'd', 'e']],
                    'max_paulisum_length': [6]
                })))
    def test_sampled_expectation(self, op_and_sim, n_qubits, symbol_names,
                                 max_paulisum_length):
        """Compute sampled expectations using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, BATCH_SIZE)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, max_paulisum_length,
                                            BATCH_SIZE)
        num_samples = [[10000]] * BATCH_SIZE

        op_expectations = op(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array,
            util.convert_to_tensor([[psum] for psum in pauli_sums]),
            num_samples)

        cirq_expectations = batch_util.batch_calculate_sampled_expectation(
            circuit_batch, resolver_batch, [[x] for x in pauli_sums],
            num_samples, sim)

        self.assertAllClose(op_expectations.numpy().flatten(),
                            cirq_expectations.flatten(),
                            rtol=1e-1,
                            atol=1e-1)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim) for (
                        op, sim) in zip(SAMPLED_EXPECTATION_OPS, SIMS)],
                    'n_qubits': [3],
                    'symbol_names': [['a', 'b', 'c', 'd', 'e']],
                    'max_paulisum_length': [6]
                })))
    def test_sampled_expectation_empty(self, op_and_sim, n_qubits, symbol_names,
                                       max_paulisum_length):
        """Test empty circuits for sampled expectation using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch = [cirq.Circuit() for _ in range(BATCH_SIZE)]
        resolver_batch = [cirq.ParamResolver({}) for _ in range(BATCH_SIZE)]

        symbol_values_array = np.array(
            [[0.0 for _ in symbol_names] for _ in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, max_paulisum_length,
                                            BATCH_SIZE)
        num_samples = [[1000]] * BATCH_SIZE

        op_expectations = op(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array,
            util.convert_to_tensor([[psum] for psum in pauli_sums]),
            num_samples)

        cirq_expectations = batch_util.batch_calculate_sampled_expectation(
            circuit_batch, resolver_batch, [[x] for x in pauli_sums],
            num_samples, sim)

        self.assertAllClose(op_expectations.numpy().flatten(),
                            cirq_expectations.flatten(),
                            rtol=1e-1,
                            atol=1e-1)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim) for (
                        op, sim) in zip(SAMPLED_EXPECTATION_OPS, SIMS)]
                })))
    def test_sampled_expectation_no_circuits(self, op_and_sim):
        """Test no circuits for states using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        circuit_batch = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_params = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        empty_ops = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)
        empty_samples = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.int32)

        op_exp = op(circuit_batch, [], empty_params, empty_ops,
                    empty_samples).numpy()
        cirq_exp = batch_util.batch_calculate_sampled_expectation([], [], [[]],
                                                                  [], sim)
        self.assertEqual(op_exp.shape, cirq_exp.shape)

    # keep the qubit count low here, all computations scale exponentially
    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim)
                                   for (op, sim) in zip(SAMPLING_OPS, SIMS)],
                    'n_qubits': [6],
                    'symbol_names': [['a', 'b', 'c', 'd', 'e']]
                })))
    def test_sampling(self, op_and_sim, n_qubits, symbol_names):
        """Compare sampling with tfq ops and Cirq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]
        qubits = cirq.GridQubit.rect(1, n_qubits)
        n_samples = int((2**n_qubits) * 1000)

        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, BATCH_SIZE, n_moments=30)
        for i in range(BATCH_SIZE):
            circuit_batch[i] += cirq.Circuit(
                *[cirq.H(qubit) for qubit in qubits])

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        op_samples = np.array(
            op(util.convert_to_tensor(circuit_batch), symbol_names,
               symbol_values_array, [n_samples]).to_list())

        op_histograms = [
            np.histogram(
                sample.dot(1 << np.arange(sample.shape[-1] - 1, -1, -1)),
                range=(0, 2**len(qubits)),
                bins=2**len(qubits))[0] for sample in op_samples
        ]

        cirq_samples = batch_util.batch_sample(circuit_batch, resolver_batch,
                                               n_samples, sim)

        cirq_histograms = [
            np.histogram(
                sample.dot(1 << np.arange(sample.shape[-1] - 1, -1, -1)),
                range=(0, 2**len(qubits)),
                bins=2**len(qubits))[0] for sample in cirq_samples
        ]

        for a, b in zip(op_histograms, cirq_histograms):
            self.assertLess(stats.entropy(a + 1e-8, b + 1e-8), 0.005)

    # keep the qubit count low here, all computations scale exponentially
    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'op_and_sim': [(op, sim)
                                   for (op, sim) in zip(SAMPLING_OPS, SIMS)],
                    'n_qubits': [3],
                    'symbol_names': [['a', 'b', 'c', 'd', 'e']]
                })))
    def test_sampling_empty(self, op_and_sim, n_qubits, symbol_names):
        """Test empty circuits for sampling using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]
        qubits = cirq.GridQubit.rect(1, n_qubits)
        n_samples = int((2**n_qubits) * 1000)

        circuit_batch = [cirq.Circuit() for _ in range(BATCH_SIZE)]
        resolver_batch = [cirq.ParamResolver({}) for _ in range(BATCH_SIZE)]

        symbol_values_array = np.array(
            [[0.0 for _ in symbol_names] for _ in resolver_batch])

        op_samples = np.array(
            op(util.convert_to_tensor(circuit_batch), symbol_names,
               symbol_values_array, [n_samples]).to_list())

        op_histograms = [
            np.histogram(
                sample.dot(1 << np.arange(sample.shape[-1] - 1, -1, -1)),
                range=(0, 2**len(qubits)),
                bins=2**len(qubits))[0] for sample in op_samples
        ]

        cirq_samples = batch_util.batch_sample(circuit_batch, resolver_batch,
                                               n_samples, sim)

        cirq_histograms = [
            np.histogram(
                sample.dot(1 << np.arange(sample.shape[-1] - 1, -1, -1)),
                range=(0, 2**len(qubits)),
                bins=2**len(qubits))[0] for sample in cirq_samples
        ]

        for a, b in zip(op_histograms, cirq_histograms):
            self.assertLess(stats.entropy(a + 1e-8, b + 1e-8), 0.005)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'op_and_sim': [(op, sim)
                               for (op, sim) in zip(SAMPLING_OPS, SIMS)]
            })))
    def test_sampling_no_circuits(self, op_and_sim):
        """Test no circuits for states using cirq and tfq."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        circuit_batch = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_params = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        num_samples = tf.convert_to_tensor([5])
        op_states = op(circuit_batch, [], empty_params, num_samples).numpy()
        cirq_samples = batch_util.batch_sample([], [], [5], sim)
        self.assertEqual(op_states.shape, cirq_samples.shape)


if __name__ == '__main__':
    tf.test.main()
