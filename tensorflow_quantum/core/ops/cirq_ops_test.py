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
"""Tests for the cirq simulation ops."""
from unittest import mock
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import cirq

from tensorflow_quantum.core.ops import cirq_ops
from tensorflow_quantum.core.serialize import serializer
from tensorflow_quantum.python import util

MOMENT_DEPTH = 25

WF_SIM = cirq.sim.sparse_simulator.Simulator()
DM_SIM = cirq.sim.density_matrix_simulator.DensityMatrixSimulator()


class CirqAnalyticalExpectationTest(tf.test.TestCase):
    """Tests get_cirq_analytical_expectation."""

    def test_get_cirq_analytical_expectation_op(self):
        """Input check the wrapper for the cirq analytical expectation op."""
        with self.assertRaisesRegex(
                TypeError,
                "simulator must inherit cirq.sim.SimulatesFinalState."):
            cirq_ops._get_cirq_analytical_expectation("junk")
        # TODO(peterse): Tighten these tests a bit..
        cirq_ops._get_cirq_analytical_expectation()
        cirq_ops._get_cirq_analytical_expectation(
            cirq.sim.sparse_simulator.Simulator())
        cirq_ops._get_cirq_analytical_expectation(
            cirq.sim.density_matrix_simulator.DensityMatrixSimulator())

    def test_cirq_analytical_expectation_op_inputs(self):
        """Test input checking in the state sim op."""
        test_op = cirq_ops._get_cirq_analytical_expectation(
            cirq.sim.sparse_simulator.Simulator())
        bits = cirq.GridQubit.rect(1, 5)
        test_circuit = serializer.serialize_circuit(
            cirq.testing.random_circuit(bits, MOMENT_DEPTH,
                                        0.9)).SerializeToString()
        test_pauli_sum = serializer.serialize_paulisum(
            cirq.PauliSum.from_pauli_strings([cirq.Z(bits[0])
                                             ])).SerializeToString()
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'symbol_names tensor must be of type string'):
            _ = test_op([test_circuit], [0], [[0]], [[test_pauli_sum]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs tensor must be of type string'):
            _ = test_op([0], ['rx'], [[0]], [test_pauli_sum])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'real-valued numeric tensor.'):
            _ = test_op([test_circuit], ['rx'], 'junk', [[test_pauli_sum]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'size of symbol_names tensor must match'):
            _ = test_op([test_circuit], ['rx'], [[1, 1]], [[test_pauli_sum]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'size of symbol_names tensor must match'):
            _ = test_op([test_circuit], ['rx', 'ry'], [[1]], [[test_pauli_sum]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'first dimension of symbol_values tensor'):
            _ = test_op([test_circuit, test_circuit], ['rx'], [[1]],
                        [test_pauli_sum])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'pauli_sums tensor must be of type string.'):
            _ = test_op([test_circuit], ['rx'], [[1]], 0)
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'pauli_sums tensor must have the same batch shape'):
            _ = test_op([test_circuit], ['rx'], [[1]],
                        [[test_pauli_sum], [test_pauli_sum]])

        _ = test_op([test_circuit], ['rx'], [[1]], [[test_pauli_sum]])
        _ = test_op([test_circuit], [], [[]], [[test_pauli_sum]])

    def test_analytic_expectation_empty_circuit(self):
        """Test empty circuits"""
        test_op = cirq_ops._get_cirq_analytical_expectation(
            cirq.sim.sparse_simulator.Simulator())
        bits = cirq.GridQubit.rect(1, 5)
        test_pauli_sum = serializer.serialize_paulisum(
            cirq.PauliSum.from_pauli_strings([cirq.Z(bits[0])
                                             ])).SerializeToString()
        test_empty_circuit = serializer.serialize_circuit(
            cirq.Circuit()).SerializeToString()
        _ = test_op([test_empty_circuit], [], [[]], [[test_pauli_sum]])


class CirqSampledExpectationTest(tf.test.TestCase):
    """Tests get_cirq_sampled_expectation."""

    def test_get_cirq_sampled_expectation_op(self):
        """Input check the wrapper for the cirq analytical expectation op."""
        with self.assertRaisesRegex(
                TypeError, "cirq.Sampler is required for sampled expectation."):
            cirq_ops._get_cirq_sampled_expectation("junk")
        # TODO(peterse): Tighten these tests a bit..
        cirq_ops._get_cirq_sampled_expectation()
        cirq_ops._get_cirq_sampled_expectation(
            cirq.sim.sparse_simulator.Simulator())
        cirq_ops._get_cirq_sampled_expectation(
            cirq.sim.density_matrix_simulator.DensityMatrixSimulator())

    def test_cirq_sampled_expectation_op_inputs(self):
        """test input checking in the state sim op."""
        test_op = cirq_ops._get_cirq_sampled_expectation(
            cirq.sim.sparse_simulator.Simulator())
        bits = cirq.GridQubit.rect(1, 5)
        test_circuit = serializer.serialize_circuit(
            cirq.testing.random_circuit(bits, MOMENT_DEPTH,
                                        0.9)).SerializeToString()
        test_pauli_sum = serializer.serialize_paulisum(
            cirq.PauliSum.from_pauli_strings([cirq.Z(bits[0])
                                             ])).SerializeToString()
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'symbol_names tensor must be of type string'):
            _ = test_op([test_circuit], [0], [[0]], [[test_pauli_sum]], [[1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs tensor must be of type string'):
            _ = test_op([0], ['rx'], [[0]], [test_pauli_sum], [[1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'real-valued numeric tensor.'):
            _ = test_op([test_circuit], ['rx'], 'junk', [[test_pauli_sum]],
                        [[1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'size of symbol_names tensor must match'):
            _ = test_op([test_circuit], ['rx'], [[1, 1]], [[test_pauli_sum]],
                        [[1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'size of symbol_names tensor must match'):
            _ = test_op([test_circuit], ['rx', 'ry'], [[1]], [[test_pauli_sum]],
                        [[1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'first dimension of symbol_values tensor'):
            _ = test_op([test_circuit, test_circuit], ['rx'], [[1]],
                        [test_pauli_sum], [[1]])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'pauli_sums tensor must be of type string.'):
            _ = test_op([test_circuit], ['rx'], [[1]], 0, [[1]])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'pauli_sums tensor must have the same batch shape'):
            _ = test_op([test_circuit], ['rx'], [[1]],
                        [[test_pauli_sum], [test_pauli_sum]], [[1]])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'num_samples tensor must have the same shape'):
            _ = test_op([test_circuit], ['rx'], [[1]], [[test_pauli_sum]],
                        [[1], [1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'num_samples tensor must be of type int32'):
            _ = test_op([test_circuit], ['rx'], [[1]], [[test_pauli_sum]],
                        [[1.0]])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'num_samples tensor must have the same shape'):
            _ = test_op([test_circuit], ['rx'], [[1]], [[test_pauli_sum]],
                        [[1], [1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'num_samples contains sample value <= 0'):
            _ = test_op([test_circuit], ['rx'], [[1]], [[test_pauli_sum]],
                        [[0]])

        _ = test_op([test_circuit], ['rx'], [[1]], [[test_pauli_sum]], [[1]])
        _ = test_op([test_circuit], [], [[]], [[test_pauli_sum]], [[1]])

    def test_sampled_expectation_empty_circuit(self):
        """Test empty circuits"""
        test_op = cirq_ops._get_cirq_sampled_expectation(
            cirq.sim.sparse_simulator.Simulator())
        bits = cirq.GridQubit.rect(1, 5)
        test_pauli_sum = serializer.serialize_paulisum(
            cirq.PauliSum.from_pauli_strings([cirq.Z(bits[0])
                                             ])).SerializeToString()
        test_empty_circuit = serializer.serialize_circuit(
            cirq.Circuit()).SerializeToString()
        _ = test_op([test_empty_circuit], [], [[]], [[test_pauli_sum]], [[1]])


class CirqSimulateStateTest(tf.test.TestCase, parameterized.TestCase):
    """Tests get_cirq_simulate_state."""

    def test_get_cirq_state_op(self):
        """Input check the wrapper for the cirq state op."""
        with self.assertRaisesRegex(
                TypeError,
                "simulator must inherit cirq.sim.SimulatesFinalState."):
            cirq_ops._get_cirq_simulate_state("junk")
        cirq_ops._get_cirq_simulate_state()
        cirq_ops._get_cirq_simulate_state(cirq.sim.sparse_simulator.Simulator())
        cirq_ops._get_cirq_simulate_state(
            cirq.sim.density_matrix_simulator.DensityMatrixSimulator())

    # TODO(trevormccrt): input checking might be parameterizeable over all ops
    # if we decide to properly input check our c++ ops
    def test_cirq_state_op_inputs(self):
        """test input checking in the state sim op."""
        test_op = cirq_ops._get_cirq_simulate_state(
            cirq.sim.sparse_simulator.Simulator())
        bits = cirq.GridQubit.rect(1, 5)
        test_circuit = serializer.serialize_circuit(
            cirq.testing.random_circuit(bits, MOMENT_DEPTH,
                                        0.9)).SerializeToString()
        # exceptions raised in the tf graph don't get passed
        # through in an identifiable way
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'symbol_names tensor must be of type string'):
            _ = test_op([test_circuit], [0], [[0]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs tensor must be of type string'):
            _ = test_op([0], ['rx'], [[0]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'real-valued numeric tensor.'):
            _ = test_op([test_circuit], ['rx'], 'junk')
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'size of symbol_names tensor must match'):
            _ = test_op([test_circuit], ['rx'], [[1, 1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'size of symbol_names tensor must match'):
            _ = test_op([test_circuit], ['rx', 'ry'], [[1]])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'first dimension of symbol_values tensor'):
            _ = test_op([test_circuit, test_circuit], ['rx'], [[1]])
        _ = test_op([test_circuit], ['rx'], [[1]])
        _ = test_op([test_circuit], [], [[]])

    @parameterized.parameters([
        {
            'op_and_sim': (cirq_ops._get_cirq_simulate_state(WF_SIM), WF_SIM),
            'all_n_qubits': [2, 3]
        },
        {
            'op_and_sim': (cirq_ops._get_cirq_simulate_state(DM_SIM), DM_SIM),
            'all_n_qubits': [2, 3]
        },
        {
            'op_and_sim': (cirq_ops._get_cirq_simulate_state(WF_SIM), WF_SIM),
            'all_n_qubits': [2, 5, 8, 10]
        },
        {
            'op_and_sim': (cirq_ops._get_cirq_simulate_state(DM_SIM), DM_SIM),
            'all_n_qubits': [2, 5, 8, 10]
        },
    ])
    def test_simulate_state_output_padding(self, op_and_sim, all_n_qubits):
        """If a circuit executing op is asked to simulate states given circuits
        acting on different numbers of qubits, the op should return a tensor
        padded with zeros up to the size of the largest circuit. The padding
        should be physically correct, such that samples taken from the padded
        states still match samples taken from the original circuit."""
        op = op_and_sim[0]
        sim = op_and_sim[1]

        circuit_batch = []
        for n_qubits in all_n_qubits:
            qubits = cirq.GridQubit.rect(1, n_qubits)
            circuit_batch += util.random_circuit_resolver_batch(qubits, 1)[0]

        tfq_results = op(util.convert_to_tensor(circuit_batch), [],
                         [[]] * len(circuit_batch))

        # don't use batch_util here to enforce consistent padding everywhere
        # without extra tests
        manual_padded_results = []
        for circuit in circuit_batch:
            result = sim.simulate(circuit)

            # density matricies should be zero everywhere except for the
            # top left corner
            if isinstance(
                    result,
                    cirq.sim.density_matrix_simulator.DensityMatrixTrialResult):
                dm = result.final_density_matrix
                blank_state = np.ones(
                    (2**max(all_n_qubits), 2**(max(all_n_qubits))),
                    dtype=np.complex64) * -2
                blank_state[:dm.shape[0], :dm.shape[1]] = dm
                manual_padded_results.append(blank_state)

            # wavefunctions should be zero everywhere to the right of the states
            # present in this system
            elif isinstance(
                    result,
                    cirq.sim.wave_function_simulator.WaveFunctionTrialResult):
                wf = result.final_state
                blank_state = np.ones(
                    (2**max(all_n_qubits)), dtype=np.complex64) * -2
                blank_state[:wf.shape[0]] = wf
                manual_padded_results.append(blank_state)

            else:
                # TODO
                raise RuntimeError('Simulator returned unknown type of result.')

        self.assertAllClose(tfq_results, manual_padded_results)

    def test_state_empty_circuit(self):
        """Test empty circuits"""
        test_op = cirq_ops._get_cirq_simulate_state(
            cirq.sim.sparse_simulator.Simulator())
        test_empty_circuit = serializer.serialize_circuit(
            cirq.Circuit()).SerializeToString()
        _ = test_op([test_empty_circuit], [], [[]])


class CirqSamplesTest(tf.test.TestCase, parameterized.TestCase):
    """Tests get_cirq_samples."""

    def test_get_cirq_sampling_op(self):
        """Input check the wrapper for the cirq sampling op."""
        with self.assertRaisesRegex(TypeError,
                                    "simulator must inherit cirq.Sampler."):
            cirq_ops._get_cirq_samples("junk")
        cirq_ops._get_cirq_samples()
        cirq_ops._get_cirq_samples(cirq.sim.sparse_simulator.Simulator())
        cirq_ops._get_cirq_samples(
            cirq.sim.density_matrix_simulator.DensityMatrixSimulator())
        mock_engine = mock.Mock()
        cirq_ops._get_cirq_samples(
            cirq.google.QuantumEngineSampler(engine=mock_engine,
                                             processor_id='test',
                                             gate_set=cirq.google.XMON))

    def test_cirq_sampling_op_inputs(self):
        """test input checking in the cirq sampling op."""
        test_op = cirq_ops._get_cirq_samples(
            cirq.sim.sparse_simulator.Simulator())

        bits = cirq.GridQubit.rect(1, 5)
        test_circuit = serializer.serialize_circuit(
            cirq.testing.random_circuit(bits, MOMENT_DEPTH,
                                        0.9)).SerializeToString()

        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'symbol_names tensor must be of type string'):
            _ = test_op([test_circuit], [0], [[0]], [10])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs tensor must be of type string'):
            _ = test_op([0], ['rx'], [[0]], [10])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'real-valued numeric tensor.'):
            _ = test_op([test_circuit], ['rx'], 'junk', [10])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'size of symbol_names tensor must match'):
            _ = test_op([test_circuit], ['rx'], [[1, 1]], [10])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'size of symbol_names tensor must match'):
            _ = test_op([test_circuit], ['rx', 'ry'], [[1]], [10])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'num_samples tensor must be of integer type'):
            _ = test_op([test_circuit], ['rx'], [[1]], "junk")
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'num_samples tensor must have size 1'):
            _ = test_op([test_circuit], ['rx'], [[1]], [10, 10])

        _ = test_op([test_circuit], ['rx'], [[1]], [10])
        _ = test_op([test_circuit], [], [[]], [10])

    @parameterized.parameters([
        {
            'op': cirq_ops._get_cirq_samples(WF_SIM),
            'all_n_qubits': [2, 3],
            'n_samples': 10
        },
        {
            'op': cirq_ops._get_cirq_samples(DM_SIM),
            'all_n_qubits': [2, 3],
            'n_samples': 10
        },
        {
            'op': cirq_ops._get_cirq_samples(WF_SIM),
            'all_n_qubits': [2, 5, 8, 10],
            'n_samples': 10
        },
        {
            'op': cirq_ops._get_cirq_samples(DM_SIM),
            'all_n_qubits': [2, 5, 8, 10],
            'n_samples': 10
        },
    ])
    def test_sampling_output_padding(self, op, all_n_qubits, n_samples):
        """Check that the sampling ops pad outputs correctly"""
        circuits = []
        expected_outputs = []
        for n_qubits in all_n_qubits:
            this_expected_output = np.zeros((n_samples, max(all_n_qubits)))
            this_expected_output[:, max(all_n_qubits) - n_qubits:] = 1
            this_expected_output[:, :max(all_n_qubits) - n_qubits] = -2
            expected_outputs.append(this_expected_output)
            circuits.append(
                cirq.Circuit(
                    *cirq.X.on_each(*cirq.GridQubit.rect(1, n_qubits))))
        results = op(util.convert_to_tensor(circuits), [], [[]] * len(circuits),
                     [n_samples]).numpy()
        self.assertAllClose(expected_outputs, results)

    def test_sample_empty_circuit(self):
        """Test empty circuits"""
        test_op = cirq_ops._get_cirq_samples(
            cirq.sim.sparse_simulator.Simulator())
        test_empty_circuit = serializer.serialize_circuit(
            cirq.Circuit()).SerializeToString()
        _ = test_op([test_empty_circuit], [], [[]], [10])


if __name__ == "__main__":
    tf.test.main()
