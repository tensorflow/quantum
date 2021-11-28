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
"""Tests that specifically target noisy expectation calculation."""
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops import batch_util
from tensorflow_quantum.core.ops.noise import noisy_sampled_expectation_op
from tensorflow_quantum.python import util


class NoisyExpectationCalculationTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq.noise.expectation."""

    def test_noisy_expectation_inputs(self):
        """Make sure noisy expectation op fails gracefully on bad inputs."""
        n_qubits = 5
        batch_size = 5
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size, include_channels=True)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)
        projector_sums = util.random_projector_sums(qubits, 3, batch_size)
        num_samples = [[10, 10]] * batch_size

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1'):
            # Circuit tensor has too many dimensions.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1.'):
            # symbol_names tensor has too many dimensions.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too many dimensions.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too few dimensions.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[0],
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too few dimensions.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch),
                symbol_names, symbol_values_array,
                util.convert_to_tensor(list(pauli_sums)),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'projector_sums must be rank 2.'):
            # pauli_sums tensor has too few dimensions.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch),
                symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor(list(projector_sums)),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too many dimensions.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                [util.convert_to_tensor([[x] for x in pauli_sums])],
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'projector_sums must be rank 2.'):
            # pauli_sums tensor has too many dimensions.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                [util.convert_to_tensor([[x] for x in projector_sums])],
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'num_samples must be rank 2'):
            # num_samples tensor has the wrong shape.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'num_samples must be rank 2'):
            # num_samples tensor has the wrong shape.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples[0])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # circuit tensor has the right type but invalid values.
            noisy_sampled_expectation_op.sampled_expectation(
                ['junk'] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type but invalid values.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), ['junk'],
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'qubits not found in circuit'):
            # pauli_sums tensor has the right type but invalid values.
            new_qubits = [cirq.GridQubit(5, 5), cirq.GridQubit(9, 9)]
            new_pauli_sums = util.random_pauli_sums(new_qubits, 2, batch_size)
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in new_pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'qubits not found in circuit'):
            # pauli_sums tensor has the right type but invalid values.
            new_qubits = [cirq.GridQubit(5, 5), cirq.GridQubit(9, 9)]
            new_projector_sums = util.random_projector_sums(new_qubits, 2, batch_size)
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in new_projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # pauli_sums tensor has the right type but invalid values 2.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                [['junk']] * batch_size,
                util.convert_to_tensor([[x] for x in new_projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # pauli_sums tensor has the right type but invalid values 2.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                [['junk']] * batch_size,
                num_samples)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # circuits tensor has the wrong type.
            noisy_sampled_expectation_op.sampled_expectation(
                [1.0] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), [0.1234],
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # pauli_sums tensor has the wrong type.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [[1.0]] * batch_size,
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # projector_sums tensor has the wrong type.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [[1.0]] * batch_size,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                num_samples)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # we are missing an argument.
            # pylint: disable=no-value-for-parameter
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, num_samples)
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]), [],
                num_samples)
            # pylint: enable=too-many-function-args

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong op size.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor([cirq.Circuit()]), symbol_names,
                symbol_values_array.astype(np.float64),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'greater than 0'):
            # pylint: disable=too-many-function-args
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                [[-1]] * batch_size)
            # pylint: enable=too-many-function-args

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong symbol_values size.
            noisy_sampled_expectation_op.sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[:int(batch_size * 0.5)],
                util.convert_to_tensor([[x] for x in pauli_sums]),
                util.convert_to_tensor([[x] for x in projector_sums]),
                num_samples)

    @parameterized.parameters([
        {
            'n_qubits': 13,
            'batch_size': 1,
            'noisy': False
        },  # ComputeLarge.
        {
            'n_qubits': 6,
            'batch_size': 25,
            'noisy': False
        },  # ComputeSmall.
        {
            'n_qubits': 6,
            'batch_size': 10,
            'noisy': True
        },  # ComputeSmall.
        {
            'n_qubits': 8,
            'batch_size': 1,
            'noisy': True
        }  # ComputeLarge.
    ])
    def test_simulate_consistency(self, batch_size, n_qubits, noisy):
        """Test consistency with batch_util.py simulation."""
        symbol_names = ['alpha', 'beta']
        qubits = cirq.GridQubit.rect(1, n_qubits)

        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size, include_channels=noisy)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums1 = util.random_pauli_sums(qubits, 3, batch_size)
        pauli_sums2 = util.random_pauli_sums(qubits, 3, batch_size)
        batch_pauli_sums = [[x, y] for x, y in zip(pauli_sums1, pauli_sums2)]
        projector_sums1 = util.random_projector_sums(qubits, 3, batch_size)
        projector_sums2 = util.random_projector_sums(qubits, 3, batch_size)
        batch_projector_sums = [
            [x, y] for x, y in zip(projector_sums1, projector_sums2)
        ]
        batch_both = [[x, y, z, t] for x, y, z, t in zip(
            pauli_sums1, pauli_sums2, projector_sums1, projector_sums2)]
        num_samples = [[10000] * 4] * batch_size

        op_exps = noisy_sampled_expectation_op.sampled_expectation(
            util.convert_to_tensor(circuit_batch),
            symbol_names, symbol_values_array,
            util.convert_to_tensor(batch_pauli_sums),
            util.convert_to_tensor(batch_projector_sums), num_samples)

        cirq_exps = batch_util.batch_calculate_expectation(
            circuit_batch, resolver_batch, batch_both,
            cirq.DensityMatrixSimulator())
        tol = 0.5
        self.assertAllClose(cirq_exps, op_exps, atol=tol, rtol=tol)

    @parameterized.parameters([{
        'channel': x
    } for x in util.get_supported_channels()])
    def test_single_channel(self, channel):
        """Individually test adding just a single channel type to circuits."""
        symbol_names = []
        batch_size = 5
        n_qubits = 6
        qubits = cirq.LineQubit.range(n_qubits)

        circuit_batch, resolver_batch = \
            util.random_circuit_resolver_batch(
                qubits, batch_size, include_channels=False)

        for i in range(batch_size):
            circuit_batch[i] = circuit_batch[i] + channel.on_each(*qubits)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums1 = util.random_pauli_sums(qubits, 3, batch_size)
        pauli_sums2 = util.random_pauli_sums(qubits, 3, batch_size)
        batch_pauli_sums = [[x, y] for x, y in zip(pauli_sums1, pauli_sums2)]
        projector_sums1 = util.random_projector_sums(qubits, 3, batch_size)
        projector_sums2 = util.random_projector_sums(qubits, 3, batch_size)
        batch_projector_sums = [
            [x, y] for x, y in zip(projector_sums1, projector_sums2)
        ]
        batch_both = [[x, y, z, t] for x, y, z, t in zip(
            pauli_sums1, pauli_sums2, projector_sums1, projector_sums2)]
        num_samples = [[20000] * 4] * batch_size

        op_exps = noisy_sampled_expectation_op.sampled_expectation(
            util.convert_to_tensor(circuit_batch),
            symbol_names, symbol_values_array,
            util.convert_to_tensor(batch_pauli_sums),
            util.convert_to_tensor(batch_projector_sums),
            num_samples)

        cirq_exps = batch_util.batch_calculate_expectation(
            circuit_batch, resolver_batch, batch_both,
            cirq.DensityMatrixSimulator())

        self.assertAllClose(cirq_exps, op_exps, atol=0.35, rtol=0.35)

    def test_correctness_empty(self):
        """Test the expectation for empty circuits."""
        empty_circuit = util.convert_to_tensor([cirq.Circuit()])
        empty_symbols = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        empty_values = tf.convert_to_tensor([[]])
        empty_paulis = tf.convert_to_tensor([[]], dtype=tf.dtypes.string)
        empty_projectors = tf.convert_to_tensor([[]], dtype=tf.dtypes.string)
        empty_n_samples = tf.convert_to_tensor([[]], dtype=tf.int32)

        out = noisy_sampled_expectation_op.sampled_expectation(
            empty_circuit, empty_symbols, empty_values, empty_paulis, empty_projectors,
            empty_n_samples)

        expected = np.array([[]], dtype=np.complex64)
        self.assertAllClose(out, expected)

    def test_correctness_no_circuit(self):
        """Test the correctness with the empty tensor."""
        empty_circuit = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_symbols = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_values = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        empty_paulis = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)
        empty_projectors = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)
        empty_n_samples = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.int32)

        out = noisy_sampled_expectation_op.sampled_expectation(
            empty_circuit, empty_symbols, empty_values, empty_paulis, empty_projectors,
            empty_n_samples)

        self.assertShapeEqual(np.zeros((0, 0)), out)


if __name__ == "__main__":
    tf.test.main()
