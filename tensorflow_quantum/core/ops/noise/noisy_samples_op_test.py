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
"""Tests that specifically target noisy sampling."""
import numpy as np
from scipy import stats
from absl.testing import parameterized
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops import batch_util
from tensorflow_quantum.core.ops.noise import noisy_samples_op
from tensorflow_quantum.python import util


class NoisySamplingTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq.noise.expectation."""

    def _compute_hists(self, x, n_qubits):
        """Compute the batchwise histograms of a sample tensor."""
        x = np.asarray(x)
        return [
            np.histogram(
                sample.dot(1 << np.arange(sample.shape[-1] - 1, -1, -1)),
                range=(0, 2**n_qubits),
                bins=2**n_qubits)[0] for sample in x
        ]

    def test_simulate_samples_inputs(self):
        """Make sure the sample op fails gracefully on bad inputs."""
        n_qubits = 5
        batch_size = 5
        num_samples = 10
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 1. Got rank 2'):
            # programs tensor has the wrong shape.
            noisy_samples_op.samples(util.convert_to_tensor([circuit_batch]),
                                     symbol_names, symbol_values_array,
                                     [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 1. Got rank 2'):
            # symbol_names tensor has the wrong shape.
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch),
                                     np.array([symbol_names]),
                                     symbol_values_array, [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 2. Got rank 3'):
            # symbol_values tensor has the wrong shape.
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch),
                                     symbol_names,
                                     np.array([symbol_values_array]),
                                     [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 2. Got rank 1'):
            # symbol_values tensor has the wrong shape 2.
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch),
                                     symbol_names, symbol_values_array[0],
                                     [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 1. Got rank 2'):
            # num_samples tensor has the wrong shape.
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch),
                                     symbol_names, symbol_values_array,
                                     [[num_samples]])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # programs tensor has the right type, but invalid value.
            noisy_samples_op.samples(['junk'] * batch_size, symbol_names,
                                     symbol_values_array, [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type, but invalid value.
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch),
                                     ['junk'], symbol_values_array,
                                     [num_samples])

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # programs tensor has the wrong type.
            noisy_samples_op.samples([1] * batch_size, symbol_names,
                                     symbol_values_array, [num_samples])

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # programs tensor has the wrong type.
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch), [1],
                                     symbol_values_array, [num_samples])

        with self.assertRaisesRegex(tf.errors.UnimplementedError,
                                    'Cast string to float is not supported'):
            # programs tensor has the wrong type.
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch),
                                     symbol_names, [['junk']] * batch_size,
                                     [num_samples])

        with self.assertRaisesRegex(Exception, 'junk'):
            # num_samples tensor has the wrong type.
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch),
                                     symbol_names, symbol_values_array,
                                     ['junk'])

        with self.assertRaisesRegex(TypeError, 'missing'):
            # too few tensors.
            # pylint: disable=no-value-for-parameter
            noisy_samples_op.samples(util.convert_to_tensor(circuit_batch),
                                     symbol_names, symbol_values_array)
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong symbol_values size.
            noisy_samples_op.samples(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[:int(batch_size * 0.5)], num_samples)

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
        qubits = cirq.LineQubit.range(n_qubits)

        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size, include_channels=noisy)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        n_samples = 10000
        op_samples = noisy_samples_op.samples(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array, [n_samples]).to_list()

        op_hists = self._compute_hists(op_samples, n_qubits)

        cirq_samples = batch_util.batch_sample(
            circuit_batch, resolver_batch, n_samples,
            cirq.DensityMatrixSimulator() if noisy else cirq.Simulator())

        cirq_hists = self._compute_hists(cirq_samples, n_qubits)
        tol = 1.5 if noisy else 1.0
        for a, b in zip(op_hists, cirq_hists):
            self.assertLess(stats.entropy(a + 1e-8, b + 1e-8), tol)

    @parameterized.parameters([{
        'channel': x
    } for x in util.get_supported_channels()])
    def test_single_channel(self, channel):
        """Individually test adding just a single channel type to circuits."""
        symbol_names = []
        batch_size = 3
        n_qubits = 5
        qubits = cirq.GridQubit.rect(1, n_qubits)

        circuit_batch, resolver_batch = \
            util.random_circuit_resolver_batch(
                qubits, batch_size, include_channels=False)

        for i in range(batch_size):
            circuit_batch[i] = circuit_batch[i] + channel.on_each(*qubits)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        n_samples = (2**n_qubits) * 1000

        op_samples = noisy_samples_op.samples(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array, [n_samples]).to_list()
        op_hists = self._compute_hists(op_samples, n_qubits)

        cirq_samples = batch_util.batch_sample(circuit_batch, resolver_batch,
                                               n_samples,
                                               cirq.DensityMatrixSimulator())
        cirq_hists = self._compute_hists(cirq_samples, n_qubits)

        for a, b in zip(op_hists, cirq_hists):
            self.assertLess(stats.entropy(a + 1e-8, b + 1e-8), 0.15)

    def test_correct_padding(self):
        """Test the variable sized circuits are properly padded."""
        symbol_names = []
        batch_size = 2
        n_qubits = 5
        qubits1 = cirq.GridQubit.rect(1, n_qubits)
        qubits2 = cirq.GridQubit.rect(1, n_qubits + 1)

        circuit_batch1, resolver_batch1 = \
            util.random_circuit_resolver_batch(
                qubits1, batch_size, include_channels=True)

        circuit_batch2, resolver_batch2 = \
            util.random_circuit_resolver_batch(
                qubits2, batch_size, include_channels=True)

        p1 = [[resolver[symbol]
               for symbol in symbol_names]
              for resolver in resolver_batch1]
        p2 = [[resolver[symbol]
               for symbol in symbol_names]
              for resolver in resolver_batch2]
        symbol_values_array = np.array(p1 + p2)

        n_samples = 10

        op_samples = noisy_samples_op.samples(
            util.convert_to_tensor(circuit_batch1 + circuit_batch2),
            symbol_names, symbol_values_array, [n_samples]).to_list()
        a_reps = np.asarray(op_samples[:2])
        b_reps = np.asarray(op_samples[2:])
        self.assertEqual(a_reps.shape, (2, 10, 5))
        self.assertEqual(b_reps.shape, (2, 10, 6))

    def test_correctness_empty(self):
        """Test the expectation for empty circuits."""
        empty_circuit = util.convert_to_tensor([cirq.Circuit()])
        empty_symbols = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        empty_values = tf.convert_to_tensor([[]])
        empty_n_samples = tf.convert_to_tensor([1], dtype=tf.int32)

        out = noisy_samples_op.samples(empty_circuit, empty_symbols,
                                       empty_values, empty_n_samples)

        expected = np.array([[[]]], dtype=np.int8)
        self.assertAllClose(out.to_tensor(), expected)

    def test_correctness_no_circuit(self):
        """Test the correctness with the empty tensor."""
        empty_circuit = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_symbols = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_values = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        empty_n_samples = tf.convert_to_tensor([1], dtype=tf.int32)

        out = noisy_samples_op.samples(empty_circuit, empty_symbols,
                                       empty_values, empty_n_samples)

        self.assertShapeEqual(np.zeros((0, 0, 0)), out.to_tensor())


if __name__ == "__main__":
    tf.test.main()
