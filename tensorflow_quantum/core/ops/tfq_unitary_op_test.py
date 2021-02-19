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
"""Tests that specifically target tfq_unitary_op."""
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq

from tensorflow_quantum.python import util
from tensorflow_quantum.core.ops import tfq_unitary_op


class UnitaryTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_calculate_unitary."""

    def test_calculate_unitary_inputs(self):
        """Make sure the unitary op fails gracefully on bad inputs."""
        unitary_op = tfq_unitary_op.get_unitary_op()
        n_qubits = 5
        batch_size = 5
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
                                    'programs must be rank 1'):
            # programs tensor has the wrong shape.
            unitary_op(util.convert_to_tensor([circuit_batch]), symbol_names,
                       symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1'):
            # symbol_names tensor has the wrong shape.
            unitary_op(util.convert_to_tensor(circuit_batch),
                       np.array([symbol_names]), symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2'):
            # symbol_values tensor has the wrong shape.
            unitary_op(util.convert_to_tensor(circuit_batch), symbol_names,
                       np.array([symbol_values_array]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2'):
            # symbol_values tensor has the wrong shape 2.
            unitary_op(util.convert_to_tensor(circuit_batch), symbol_names,
                       symbol_values_array[0])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # programs tensor has the right type, but invalid value.
            unitary_op(['junk'] * batch_size, symbol_names, symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type, but invalid value.
            unitary_op(util.convert_to_tensor(circuit_batch), ['junk'],
                       symbol_values_array)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # programs tensor has the wrong type.
            unitary_op([1] * batch_size, symbol_names, symbol_values_array)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            unitary_op(util.convert_to_tensor(circuit_batch), [1],
                       symbol_values_array)

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            unitary_op(util.convert_to_tensor(circuit_batch), symbol_names,
                       [['junk']] * batch_size)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # too few tensors.
            # pylint: disable=no-value-for-parameter
            unitary_op(util.convert_to_tensor(circuit_batch), symbol_names)
            # pylint: enable=no-value-for-parameter

        # TODO (mbbrough): determine if we should allow extra arguments ?
        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            unitary_op(util.convert_to_tensor(circuit_batch), symbol_names,
                       symbol_values_array, [])

    @parameterized.parameters([
        {
            'all_n_qubits': [2, 3]
        },
        {
            'all_n_qubits': [1, 5, 8]
        },
    ])
    def test_calculate_unitary_output_padding(self, all_n_qubits):
        """If calculate_unitary is asked to calculate matrices given circuits
        acting on different numbers of qubits, the op should return a tensor
        padded with zeros up to the size of the largest circuit."""
        unitary_op = tfq_unitary_op.get_unitary_op()
        circuit_batch = []
        for n_qubits in all_n_qubits:
            qubits = cirq.GridQubit.rect(1, n_qubits)
            circuit_batch += util.random_circuit_resolver_batch(qubits, 1)[0]

        tfq_results = unitary_op(util.convert_to_tensor(circuit_batch), [],
                                 [[]] * len(circuit_batch))

        results = [cirq.unitary(circuit) for circuit in circuit_batch]

        self.assertAllClose(tfq_results.to_list(), results, atol=1e-5)

    def test_calculate_unitary_empty(self):
        """Ensure calculate_unitary is consistent with empty circuits."""
        unitary_op = tfq_unitary_op.get_unitary_op()
        empty_u = cirq.unitary(cirq.Circuit())
        tfq_empty_u = unitary_op(util.convert_to_tensor([cirq.Circuit()]), [],
                                 [[]])

        self.assertAllClose(tfq_empty_u, [empty_u], atol=1e-5)  # wrap in batch.

    def test_calculate_unitary_no_circuit(self):
        """Ensure calculate_unitary is consistent with no circuits."""
        unitary_op = tfq_unitary_op.get_unitary_op()
        no_circuit = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_values = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        tfq_empty_u = unitary_op(no_circuit, [], empty_values)
        expected_shape = tf.TensorShape([0, None, None])
        self.assertEqual(tfq_empty_u.shape.as_list(), expected_shape.as_list())

    @parameterized.parameters([{
        'n_qubits': 6,
        'unitary_op': tfq_unitary_op.get_unitary_op(True)
    }, {
        'n_qubits': 7,
        'unitary_op': tfq_unitary_op.get_unitary_op(True)
    }, {
        'n_qubits': 6,
        'unitary_op': tfq_unitary_op.get_unitary_op(False)
    }, {
        'n_qubits': 7,
        'unitary_op': tfq_unitary_op.get_unitary_op(False)
    }])
    def test_calculate_unitary_consistency_symbol_free(self, n_qubits,
                                                       unitary_op):
        """Test calculate_unitary works without symbols."""
        unitary_op = tfq_unitary_op.get_unitary_op()
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, _ = util.random_circuit_resolver_batch(qubits, 25)

        tfq_results = unitary_op(util.convert_to_tensor(circuit_batch), [],
                                 [[]] * len(circuit_batch))

        results = [cirq.unitary(circuit) for circuit in circuit_batch]

        self.assertAllClose(tfq_results, results, atol=1e-5)

    @parameterized.parameters([{
        'n_qubits': 3,
        'unitary_op': tfq_unitary_op.get_unitary_op(True)
    }, {
        'n_qubits': 4,
        'unitary_op': tfq_unitary_op.get_unitary_op(True)
    }, {
        'n_qubits': 3,
        'unitary_op': tfq_unitary_op.get_unitary_op(False)
    }, {
        'n_qubits': 4,
        'unitary_op': tfq_unitary_op.get_unitary_op(False)
    }])
    def test_calculate_unitary_consistency(self, n_qubits, unitary_op):
        """Test that calculate_unitary works with symbols."""
        unitary_op = tfq_unitary_op.get_unitary_op()
        qubits = cirq.GridQubit.rect(1, n_qubits)
        symbols = ['alpha', 'beta', 'gamma']
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(qubits, symbols, 25)

        values = np.empty((len(circuit_batch), len(symbols)))
        for i in range(len(circuit_batch)):
            for j in range(len(symbols)):
                values[i][j] = resolver_batch[i][symbols[j]]

        tfq_results = unitary_op(util.convert_to_tensor(circuit_batch), symbols,
                                 values)

        results = []
        for circuit, resolver in zip(circuit_batch, resolver_batch):
            resolved_circuit = cirq.resolve_parameters(circuit, resolver)
            results.append(cirq.unitary(resolved_circuit))

        self.assertAllClose(tfq_results, results, atol=1e-5)


if __name__ == "__main__":
    tf.test.main()
