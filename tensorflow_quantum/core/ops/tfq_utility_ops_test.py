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
"""Tests for tfq utility ops."""
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import cirq

from tensorflow_quantum.core.ops import tfq_utility_ops
from tensorflow_quantum.core.serialize import serializer
from tensorflow_quantum.python import util


class CircuitAppendOpTest(tf.test.TestCase, parameterized.TestCase):
    """Test the in-graph circuit append op."""

    def test_append_input_checking(self):
        """Check that the append op has correct input checking."""
        test_circuit = serializer.serialize_circuit(
            cirq.Circuit(cirq.X.on(cirq.GridQubit(0, 0)))).SerializeToString()
        with self.assertRaisesRegex(TypeError, 'Cannot convert \\[1\\]'):
            tfq_utility_ops.tfq_append_circuit([test_circuit], [1])
        with self.assertRaisesRegex(TypeError, 'Cannot convert \\[1\\]'):
            tfq_utility_ops.tfq_append_circuit([1], [test_circuit])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            tfq_utility_ops.tfq_append_circuit(['wrong'], ['wrong'])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes.'):
            tfq_utility_ops.tfq_append_circuit([test_circuit],
                                               [test_circuit, test_circuit])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes.'):
            tfq_utility_ops.tfq_append_circuit([test_circuit, test_circuit],
                                               [test_circuit])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes'):
            tfq_utility_ops.tfq_append_circuit([], [test_circuit])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1. Got rank 2'):
            tfq_utility_ops.tfq_append_circuit([[test_circuit, test_circuit]],
                                               [[test_circuit, test_circuit]])

        with self.assertRaisesRegex(TypeError,
                                    'missing 1 required positional argument'):
            # pylint: disable=no-value-for-parameter
            tfq_utility_ops.tfq_append_circuit([test_circuit])
            # pylint: enable=no-value-for-parameter

        # TODO (mbbrough): should this line work or no. what is the TF
        # standard here ?
        tfq_utility_ops.tfq_append_circuit([test_circuit], [test_circuit],
                                           [test_circuit])

        # These tests really just makes sure we can cast output
        res = tfq_utility_ops.tfq_append_circuit([], [])

        self.assertDTypeEqual(res.numpy().astype(np.str), np.dtype('<U1'))

    @parameterized.parameters([{
        'max_n_bits': 20,
        'symbols': ['a', 'b', 'c'],
        'n_circuits': 5
    }])
    def test_append_circuit(self, max_n_bits, symbols, n_circuits):
        """Generate a bunch of circuits of different lengths acting on different
        numbers of qubits and append them using our op, checking that results
        are consistant with the native cirq method.
        """
        base_circuits = []
        circuits_to_append = []
        qubits = cirq.GridQubit.rect(1, max_n_bits)
        other_qubits = cirq.GridQubit.rect(2, max_n_bits)

        base_circuits, _ = util.random_symbol_circuit_resolver_batch(
            qubits, symbols, n_circuits, include_scalars=False)

        circuits_to_append, _ = util.random_symbol_circuit_resolver_batch(
            other_qubits, symbols, n_circuits, include_scalars=False)

        serialized_base_circuits = util.convert_to_tensor(base_circuits)
        serialized_circuits_to_append = util.convert_to_tensor(
            circuits_to_append)

        tfq_results = tfq_utility_ops.tfq_append_circuit(
            serialized_base_circuits, serialized_circuits_to_append)

        tfq_results = util.from_tensor(tfq_results)
        cirq_results = [
            a + b for a, b in zip(base_circuits, circuits_to_append)
        ]
        self.assertAllEqual(util.convert_to_tensor(tfq_results),
                            util.convert_to_tensor(cirq_results))

    @parameterized.parameters([{
        'padded_array': [[[1, 0, 0, 0], [1, 1, 1, 1]],
                         [[1, 1, -2, -2], [0, 0, -2, -2]],
                         [[0, 0, -2, -2], [1, 0, -2, -2]]]
    }, {
        'padded_array': [[[0, 0, 0, 0], [1, 1, 1, 1]]]
    }, {
        'padded_array': [[[1, 1, -2, -2], [0, 1, -2, -2], [0, 0, -2, -2]]]
    }])
    def test_padded_to_ragged(self, padded_array):
        """Test for padded_to_ragged utility."""
        mask = np.where(np.array(padded_array) > -1, True, False)
        expected = tf.ragged.boolean_mask(padded_array, mask)
        actual = tfq_utility_ops.padded_to_ragged(
            np.array(padded_array, dtype=float))
        self.assertAllEqual(expected, actual)


class ResolveParametersOpTest(tf.test.TestCase, parameterized.TestCase):
    """Test the in-graph parameter resolving op."""

    def test_resolve_parameters_input_checking(self):
        """Check that the append op has correct input checking."""
        test_circuit = serializer.serialize_circuit(
            cirq.Circuit(cirq.X.on(cirq.GridQubit(0, 0)))).SerializeToString()
        with self.assertRaisesRegex(TypeError, 'Cannot convert \\[1\\]'):
            tfq_utility_ops.tfq_append_circuit([test_circuit], [1])
        with self.assertRaisesRegex(TypeError, 'Cannot convert \\[1\\]'):
            tfq_utility_ops.tfq_append_circuit([1], [test_circuit])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            tfq_utility_ops.tfq_append_circuit(['wrong'], ['wrong'])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes.'):
            tfq_utility_ops.tfq_append_circuit([test_circuit],
                                               [test_circuit, test_circuit])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes.'):
            tfq_utility_ops.tfq_append_circuit([test_circuit, test_circuit],
                                               [test_circuit])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes'):
            tfq_utility_ops.tfq_append_circuit([], [test_circuit])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1. Got rank 2'):
            tfq_utility_ops.tfq_append_circuit([[test_circuit, test_circuit]],
                                               [[test_circuit, test_circuit]])

        with self.assertRaisesRegex(TypeError,
                                    'missing 1 required positional argument'):
            # pylint: disable=no-value-for-parameter
            tfq_utility_ops.tfq_append_circuit([test_circuit])
            # pylint: enable=no-value-for-parameter

        # TODO (mbbrough): should this line work or no. what is the TF
        # standard here ?
        tfq_utility_ops.tfq_append_circuit([test_circuit], [test_circuit],
                                           [test_circuit])

        # These tests really just makes sure we can cast output
        res = tfq_utility_ops.tfq_append_circuit([], [])

        self.assertDTypeEqual(res.numpy().astype(np.str), np.dtype('<U1'))

    @parameterized.parameters([{
        'max_n_bits': 20,
        'symbols': ['a', 'b', 'c'],
        'n_circuits': 5
    }])
    def test_append_circuit(self, max_n_bits, symbols, n_circuits):
        """Generate a bunch of circuits of different lengths acting on different
        numbers of qubits and append them using our op, checking that results
        are consistant with the native cirq method.
        """
        base_circuits = []
        circuits_to_append = []
        qubits = cirq.GridQubit.rect(1, max_n_bits)
        other_qubits = cirq.GridQubit.rect(2, max_n_bits)

        base_circuits, _ = util.random_symbol_circuit_resolver_batch(
            qubits, symbols, n_circuits, include_scalars=False)

        circuits_to_append, _ = util.random_symbol_circuit_resolver_batch(
            other_qubits, symbols, n_circuits, include_scalars=False)

        serialized_base_circuits = util.convert_to_tensor(base_circuits)
        serialized_circuits_to_append = util.convert_to_tensor(
            circuits_to_append)

        tfq_results = tfq_utility_ops.tfq_append_circuit(
            serialized_base_circuits, serialized_circuits_to_append)

        tfq_results = util.from_tensor(tfq_results)
        cirq_results = [
            a + b for a, b in zip(base_circuits, circuits_to_append)
        ]
        self.assertAllEqual(util.convert_to_tensor(tfq_results),
                            util.convert_to_tensor(cirq_results))

    @parameterized.parameters([{
        'padded_array': [[[1, 0, 0, 0], [1, 1, 1, 1]],
                         [[1, 1, -2, -2], [0, 0, -2, -2]],
                         [[0, 0, -2, -2], [1, 0, -2, -2]]]
    }, {
        'padded_array': [[[0, 0, 0, 0], [1, 1, 1, 1]]]
    }, {
        'padded_array': [[[1, 1, -2, -2], [0, 1, -2, -2], [0, 0, -2, -2]]]
    }])
    def test_padded_to_ragged(self, padded_array):
        """Test for padded_to_ragged utility."""
        mask = np.where(np.array(padded_array) > -1, True, False)
        expected = tf.ragged.boolean_mask(padded_array, mask)
        actual = tfq_utility_ops.padded_to_ragged(
            np.array(padded_array, dtype=float))
        self.assertAllEqual(expected, actual)


if __name__ == '__main__':
    tf.test.main()
