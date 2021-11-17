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
import sympy
import tensorflow as tf

from absl.testing import parameterized
import cirq

from tensorflow_quantum.core.ops import tfq_utility_ops
from tensorflow_quantum.core.serialize import serializer
from tensorflow_quantum.python import util


class AppendCircuitOpTest(tf.test.TestCase, parameterized.TestCase):
    """Test the in-graph circuit append op."""

    def test_append_input_checking(self):
        """Check that the append op has correct input checking."""
        test_circuit = serializer.serialize_circuit(
            cirq.Circuit(cirq.X.on(cirq.GridQubit(0, 0)))).SerializeToString()
        with self.assertRaisesRegex(TypeError, 'Cannot convert \\[1\\]'):
            tfq_utility_ops.append_circuit([test_circuit], [1])
        with self.assertRaisesRegex(TypeError, 'Cannot convert \\[1\\]'):
            tfq_utility_ops.append_circuit([1], [test_circuit])
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            tfq_utility_ops.append_circuit(['wrong'], ['wrong'])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes.'):
            tfq_utility_ops.append_circuit([test_circuit],
                                           [test_circuit, test_circuit])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes.'):
            tfq_utility_ops.append_circuit([test_circuit, test_circuit],
                                           [test_circuit])
        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                'programs and programs_to_append must have matching sizes'):
            tfq_utility_ops.append_circuit([], [test_circuit])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1. Got rank 2'):
            tfq_utility_ops.append_circuit([[test_circuit, test_circuit]],
                                           [[test_circuit, test_circuit]])

        with self.assertRaisesRegex(TypeError,
                                    'missing 1 required positional argument'):
            # pylint: disable=no-value-for-parameter
            tfq_utility_ops.append_circuit([test_circuit])
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(TypeError,
                                    '2 positional arguments but 3 were given'):
            # pylint: disable=too-many-function-args
            tfq_utility_ops.append_circuit([test_circuit], [test_circuit],
                                           [test_circuit])
            # pylint: enable=too-many-function-args

        # These tests really just makes sure we can cast output
        res = tfq_utility_ops.append_circuit([], [])

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
        other_qubits = cirq.GridQubit.rect(2, max_n_bits) + [cirq.LineQubit(10)]

        base_circuits, _ = util.random_symbol_circuit_resolver_batch(
            qubits,
            symbols,
            n_circuits,
            include_scalars=True,
            include_channels=True)

        circuits_to_append, _ = util.random_symbol_circuit_resolver_batch(
            other_qubits,
            symbols,
            n_circuits,
            include_scalars=True,
            include_channels=True)

        serialized_base_circuits = util.convert_to_tensor(base_circuits)
        serialized_circuits_to_append = util.convert_to_tensor(
            circuits_to_append)

        tfq_results = tfq_utility_ops.append_circuit(
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
        """Check that the resolve parameters op has correct input checking."""
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
                                    'must be rank 1'):
            # programs tensor has the wrong shape (too many dims).
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'must be rank 1'):
            # programs tensor has the wrong shape (too few dims).
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch)[0], symbol_names,
                symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'must be rank 1'):
            # symbol_names tensor has the wrong shape (too many dims).
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'must be rank 1'):
            # symbol_names tensor has the wrong shape (too few dims).
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch), symbol_names[0],
                symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2'):
            # symbol_values tensor has the wrong shape (too many dims).
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2'):
            # symbol_values tensor has the wrong shape (too few dims).
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[0])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # programs tensor has the right type, but invalid value.
            tfq_utility_ops.resolve_parameters(['junk'] * batch_size,
                                               symbol_names,
                                               symbol_values_array)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # programs tensor has the wrong type.
            tfq_utility_ops.resolve_parameters([1] * batch_size, symbol_names,
                                               symbol_values_array)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch), [1], symbol_values_array)

        with self.assertRaisesRegex(tf.errors.UnimplementedError,
                                    'Cast string to float is not supported'):
            # symbol_values tensor has the wrong type.
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # too few tensors.
            # pylint: disable=no-value-for-parameter
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch), symbol_names)
            # pylint: enable=no-value-for-parameter

    def test_resolve_parameters_consistency_basic(self):
        """Compare tfq op to cirq resolving."""
        qubits = cirq.GridQubit.rect(1, 4)
        circuit = cirq.Circuit()
        symbols = []
        for n, q in enumerate(qubits):
            new_bit = sympy.Symbol("bit_{}".format(n))
            circuit += cirq.X(q)**new_bit
            symbols.append(new_bit)
        symbol_names = [str(s) for s in symbols]

        bitstring_list = [[0, 0, 0, 0], [0, 1, 0, 1], [1, 0, 1, 1]]
        circuit_list = []
        resolver_list = []
        for bitstring in bitstring_list:
            resolve_dict = {}
            for s, b in zip(symbols, bitstring):
                resolve_dict[s] = b
            resolver_list.append(cirq.ParamResolver(resolve_dict))
            circuit_list.append(circuit)

        test_resolved_circuits = util.from_tensor(
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_list), symbol_names,
                np.asarray(bitstring_list)))

        expected_resolved_circuits = []
        for circuit, resolver in zip(circuit_list, resolver_list):
            expected_resolved_circuits.append(
                cirq.resolve_parameters(circuit, resolver))

        for exp_c, test_c in zip(expected_resolved_circuits,
                                 test_resolved_circuits):
            self.assertAllEqual(exp_c, test_c)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'n_qubits': [3, 7],
                    'symbol_names': [['a'], ['a', 'b'],
                                     ['a', 'b', 'c', 'd', 'e']]
                })))
    def test_resolve_parameters_consistency(self, n_qubits, symbol_names):
        """Compare tfq op to cirq resolving for randomized circuits."""

        # Get random circuit batches
        qubits = cirq.GridQubit.rect(1, n_qubits)
        batch_size = 15
        n_moments = 15
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size,
                n_moments=n_moments,
                include_channels=True,
                include_scalars=True)

        # Remove one of the symbols from the resolvers
        symbol_names_partial = symbol_names[1:]
        symbol_values_array_partial = np.array(
            [[resolver[symbol]
              for symbol in symbol_names_partial]
             for resolver in resolver_batch])
        resolver_batch_partial = [
            cirq.ParamResolver(
                {symbol: resolver[symbol]
                 for symbol in symbol_names_partial})
            for resolver in resolver_batch
        ]

        # Resolve in two ways and compare results
        test_resolved_circuits = util.from_tensor(
            tfq_utility_ops.resolve_parameters(
                util.convert_to_tensor(circuit_batch), symbol_names_partial,
                symbol_values_array_partial))
        expected_resolved_circuits = []
        for circuit, resolver in zip(circuit_batch, resolver_batch_partial):
            expected_resolved_circuits.append(
                cirq.resolve_parameters(circuit, resolver))
        # TODO(zaqqwerty): Find a way to eliminate parsing.
        for test_c, exp_c in zip(test_resolved_circuits,
                                 expected_resolved_circuits):
            for test_m, exp_m in zip(test_c, exp_c):
                for test_o, exp_o in zip(test_m, exp_m):
                    self.assertTrue(util.gate_approx_eq(test_o.gate,
                                                        exp_o.gate))


if __name__ == '__main__':
    tf.test.main()
