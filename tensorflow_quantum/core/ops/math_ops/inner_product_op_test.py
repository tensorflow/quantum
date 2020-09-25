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
"""Tests that specifically target tfq_simulate_ops."""
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops.math_ops import inner_product_op
from tensorflow_quantum.python import util


class InnerProductTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_inner_product."""

    def test_inner_product_inputs(self):
        """Make sure that inner_product fails gracefully on bad inputs."""
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

        other_batch = [
            util.random_circuit_resolver_batch(qubits, 3)[0]
            for i in range(batch_size)
        ]

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1'):
            # Circuit tensor has too many dimensions.
            inner_product_op.inner_product(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                symbol_values_array, util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1.'):
            # symbol_names tensor has too many dimensions.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                symbol_values_array, util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too many dimensions.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]),
                util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too few dimensions.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[0], util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'other_programs must be rank 2.'):
            # other_programs tensor has too few dimensions.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, util.convert_to_tensor(circuit_batch))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'other_programs must be rank 2.'):
            # pauli_sums tensor has too many dimensions.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in other_batch]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # circuit tensor has the right type but invalid values.
            inner_product_op.inner_product(['junk'] * batch_size, symbol_names,
                                           symbol_values_array,
                                           util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type but invalid values.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), ['junk'],
                symbol_values_array, util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'not found in reference circuit'):
            # other_programs tensor has the right type but operates on
            # qubits that the reference ciruit doesn't have.
            new_qubits = [cirq.GridQubit(5, 5), cirq.GridQubit(9, 9)]
            new_circuits, _ = util.random_circuit_resolver_batch(
                new_qubits, batch_size)
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in new_circuits]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'not found in paired circuit'):
            # other_programs tensor has the right type but operates on
            # qubits that the reference ciruit doesn't have.
            new_qubits = cirq.GridQubit.rect(1, n_qubits - 1)
            new_circuits, _ = util.random_circuit_resolver_batch(
                new_qubits, batch_size)
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in new_circuits]))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # circuits tensor has the wrong type.
            inner_product_op.inner_product([1.0] * batch_size, symbol_names,
                                           symbol_values_array,
                                           util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), [0.1234],
                symbol_values_array, util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size, util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # other_programs tensor has the wrong type.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [[1.0]] * batch_size)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # we are missing an argument.
            # pylint: disable=no-value-for-parameter
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array)
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, util.convert_to_tensor(other_batch), [])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # batch programs has wrong batch size.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor(other_batch[:int(batch_size * 0.5)]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # batch programs has wrong batch size.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[::int(batch_size * 0.5)],
                util.convert_to_tensor(other_batch))

        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                expected_regex='Found symbols in other_programs'):
            # other_programs has symbols.
            inner_product_op.inner_product(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in circuit_batch]))

        res = inner_product_op.inner_product(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array.astype(np.float64),
            util.convert_to_tensor(other_batch))
        self.assertDTypeEqual(res, np.complex64)

    @parameterized.parameters([
        {
            'n_qubits': 5,
            'batch_size': 10,
            'inner_dim_size': 1
        },
        {
            'n_qubits': 10,
            'batch_size': 10,
            'inner_dim_size': 2
        },
        {
            'n_qubits': 5,
            'batch_size': 10,
            'inner_dim_size': 5
        },
    ])
    def test_correctness_with_symbols(self, n_qubits, batch_size,
                                      inner_dim_size):
        """Test that inner_product works with symbols."""
        symbol_names = ['alpha', 'beta', 'gamma']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        other_batch = [
            util.random_circuit_resolver_batch(qubits, inner_dim_size)[0]
            for i in range(batch_size)
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names = tf.convert_to_tensor(symbol_names,
                                            dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor(symbol_values_array)

        out = inner_product_op.inner_product(programs, symbol_names,
                                             symbol_values, other_programs)

        out_arr = np.empty((batch_size, inner_dim_size), dtype=np.complex64)
        for i in range(batch_size):
            final_circuit = cirq.resolve_parameters(circuit_batch[i],
                                                    resolver_batch[i])
            final_wf = cirq.final_wavefunction(final_circuit)
            for j in range(inner_dim_size):
                internal_wf = cirq.final_wavefunction(other_batch[i][j])
                out_arr[i][j] = np.vdot(final_wf, internal_wf)

        self.assertAllClose(out, out_arr)

    @parameterized.parameters([
        {
            'n_qubits': 5,
            'batch_size': 10,
            'inner_dim_size': 1
        },
        {
            'n_qubits': 10,
            'batch_size': 10,
            'inner_dim_size': 2
        },
        {
            'n_qubits': 5,
            'batch_size': 10,
            'inner_dim_size': 5
        },
    ])
    def test_correctness_without_symbols(self, n_qubits, batch_size,
                                         inner_dim_size):
        """Test that inner_product works with symbols."""
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, _ = \
            util.random_circuit_resolver_batch(
                qubits, batch_size)

        other_batch = [
            util.random_circuit_resolver_batch(qubits, inner_dim_size)[0]
            for i in range(batch_size)
        ]

        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor([[] for _ in range(batch_size)])

        out = inner_product_op.inner_product(programs, symbol_names,
                                             symbol_values, other_programs)

        out_arr = np.empty((batch_size, inner_dim_size), dtype=np.complex64)
        for i in range(batch_size):
            final_wf = cirq.final_wavefunction(circuit_batch[i])
            for j in range(inner_dim_size):
                internal_wf = cirq.final_wavefunction(other_batch[i][j])
                out_arr[i][j] = np.vdot(final_wf, internal_wf)

        self.assertAllClose(out, out_arr)

    def test_correctness_empty(self):
        """Test the inner product between two empty circuits."""

        empty_cicuit = util.convert_to_tensor([cirq.Circuit()])
        empty_symbols = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        empty_values = tf.convert_to_tensor([[]])
        other_program = util.convert_to_tensor([[cirq.Circuit()]])

        out = inner_product_op.inner_product(empty_cicuit, empty_symbols,
                                             empty_values, other_program)
        expected = np.array([[1.0]], dtype=np.complex64)
        self.assertAllClose(out, expected)


if __name__ == "__main__":
    tf.test.main()
