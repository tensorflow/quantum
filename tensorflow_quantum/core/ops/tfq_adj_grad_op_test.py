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
# Remove PYTHONPATH collisions for protobuf.
import sys
new_path = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = new_path

import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq
import sympy

from tensorflow_quantum.python import util
from tensorflow_quantum.core.ops import tfq_adj_grad_op


class ADJGradTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_calculate_unitary."""

    def test_adj_grad_inputs(self):
        """Make sure that the expectation op fails gracefully on bad inputs."""
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

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)
        upstream_grads = np.ones((batch_size, len(symbol_names)))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1'):
            # Circuit tensor has too many dimensions.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1.'):
            # symbol_names tensor has too many dimensions.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too many dimensions.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(np.array([symbol_values_array])),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too few dimensions.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array[0]),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too few dimensions.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor(list(pauli_sums)),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too many dimensions.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[[x]] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # circuit tensor has the right type but invalid values.
            tfq_adj_grad_op.tfq_adj_grad(
                ['junk'] * batch_size, symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type but invalid values.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), ['junk'],
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'qubits not found in circuit'):
            # pauli_sums tensor has the right type but invalid values.
            new_qubits = [cirq.GridQubit(5, 5), cirq.GridQubit(9, 9)]
            new_pauli_sums = util.random_pauli_sums(new_qubits, 2, batch_size)
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in new_pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # pauli_sums tensor has the right type but invalid values 2.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                [['junk']] * batch_size, tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # circuits tensor has the wrong type.
            tfq_adj_grad_op.tfq_adj_grad(
                [1.0] * batch_size, symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), [0.1234],
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # pauli_sums tensor has the wrong type.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array), [[1.0]] * batch_size,
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(TypeError, 'missing'):
            # we are missing an argument.
            # pylint: disable=no-value-for-parameter
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                tf.convert_to_tensor(upstream_grads))
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads), [])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong op size.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor([cirq.Circuit()]), symbol_names,
                symbol_values_array.astype(np.float64),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='rank 2'):
            # wrong grad shape.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor([upstream_grads]))

        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                expected_regex='gradients and circuits do not match'):
            # wrong grad batch size.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor([[0 for i in range(len(symbol_names))]]))

        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                expected_regex='gradients and pauli sum dimension do not match'
        ):
            # wrong grad inner size.
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor(circuit_batch), symbol_names,
                tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor([[0, 0] for _ in range(len(circuit_batch))
                                     ]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='cirq.Channel'):
            # attempting to use noisy circuit.
            noisy_circuit = cirq.Circuit(cirq.depolarize(0.3).on_each(*qubits))
            tfq_adj_grad_op.tfq_adj_grad(
                util.convert_to_tensor([noisy_circuit for _ in circuit_batch]),
                symbol_names, tf.convert_to_tensor(symbol_values_array),
                util.convert_to_tensor([[x] for x in pauli_sums]),
                tf.convert_to_tensor(upstream_grads))

    def test_calculate_adj_grad_empty(self):
        """Verify that the empty case is handled gracefully."""
        out = tfq_adj_grad_op.tfq_adj_grad(
            util.convert_to_tensor([cirq.Circuit()]),
            tf.convert_to_tensor([], dtype=tf.dtypes.string),
            tf.convert_to_tensor([[]]),
            tf.convert_to_tensor([[]], dtype=tf.dtypes.string),
            tf.convert_to_tensor([[]]))
        self.assertShapeEqual(np.zeros((1, 0)), out)

    def test_calculate_adj_grad_no_circuit(self):
        """Verify that the no circuit case is handled gracefully."""
        out = tfq_adj_grad_op.tfq_adj_grad(
            tf.raw_ops.Empty(shape=(0,), dtype=tf.string),
            tf.raw_ops.Empty(shape=(0,), dtype=tf.string),
            tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32),
            tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string),
            tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32),
        )
        self.assertShapeEqual(np.zeros((0, 0)), out)

    def test_calculate_adj_grad_simple_case(self):
        """Make sure that adjoint gradient works on simple input case."""
        n_qubits = 2
        batch_size = 1
        symbol_names = ['alpha', 'beta']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
        [cirq.Circuit(cirq.X(qubits[0]) ** sympy.Symbol('alpha'),
            cirq.Y(qubits[1]) ** sympy.Symbol('beta'),
            cirq.CNOT(qubits[0], qubits[1]))], [{'alpha': 0.123, 'beta': 0.456}]

        op_batch = [
            [cirq.Z(qubits[0]), cirq.X(qubits[1])] for _ in range(batch_size)
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        prev_grads = tf.ones([batch_size, len(symbol_names)])

        out = tfq_adj_grad_op.tfq_adj_grad(
            util.convert_to_tensor(circuit_batch),
            tf.convert_to_tensor(symbol_names),
            tf.convert_to_tensor(symbol_values_array),
            util.convert_to_tensor(op_batch), prev_grads)

        self.assertAllClose(out, np.array([[-1.18392, 0.43281]]), atol=1e-3)

    def test_calculate_adj_grad_simple_case2(self):
        """Make sure the adjoint gradient works on another simple input case."""
        n_qubits = 2
        batch_size = 1
        symbol_names = ['alpha', 'beta', 'gamma']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
        [cirq.Circuit(cirq.X(qubits[0]) ** sympy.Symbol('alpha'),
            cirq.Y(qubits[1]) ** sympy.Symbol('beta'),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.FSimGate(sympy.Symbol('gamma'), 0.5)(qubits[0], qubits[1]))
        ], [{'alpha': 0.123, 'beta': 0.456, 'gamma': 0.789}]

        op_batch = [
            [cirq.Z(qubits[0]), cirq.X(qubits[1])] for _ in range(batch_size)
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        prev_grads = tf.ones([batch_size, len(op_batch[0])])

        out = tfq_adj_grad_op.tfq_adj_grad(
            util.convert_to_tensor(circuit_batch),
            tf.convert_to_tensor(symbol_names),
            tf.convert_to_tensor(symbol_values_array),
            util.convert_to_tensor(op_batch), prev_grads)

        self.assertAllClose(out,
                            np.array([[-2.100, -1.7412, -1.5120]]),
                            atol=1e-3)

    def test_calculate_adj_grad_simple_case_shared(self):
        """Make sure the adjoint gradient works on a shared symbol gate."""
        n_qubits = 2
        batch_size = 1
        symbol_names = ['alpha', 'beta', 'gamma']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
        [cirq.Circuit(cirq.X(qubits[0]) ** sympy.Symbol('alpha'),
            cirq.Y(qubits[1]) ** sympy.Symbol('beta'),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.FSimGate(
                sympy.Symbol('gamma'),
                sympy.Symbol('gamma'))(qubits[0], qubits[1]))
        ], [{'alpha': 0.123, 'beta': 0.456, 'gamma': 0.789}]

        op_batch = [
            [cirq.Z(qubits[0]), cirq.X(qubits[1])] for _ in range(batch_size)
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        prev_grads = tf.ones([batch_size, len(op_batch[0])])

        out = tfq_adj_grad_op.tfq_adj_grad(
            util.convert_to_tensor(circuit_batch),
            tf.convert_to_tensor(symbol_names),
            tf.convert_to_tensor(symbol_values_array),
            util.convert_to_tensor(op_batch), prev_grads)

        self.assertAllClose(out,
                            np.array([[-2.3484, -1.7532, -1.64264]]),
                            atol=1e-3)

    def test_calculate_adj_grad_simple_case_single(self):
        """Make sure the adjoint gradient works on a one symbol for all gate."""
        n_qubits = 2
        batch_size = 1
        symbol_names = ['alpha', 'beta', 'gamma']
        qubits = cirq.LineQubit.range(n_qubits)
        circuit_batch, resolver_batch = \
        [cirq.Circuit(cirq.X(qubits[0]) ** sympy.Symbol('alpha'),
            cirq.Y(qubits[1]) ** sympy.Symbol('alpha'),
            cirq.CNOT(qubits[0], qubits[1]),
            cirq.FSimGate(
                -0.56,
                sympy.Symbol('alpha'))(qubits[0], qubits[1]))
        ], [{'alpha': 0.123, 'beta': 0.456, 'gamma': 0.789}]

        op_batch = [
            [cirq.Z(qubits[0]), cirq.X(qubits[1])] for _ in range(batch_size)
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        prev_grads = tf.ones([batch_size, len(op_batch[0])])

        out = tfq_adj_grad_op.tfq_adj_grad(
            util.convert_to_tensor(circuit_batch),
            tf.convert_to_tensor(symbol_names),
            tf.convert_to_tensor(symbol_values_array),
            util.convert_to_tensor(op_batch), prev_grads)

        self.assertAllClose(out, np.array([[1.2993, 0, 0]]), atol=1e-3)


if __name__ == "__main__":
    tf.test.main()
