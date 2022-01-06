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
"""Tests that specifically target simulate_mps."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import numpy as np
import tensorflow as tf
import cirq
import cirq_google
import sympy

from tensorflow_quantum.core.ops.math_ops import simulate_mps
from tensorflow_quantum.python import util


class SimulateMPS1DExpectationTest(tf.test.TestCase):
    """Tests mps_1d_expectation."""

    def test_simulate_mps_1d_expectation_inputs(self):
        """Makes sure that the op fails gracefully on bad inputs."""
        n_qubits = 5
        batch_size = 5
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch = [
            cirq.Circuit(
                cirq.X(qubits[0])**sympy.Symbol(symbol_names[0]),
                cirq.Z(qubits[1]),
                cirq.CNOT(qubits[2], qubits[3]),
                cirq.Y(qubits[4])**sympy.Symbol(symbol_names[0]),
            ) for _ in range(batch_size)
        ]
        resolver_batch = [{symbol_names[0]: 0.123} for _ in range(batch_size)]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1'):
            # Circuit tensor has too many dimensions.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1.'):
            # symbol_names tensor has too many dimensions.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too many dimensions.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]),
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too few dimensions.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[0],
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too few dimensions.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, util.convert_to_tensor(list(pauli_sums)))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too many dimensions.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[[x]] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # circuit tensor has the right type but invalid values.
            simulate_mps.mps_1d_expectation(
                ['junk'] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type but invalid values.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), ['junk'],
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'qubits not found in circuit'):
            # pauli_sums tensor has the right type but invalid values.
            new_qubits = [cirq.GridQubit(5, 5), cirq.GridQubit(9, 9)]
            new_pauli_sums = util.random_pauli_sums(new_qubits, 2, batch_size)
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in new_pauli_sums]))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # circuits tensor has the wrong type.
            simulate_mps.mps_1d_expectation(
                [1.0] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), [0.1234],
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # pauli_sums tensor has the wrong type.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [[1.0]] * batch_size)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # we are missing an argument.
            # pylint: disable=no-value-for-parameter
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array)
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'at least minimum 4'):
            # pylint: disable=too-many-function-args
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), 1)

        with self.assertRaisesRegex(TypeError, 'Expected int'):
            # bond_dim should be int.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), [])

        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), 1, [])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong op size.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums
                                       ][:int(batch_size * 0.5)]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong symbol_values size.
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[:int(batch_size * 0.5)],
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='cirq.Channel'):
            # attempting to use noisy circuit.
            noisy_circuit = cirq.Circuit(cirq.depolarize(0.3).on_each(*qubits))
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor([noisy_circuit for _ in pauli_sums]),
                symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='not in 1D topology'):
            # attempting to use a circuit not in 1D topology
            # 0--1--2--3
            #        \-4
            circuit_not_1d = cirq.Circuit(
                cirq.X(qubits[0])**sympy.Symbol(symbol_names[0]),
                cirq.Z(qubits[1])**sympy.Symbol(symbol_names[0]),
                cirq.CNOT(qubits[2], qubits[3]),
                cirq.CNOT(qubits[2], qubits[4]),
            )
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor([circuit_not_1d for _ in pauli_sums]),
                symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='not in 1D topology'):
            # attempting to use a circuit in 1D topology, which looks in 2D.
            # 0--1
            #  \-2-\
            #    3--4  == 1--0--2--4--3
            circuit_not_1d = cirq.Circuit(
                cirq.CNOT(qubits[0], qubits[1]),
                cirq.CNOT(qubits[0], qubits[2]),
                cirq.CNOT(qubits[2], qubits[4]),
                cirq.CNOT(qubits[3], qubits[4]),
            )
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor([circuit_not_1d for _ in pauli_sums]),
                symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='Found: 3 qubit gate'):
            # attempting to use 3 qubit gate
            three_qb_circuit = cirq.Circuit(
                cirq.ISWAP(qubits[0], qubits[1]).controlled_by(qubits[2]),
                cirq.X.on_each(*qubits))
            simulate_mps.mps_1d_expectation(
                util.convert_to_tensor([three_qb_circuit for _ in pauli_sums]),
                symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        res = simulate_mps.mps_1d_expectation(
            util.convert_to_tensor([cirq.Circuit() for _ in pauli_sums]),
            symbol_names, symbol_values_array.astype(np.float64),
            util.convert_to_tensor([[x] for x in pauli_sums]))
        self.assertDTypeEqual(res, np.float32)

    def test_simulate_mps_1d_expectation_simple(self):
        """Makes sure that the op shows the same result with Cirq."""
        n_qubits = 5
        batch_size = 5
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch = [
            cirq.Circuit(
                cirq.X(qubits[0])**sympy.Symbol(symbol_names[0]),
                cirq.Z(qubits[1]),
                cirq.CNOT(qubits[2], qubits[3]),
                cirq.Y(qubits[4])**sympy.Symbol(symbol_names[0]),
            ) for _ in range(batch_size)
        ]
        resolver_batch = [{symbol_names[0]: 0.123} for _ in range(batch_size)]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = [
            cirq.Z(qubits[0]) * cirq.X(qubits[4]) for _ in range(batch_size)
        ]

        cirq_result = [
            cirq.Simulator().simulate_expectation_values(c, p, r)
            for c, p, r in zip(circuit_batch, pauli_sums, resolver_batch)
        ]
        # Default bond_dim=4
        mps_result = simulate_mps.mps_1d_expectation(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array,
            util.convert_to_tensor([[x] for x in pauli_sums]))
        # Expected value of 0.349...
        self.assertAllClose(mps_result, cirq_result)

    def _make_1d_circuit(self, qubits, depth):
        """Create a 1d ladder circuit."""
        even_pairs = list(zip(qubits[::2], qubits[1::2]))
        odd_pairs = list(zip(qubits[1::2], qubits[2::2]))
        ret = cirq.Circuit()

        for _ in range(depth):
            # return ret
            ret += [(cirq.Y(q)**np.random.random()) for q in qubits]
            ret += [
                cirq_google.SycamoreGate()(q0, q1)**np.random.random()
                for q0, q1 in even_pairs
            ]
            ret += [(cirq.Y(q)**np.random.random()) for q in qubits]
            ret += [
                cirq_google.SycamoreGate()(q1, q0)**np.random.random()
                for q0, q1 in odd_pairs
            ]

        return ret

    def test_complex_equality(self):
        """Check moderate sized 1d random circuits."""
        batch_size = 10
        qubits = cirq.GridQubit.rect(1, 8)
        circuit_batch = [
            self._make_1d_circuit(qubits, 3) for _ in range(batch_size)
        ]

        pauli_sums = [[
            cirq.Z(qubits[0]),
            cirq.Z(qubits[-1]),
            cirq.Z(qubits[0]) * cirq.Z(qubits[-1]),
            cirq.Z(qubits[0]) + cirq.Z(qubits[-1])
        ] for _ in range(batch_size)]
        symbol_names = []
        resolver_batch = [{} for _ in range(batch_size)]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        cirq_result = [
            cirq.Simulator().simulate_expectation_values(c, p, r)
            for c, p, r in zip(circuit_batch, pauli_sums, resolver_batch)
        ]
        mps_result = simulate_mps.mps_1d_expectation(
            util.convert_to_tensor(circuit_batch),
            symbol_names,
            symbol_values_array,
            util.convert_to_tensor(pauli_sums),
            bond_dim=32)
        self.assertAllClose(mps_result, cirq_result, atol=1e-5)

    def test_correctness_empty(self):
        """Tests the mps op with empty circuits."""

        empty_circuit = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_symbols = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_values = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        empty_paulis = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)

        out = simulate_mps.mps_1d_expectation(empty_circuit, empty_symbols,
                                              empty_values, empty_paulis, 32)

        self.assertShapeEqual(np.zeros((0, 0)), out)


if __name__ == "__main__":
    tf.test.main()
