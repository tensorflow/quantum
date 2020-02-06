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

from absl.testing import parameterized
import cirq
import numpy as np
import tensorflow as tf

from random_clifford_circuit import random_clifford_circuit


class RandomCliffordCircuitTest(parameterized.TestCase, tf.test.TestCase):
    """Test the Random Clifford Circuit model."""

    def test_random_clifford_circuit_inputs(self):
        """Test for input validation."""
        qubits = cirq.GridQubit.rect(3, 2)
        n_moments = 10
        op_density = 0.9
        with self.assertRaisesRegex(TypeError, 'RandomState'):
            random_clifford_circuit(qubits,
                                    n_moments,
                                    op_density,
                                    random_state="string")
        with self.assertRaisesRegex(TypeError, 'RandomState'):
            random_clifford_circuit(qubits,
                                    n_moments,
                                    op_density,
                                    random_state=[1, 2, 3])

        with self.assertRaisesRegex(TypeError, 'iterable'):
            random_clifford_circuit(cirq.GridQubit(0, 0),
                                    n_moments,
                                    op_density,
                                    random_state=None)
        with self.assertRaisesRegex(TypeError, 'iterable'):
            random_clifford_circuit(cirq.LineQubit(0),
                                    n_moments,
                                    op_density,
                                    random_state=None)

        with self.assertRaisesRegex(ValueError, '2 qubits'):
            random_clifford_circuit([cirq.GridQubit(0, 0)],
                                    n_moments,
                                    op_density,
                                    random_state=None)

    def test_reproducible_circuit(self):
        """Test that circuits are reproducible via random state seeding."""
        qubits = cirq.GridQubit.rect(4, 2)
        n_moments = 13
        op_density = 0.8
        rng = np.random.RandomState(4902796)

        c1 = cirq.Circuit(*random_clifford_circuit(
            qubits, n_moments, op_density, random_state=rng))

        rng = np.random.RandomState(4902796)
        c2 = cirq.Circuit(*random_clifford_circuit(
            qubits, n_moments, op_density, random_state=rng))
        self.assertEqual(c1, c2)

    def test_only_cliffords(self):
        """Test that the circuit contains only Cliffords."""
        qubits = cirq.GridQubit.rect(4, 2)
        n_moments = 10
        op_density = 0.9
        circuit = cirq.Circuit(
            *random_clifford_circuit(qubits, n_moments, op_density))
        cliffords = set(
            [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.CZ, cirq.CNOT, cirq.SWAP])
        non_id_gates = [op.gate for op in circuit.all_operations()]
        self.assertTrue(set(non_id_gates).issubset(cliffords))

    @parameterized.parameters([5, 7, 11, 20])
    def test_random_clifford_circuit_depth(self, n_moments):
        """Test that the circuit has the number of moments requested."""
        qubits = cirq.GridQubit.rect(3, 2)
        op_density = 0.9
        circuit = cirq.Circuit(
            *random_clifford_circuit(qubits, n_moments, op_density))
        self.assertEqual(len(circuit), n_moments)


if __name__ == "__main__":
    tf.test.main()
