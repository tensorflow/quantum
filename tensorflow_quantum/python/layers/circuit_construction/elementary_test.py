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
"""Tests for the elementary layers."""
import tensorflow as tf
import cirq
import sympy

from tensorflow_quantum.python import util
from tensorflow_quantum.python.layers.circuit_construction import elementary


class AddCircuitTest(tf.test.TestCase):
    """Test AddCircuit works with various inputs."""

    def test_addcircuit_instantiate(self):
        """Test that a addcircuit layer can be instantiated correctly."""
        elementary.AddCircuit()

    def test_addcircuit_keras_error(self):
        """Test that addcircuit layer errors in keras call."""
        add = elementary.AddCircuit()
        circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cannot be parsed"):
            add(circuit, append='junk')

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cannot be parsed"):
            add(circuit, prepend='junk')

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cannot be parsed"):
            add('junk', prepend=circuit)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="append or prepend"):
            add(circuit)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="append and prepend"):
            add(circuit, append=circuit, prepend=circuit)

    def test_addcircuit_op_error(self):
        """Test that addcircuit will error inside of ops correctly."""
        add = elementary.AddCircuit()
        circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="matching sizes"):
            # append is wrong shape.
            add(circuit, append=[circuit, circuit])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="matching sizes"):
            # prepend is wrong shape.
            add(circuit, prepend=[circuit, circuit])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="rank 1"):
            # prepend is wrong shape.
            add(circuit, prepend=[[circuit]])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="rank 1"):
            # append is wrong shape.
            add(circuit, append=[[circuit]])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="rank 1"):
            # circuit is wrong shape.
            add([[circuit]], append=[circuit])

    def test_addcircuit_simple_inputs(self):
        """Test the valid cases."""
        add = elementary.AddCircuit()
        circuit = cirq.Circuit(
            cirq.X(cirq.GridQubit(0, 0))**(sympy.Symbol('alpha') * sympy.pi))
        add([circuit, circuit], append=circuit)
        add([circuit, circuit], prepend=circuit)
        add(circuit, append=circuit)
        add(circuit, prepend=circuit)

    def test_addcircuit_modify(self):
        """Test that a addcircuit layer correctly modifies input circuits."""
        bits = cirq.GridQubit.rect(1, 20)
        circuit_a = cirq.testing.random_circuit(bits, 10, 0.9,
                                                util.get_supported_gates())
        circuit_b = cirq.testing.random_circuit(bits, 10, 0.9,
                                                util.get_supported_gates())

        expected_append = util.convert_to_tensor([circuit_a + circuit_b])
        expected_prepend = util.convert_to_tensor([circuit_b + circuit_a])

        append_layer = elementary.AddCircuit()
        prepend_layer = elementary.AddCircuit()

        actual_append = util.convert_to_tensor(
            util.from_tensor(append_layer(circuit_a, append=circuit_b)))
        actual_prepend = util.convert_to_tensor(
            util.from_tensor(prepend_layer(circuit_a, prepend=circuit_b)))

        self.assertEqual(
            util.program_from_tensor_to_ascii_proto(expected_append.numpy()[0]),
            util.program_from_tensor_to_ascii_proto(actual_append.numpy()[0]))
        self.assertEqual(
            util.program_from_tensor_to_ascii_proto(
                expected_prepend.numpy()[0]),
            util.program_from_tensor_to_ascii_proto(actual_prepend.numpy()[0]))


if __name__ == "__main__":
    tf.test.main()
