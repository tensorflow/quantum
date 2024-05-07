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
"""Tests for tensorflow_quantum.layers.circuit_executors.unitary."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys

NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import numpy as np
from absl.testing import parameterized
import sympy
import tensorflow as tf
import cirq

from tensorflow_quantum.python.layers.circuit_executors import unitary
from tensorflow_quantum.python import util


class UnitaryTest(parameterized.TestCase, tf.test.TestCase):
    """Basic tests for the State layer."""

    def test_unitary_create(self):
        """Test that State layers can be created."""
        _ = unitary.Unitary()

    def test_basic_inputs(self):
        """Test that State layer outputs work end to end."""
        simple_circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0)))
        cirq_u = cirq.unitary(simple_circuit)
        tfq_u = unitary.Unitary()(simple_circuit)
        self.assertAllClose(tfq_u, [cirq_u])

    def test_basic_inputs_fixed(self):
        """Test that State layer outputs work on hand case."""
        simple_circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))
        true_u = np.array([[0, 1], [1, 0]], dtype=np.complex64)
        tfq_u = unitary.Unitary()(simple_circuit)
        self.assertAllClose(tfq_u, [true_u])

    def test_single_circuit_batch_inputs(self):
        """Test that a single circuit with multiple parameters works."""
        a_symbol = sympy.Symbol('alpha')
        some_values = np.array([[0.5], [3.5]])
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))**a_symbol)
        results = unitary.Unitary()(circuit,
                                    symbol_names=[a_symbol],
                                    symbol_values=some_values)
        u_1 = cirq.unitary(
            cirq.resolve_parameters(circuit, {a_symbol: some_values[0][0]}))
        u_2 = cirq.unitary(
            cirq.resolve_parameters(circuit, {a_symbol: some_values[1][0]}))

        self.assertAllClose(results, [u_1, u_2])

    def test_multi_circuit_batch(self):
        """Test that a batch of circuits works."""
        a_symbol = sympy.Symbol('alpha')
        some_values = np.array([[0.5], [3.5]])
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))**a_symbol)
        results = unitary.Unitary()(util.convert_to_tensor([circuit, circuit]),
                                    symbol_names=[a_symbol],
                                    symbol_values=some_values)
        u_1 = cirq.unitary(
            cirq.resolve_parameters(circuit, {a_symbol: some_values[0][0]}))
        u_2 = cirq.unitary(
            cirq.resolve_parameters(circuit, {a_symbol: some_values[1][0]}))

        self.assertAllClose(results, [u_1, u_2])

    def test_input_errors(self):
        """Test that bad inputs caught input_check.py."""
        u_calc = unitary.Unitary()
        symbol = sympy.Symbol('alpha')
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))**symbol)
        with self.assertRaisesRegex(Exception, expected_regex=""):
            # no value provided.
            u_calc([circuit, circuit], symbol_names=[symbol], repetitions=5)

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # no name provided.
            u_calc([circuit, circuit],
                   symbol_names=[],
                   symbol_values=[[2.0], [3.0]])

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # deceptive, but the circuit shouldn't be in a list. otherwise fine.
            u_calc([circuit],
                   symbol_names=['alpha'],
                   symbol_values=[[2.0], [3.0]])

    def test_op_errors(self):
        """Test that op errors can be hit."""
        u_calc = unitary.Unitary()
        symbol = sympy.Symbol('alpha')
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))**symbol)
        with self.assertRaisesRegex(Exception, expected_regex=""):
            # wrong symbol name.
            u_calc([circuit],
                   symbol_names=['alphaaaa'],
                   symbol_values=[[2.0], [3.0]])

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # too many symbol values provided.
            u_calc(circuit,
                   symbol_names=['alpha'],
                   symbol_values=[[2.0, 4.0], [3.0, 5.0]])

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # Wrong batch_size for symbol values.
            u_calc([circuit],
                   symbol_names=['alpha'],
                   symbol_values=np.zeros((3, 1)))


if __name__ == '__main__':
    tf.test.main()
