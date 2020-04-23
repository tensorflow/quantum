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
"""Tests for tensorflow_quantum.layers.circuit_executors.input_checks."""
import numpy as np
import sympy
import tensorflow as tf

import cirq
from tensorflow_quantum.python import util
from tensorflow_quantum.python.layers.circuit_executors import input_checks


class ExpandCircuitsTest(tf.test.TestCase):
    """Confirm circuits and symbols are upgraded correctly."""

    def test_expand_circuits_error(self):
        """Test that expand_circuits errors on bad arguments."""

        # Valid test constants
        qubit = cirq.GridQubit(0, 0)
        symbol = sympy.Symbol('alpha')
        names_tensor = tf.convert_to_tensor([str(symbol)],
                                            dtype=tf.dtypes.string)
        circuit_tensor = util.convert_to_tensor(
            [cirq.Circuit(cirq.H(qubit)**symbol)])
        values_tensor = tf.convert_to_tensor([[0.5]], dtype=tf.dtypes.float32)

        # Invalid test contants
        bad_tensor = tf.constant([1, 2])
        names_tensor_double = tf.convert_to_tensor(
            [str(symbol), str(symbol)], dtype=tf.dtypes.string)

        # Bad circuit arg
        with self.assertRaisesRegex(
                TypeError, expected_regex="must contain serialized circuits"):
            input_checks.expand_circuits(bad_tensor,
                                       symbol_names=names_tensor,
                                       symbol_values=values_tensor)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="inputs must be a "):
            input_checks.expand_circuits('junk',
                                       symbol_names=names_tensor,
                                       symbol_values=values_tensor)

        # Bad name arg
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="string or sympy.Symbol"):
            input_checks.expand_circuits(circuit_tensor,
                                       symbol_names=[symbol, 5.0],
                                       symbol_values=values_tensor)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="string or sympy.Symbol"):
            input_checks.expand_circuits(circuit_tensor,
                                       symbol_names=[[]],
                                       symbol_values=values_tensor)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="must be unique."):
            input_checks.expand_circuits(circuit_tensor,
                                       symbol_names=[symbol, symbol],
                                       symbol_values=values_tensor)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must have dtype string"):
            input_checks.expand_circuits(circuit_tensor,
                                       symbol_names=bad_tensor,
                                       symbol_values=values_tensor)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="must be unique."):
            input_checks.expand_circuits(circuit_tensor,
                                       symbol_names=names_tensor_double,
                                       symbol_values=values_tensor)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must be list-like"):
            input_checks.expand_circuits(circuit_tensor,
                                       symbol_names='junk',
                                       symbol_values=values_tensor)

        # Bad value arg
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must have dtype float32"):
            input_checks.expand_circuits(circuit_tensor,
                                       symbol_names=names_tensor,
                                       symbol_values=names_tensor)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must be list-like"):
            input_checks.expand_circuits(circuit_tensor,
                                       symbol_names=names_tensor,
                                       symbol_values='junk')

    def test_allowed_cases(self):
        """Ensure all allowed input combinations are upgraded correctly."""
        qubits = cirq.GridQubit.rect(1, 2)

        # Test all symbol-filled circuit configurations
        names_symbol_list = list(sympy.symbols("s0:2"))
        names_string_list = [str(s) for s in names_symbol_list]
        names_symbol_tuple = tuple(names_symbol_list)
        names_string_tuple = tuple(names_string_list)
        names_symbol_ndarray = np.array(names_symbol_list)
        names_string_ndarray = np.array(names_string_list)
        names_tensor = tf.convert_to_tensor(names_string_list,
                                            dtype=tf.dtypes.string)
        circuit_alone = cirq.Circuit(
            cirq.H(qubits[0])**names_symbol_list[0],
            cirq.X(qubits[1])**names_symbol_list[1])
        circuit_list = [circuit_alone for _ in range(3)]
        circuit_tuple = tuple(circuit_list)
        circuit_tensor = util.convert_to_tensor(circuit_list)
        values_list = [[1], [2], [3]]
        values_tuple = tuple(values_list)
        values_ndarray = np.array(values_list)
        values_tensor = tf.convert_to_tensor(values_list,
                                             dtype=tf.dtypes.float32)
        for names in [
                names_symbol_list, names_string_list, names_symbol_tuple,
                names_string_tuple, names_symbol_ndarray, names_string_ndarray,
                names_tensor
        ]:
            for circuit in [
                    circuit_alone, circuit_list, circuit_tuple, circuit_tensor
            ]:
                for values in [
                        values_list, values_tuple, values_ndarray, values_tensor
                ]:
                    circuit_test, names_test, values_test = \
                        input_checks.expand_circuits(circuit, names, values)
                    self.assertAllEqual(circuit_test, circuit_tensor)
                    self.assertAllEqual(names_test, names_tensor)
                    self.assertAllEqual(values_test, values_tensor)

        # Test the case of empty symbols
        names_tensor = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        values_tensor = tf.convert_to_tensor([[]] * 3, dtype=tf.dtypes.float32)
        for circuit in [circuit_list, circuit_tuple, circuit_tensor]:
            circuit_test, names_test, values_test = input_checks.expand_circuits(
                circuit)
            self.assertAllEqual(circuit_test, circuit_tensor)
            self.assertAllEqual(names_test, names_tensor)
            self.assertAllEqual(values_test, values_tensor)


class ExpandOperatorsTest(tf.test.TestCase):
    """Confirm operators upgrade correctly."""

    def test_expand_operators_errors(self):
        with self.assertRaisesRegex(RuntimeError,
                                    expected_regex="operators not provided"):
            expectation.Expectation()(symb_circuit,
                                      symbol_names=[symbol],
                                      symbol_values=[[0.5]])

    def test_allowed_cases():
        batch_dim = 3
        bare_string = cirq.Z(bit)
        bare_sum = cirq.PauliSum.from_pauli_strings([bare_string])
        op_list
        op_tuple
        op_ndarray
        op_tensor = util.convert_to_tensor(op_list)
        for op in [bare_sum, bare_string, op_list, op_tuple, op_ndarray, op_tensor]:
            op_test = input_checks.expand_operators(op, batch_dim)
            self.assertAllEqual(op_test, op_tensor)


if __name__ == '__main__':
    tf.test.main()
