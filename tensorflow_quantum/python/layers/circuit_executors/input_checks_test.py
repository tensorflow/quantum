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
from tensorflow_quantum.python.layers.circuit_executors import input_checks
from tensorflow_quantum.python import util


class ExpandCircuitsTest(tf.test.TestCase):
    """Confirm circuits and symbols are upgraded correctly."""

    def test_expand_circuits_type_inputs_error(self):
        """Test that expand_circuits errors on bad types."""

        qubit = cirq.GridQubit(0, 0)
        symbol = sympy.Symbol('alpha')
        symb_circuit = cirq.Circuit(cirq.H(qubit)**symbol)
        symbol_tensor = tf.convert_to_tensor(
                [str(symbol)], dtype=tf.dtypes.string)
        bad_tensor = tf.constant([1, 2])
        symbol_tensor_double = tf.convert_to_tensor(
                [str(symbol), str(symbol)], dtype=tf.dtypes.string)
        
        # Bad circuit arg
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must contain serialized circuits"):
            input_checks.expand_circuits(bad_tensor, symbol_names=symbol_tensor,
                                         symbol_values=[[0.5]])
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="inputs must be a "):
            input_checks.expand_circuits('junk', symbol_names=symbol_tensor,
                                         symbol_values=[[0.5]])

        # Bad name arg
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="string or sympy.Symbol"):
            input_checks.expand_circuits(symb_circuit, symbol_names=[symbol, 5.0])
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="string or sympy.Symbol"):
            input_checks.expand_circuits(symb_circuit, symbol_names=[[]])
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="must be unique."):
            input_checks.expand_circuits(symb_circuit, symbol_names=[symbol, symbol])
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must have dtype string"):
            input_checks.expand_circuits(symb_circuit, symbol_names=bad_tensor)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="must be unique."):
            input_checks.expand_circuits(symb_circuit, symbol_names=symbol_tensor_double)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must be list-like"):
            input_checks.expand_circuits(symb_circuit, symbol_names='junk')

        # Bad value arg
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must have dtype float32"):
            input_checks.expand_circuits(symb_circuit, symbol_names=symbol_tensor,
                                         symbol_values=symbol_tensor)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="must be list-like"):
            input_checks.expand_circuits(symb_circuit, symbol_names=symbol_tensor,
                                         symbol_values='junk')


    def test_allowed_cases(self):
        """Ensure all allowed input combinations are upgraded correctly."""
        qubits = cirq.GridQubit.rect(1, 3);
        names_symbol = list(sympy.symbols("s1:4"))
        names_string = [str(s) for s in names_symbol]
        names_tuple = tuple(names_string)
        names_ndarray = np.array(names_string)
        names_tensor = tf.convert_to_tensor(names_string,
                                            dtype=tf.dtypes.string)
        circuit_list = [cirq.Circuit(cirq.H(qubits[i])**names_symbol[i])
                        for i in range(3)]
        circuit_alone = circuit_list[0]
        circuit_tensor = util.convert_to_tensor(circuit_list)

        values_list = [[]]


if __name__ == '__main__':
    tf.test.main()
