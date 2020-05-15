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
"""Tests for the qudit layers."""
import tensorflow as tf
import cirq
import sympy

from tensorflow_quantum.python.layers.circuit_construction import qudit


class ProjectorOnOneTest(tf.test.TestCase):
    """Test the projector_on_one function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""
        for bad_arg in [-5, 12, cirq.LineQubit(6), []]:
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cirq.GridQubit"):
            integer.projector_on_one(bad_arg)

    def test_return(self):
        """Confirm that GridQubit input returns correct PauliSum."""
        for i in range(12):
            for j in range(12):
                this_q = cirq.GridQubit(i, j)
                expected_psum = 0.5*cirq.I(this_q) - 0.5*cirq.Z(this_q)
                test_psum = qudit.projector_on_one(this_q)
                self.assertEqual(expected_psum, test_psum)

    
class IntegerOperatorTest(tf.test.TestCase):
    """Test the integer_operator function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""

    def test_return(self):
        """Confirm that correct operators are created."""
    

class LayerInputCheckTest(tf.test.TestCase):
    """Confirm layer arguments are error checked and upgraded correctly."""

    def test_bad_cases(self):
        """Test that qudit_layer_input_check errors on bad arguments."""
        add = elementary.AddCircuit()
        circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))

        # Bad input argument
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cannot be parsed"):
            qudit_input_check(circuit, append='junk')

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

    def test_allowed_cases(self):
        """Ensure all allowed input combinations are upgraded correctly."""
        

class AppendCostExpTest(tf.test.TestCase):
    """Test AppendCostExp."""

    def test_append_cost_exp_instantiate(self):
        """Test that a addcircuit layer can be instantiated correctly."""
        qudit.AppendCostExp()


    def test_append_cost_exp_op_error(self):
        """Test that addcircuit will error inside of ops correctly."""
        add = elementary.AddCircuit()
        circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="rank 1"):
            # circuit is wrong shape.
            add([[circuit]], )

    def test_append_cost_exp(self):
        """Test that the correct circuit is generated."""
        precisions = [3, 4]
        cliques = {(0,): 2, (1,): 3, (1, 2,): 4}
        my_cost_exp = AppendCostExp(cirq.Circuit(), precisions=precisions, cost=cliques)
        
    def test_append_cost_exp_append(self):
        


if __name__ == "__main__":
    tf.test.main()
