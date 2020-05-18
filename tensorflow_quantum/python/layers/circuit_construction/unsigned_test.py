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

from tensorflow_quantum.python.layers.circuit_construction import unsigned
from tensorflow_quantum.python.layers.circuit_executors import expectation


class ProjectorOnOneTest(tf.test.TestCase):
    """Test the projector_on_one function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""
        for bad_arg in [-5, 12, cirq.LineQubit(6), []]:
            with self.assertRaisesRegex(TypeError,
                                        expected_regex="cirq.GridQubit"):
                _ = unsigned.projector_on_one(bad_arg)

    def test_return(self):
        """Confirm that GridQubit input returns correct PauliSum."""
        for i in range(12):
            for j in range(12):
                this_q = cirq.GridQubit(i, j)
                expected_psum = 0.5*cirq.I(this_q) - 0.5*cirq.Z(this_q)
                test_psum = unsigned.projector_on_one(this_q)
                self.assertEqual(expected_psum, test_psum)

    
class IntegerOperatorTest(tf.test.TestCase):
    """Test the integer_operator function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""
        for bad_arg in [-7, 3, cirq.LineQubit(2), ["junk"]]:
            with self.assertRaisesRegex(TypeError,
                                        expected_regex="cirq.GridQubit"):
                _ = unsigned.integer_operator(bad_arg)

    def test_return(self):
        """Confirm that correct operators are created."""
        for i in range(5):
            qubits = cirq.GridQubit.rect(1, i + 1)
            expected_psum = (2**(i+1) - 1)*0.5*cirq.I(qubits[0])
            for loc, q in enumerate(qubits):
                expected_psum -= 2**(i - loc)*0.5*cirq.Z(q)
            test_psum = unsigned.integer_operator(qubits)
            self.assertEqual(expected_psum, test_psum)


class RegistersFromPrecisionsTest(tf.test.TestCase):
    """Test the registers_from_precisions function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""
        for bad_arg in [-7, 3, cirq.LineQubit(2), [[]], ["junk"]]:
            with self.assertRaisesRegex(TypeError,
                                        expected_regex=""):
                _ = unsigned.registers_from_precisions(bad_arg)

    def test_return(self):
        """Confirm that registers are built correctly."""
        precisions = [3, 2, 4]
        register_list = unsigned.registers_from_precisions(precisions)
        test_set = set()
        for r, p in zip(register_list, precisions):
            self.assertEqual(len(r), p)
            for q in r:
                test_set.add(q)
                self.assertIsInstance(q, cirq.GridQubit)
        # Ensure all qubits are unique
        self.assertEqual(len(test_set), sum(precisions))


class BuildCostPsumTest(tf.test.TestCase):
    """Test the build_cost_psum function."""

    def test_bad_inputs(self):
        """Confirm function raises error on bad inputs."""

    def test_psum_return(self):
        """Confirm operators are built correctly."""

    def test_counting(self):
        """Confirm we get numbers as expected from quantum integer psums."""
        precisions = [5, 3, 4]
        registers = unsigned.registers_from_precisions(precisions)
        cliques = [{(0,): 1}, {(1,): 1}, {(2,): 1}]
        cliques_psums = [unsigned.build_cost_psum(precisions, c) for c in cliques]
        exp_layer = expectation.Expectation()
        def append_register_bits(r, r_int, precisions, register_list):
            r_int_str = format(r_int, 'b')
            bit_circ = cirq.Circuit()
            for n, c in enumerate(r_int_str[::-1]):
                if bool(int(c)):
                    bit_circ += cirq.X(registers[r][precisions[r] - 1 - n])
            for q in registers[r]:
                bit_circ += cirq.I(q)
            return bit_circ
        # Test that all counts increment correctly
        for i in range(precisions[0]):
            bit_circ = append_register_bits(0, i, precisions, registers)
            for j in range(precisions[1]):
                bit_circ += append_register_bits(1, j, precisions, registers)
                for k in range(precisions[2]):
                    bit_circ += append_register_bits(2, k, precisions, registers)
                    test_val = exp_layer(bit_circ, operators=cliques_psums)
                    self.assertAllClose([i, j, k], test_val.numpy()[0], atol=1e-5)


# class AppendCostExpTest(tf.test.TestCase):
#     """Test AppendCostExp."""

#     def test_append_cost_exp_instantiate(self):
#         """Test that a addcircuit layer can be instantiated correctly."""
#         qudit.AppendCostExp()


#     def test_append_cost_exp_op_error(self):
#         """Test that addcircuit will error inside of ops correctly."""
#         add = elementary.AddCircuit()
#         circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))

#         with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
#                                     expected_regex="rank 1"):
#             # circuit is wrong shape.
#             add([[circuit]], )

#     def test_append_cost_exp(self):
#         """Test that the correct circuit is generated."""
#         precisions = [3, 4]
#         cliques = {(0,): 2, (1,): 3, (1, 2,): 4}
#         my_cost_exp = AppendCostExp(cirq.Circuit(), precisions=precisions, cost=cliques)
        
#     def test_append_cost_exp_append(self):
        


if __name__ == "__main__":
    tf.test.main()
