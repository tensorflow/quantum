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

from tensorflow_quantum.python import util
from tensorflow_quantum.python.layers.circuit_construction import unsigned
from tensorflow_quantum.python.layers.circuit_executors import expectation


def _append_register_bits(r, r_int, precisions, register_list):
    """Generate a bitstring corresponding to r_int on the indicated register."""
    r_int_str = format(r_int, 'b')
    bit_circ = cirq.Circuit()
    for n, c in enumerate(r_int_str[::-1]):
        if bool(int(c)):
            bit_circ += cirq.X(register_list[r][precisions[r] - 1 - n])
    for q in register_list[r]:
        bit_circ += cirq.I(q)
    return bit_circ


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
                expected_psum = 0.5 * cirq.I(this_q) - 0.5 * cirq.Z(this_q)
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
            expected_psum = (2**(i + 1) - 1) * 0.5 * cirq.I(qubits[0])
            for loc, q in enumerate(qubits):
                expected_psum -= 2**(i - loc) * 0.5 * cirq.Z(q)
            test_psum = unsigned.integer_operator(qubits)
            self.assertEqual(expected_psum, test_psum)


class RegistersFromPrecisionsTest(tf.test.TestCase):
    """Test the registers_from_precisions function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""
        for bad_arg in [-7, 3, cirq.LineQubit(2), [[]], ["junk"]]:
            with self.assertRaisesRegex(TypeError, expected_regex=""):
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


class BuildCliquesPsumTest(tf.test.TestCase):
    """Test the build_cliques_psum function."""

    def test_bad_inputs(self):
        """Confirm function raises error on bad inputs."""
        p = [3, 4]
        c = {(0,): 2, (1,): 3, (0, 1): 4}
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = unsigned.build_cliques_psum("junk", c)
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = unsigned.build_cliques_psum(p, "junk")

    def test_psum_return(self):
        """Confirm operators are built correctly."""
        p = [3, 4]
        c = {(0,): 2, (1,): 3, (0, 1): 4}
        test_psum = unsigned.build_cliques_psum(p, c)
        registers = unsigned.registers_from_precisions(p)
        j0 = unsigned.integer_operator(registers[0])
        j1 = unsigned.integer_operator(registers[1])
        expected_psum = 2 * j0 + 3 * j1 + 4 * j0 * j1
        self.assertEqual(expected_psum, test_psum)

    def test_counting(self):
        """Confirm we get numbers as expected from quantum integer psums."""
        precisions = [5, 3, 4]
        registers = unsigned.registers_from_precisions(precisions)
        cliques = [{(0,): 1}, {(1,): 1}, {(2,): 1}]
        cliques_psums = [
            unsigned.build_cliques_psum(precisions, c) for c in cliques
        ]
        exp_layer = expectation.Expectation()
        # Test that all counts increment correctly
        for i in range(2**precisions[0]):
            bit_circ_i = _append_register_bits(0, i, precisions, registers)
            for j in range(2**precisions[1]):
                bit_circ_j = _append_register_bits(1, j, precisions, registers)
                for k in range(2**precisions[2]):
                    bit_circ_k = _append_register_bits(2, k, precisions,
                                                       registers)
                    test_val = exp_layer(bit_circ_i + bit_circ_j + bit_circ_k,
                                         operators=cliques_psums)
                    self.assertAllClose([[i, j, k]],
                                        test_val.numpy(),
                                        atol=1e-5)

    def test_squares(self):
        """Confirm we get squared numbers as expected from q integer psums."""
        precisions = [5]
        registers = unsigned.registers_from_precisions(precisions)
        cliques = {(0, 0): 1}
        cliques_psums = unsigned.build_cliques_psum(precisions, cliques)
        exp_layer = expectation.Expectation()
        # Test that all counts increment correctly
        for i in range(2**precisions[0]):
            bit_circ_i = _append_register_bits(0, i, precisions, registers)
            test_val = exp_layer(bit_circ_i, operators=cliques_psums)
            self.assertAllClose([[i**2]], test_val.numpy(), atol=1e-5)


class AppendCliquesExpTest(tf.test.TestCase):
    """Test the AppendCliquesExp class."""

    def test_append_cliques_exp_instantiate(self):
        """Test that an AppendCliquesExp layer can be instantiated."""
        p = [5]
        c = {(0,): 1}
        _ = unsigned.AppendCliquesExp(p, c)

    def test_append_cliques_exp_layer_error(self):
        """Test that layer instantiation raises error on bad input."""
        p = [5]
        c = {(0,): 1}
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = unsigned.AppendCliquesExp("junk", c)
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = unsigned.AppendCliquesExp(p, "junk")
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = unsigned.AppendCliquesExp(p, c, [])

    def test_append_cliques_exp_op_error(self):
        """Test that AppendCliquesExp will error inside of ops correctly."""
        p = [2, 5]
        c = {(1,): 1}
        circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="rank 1"):
            unsigned.AppendCliquesExp(p, c)([[circuit]])

    def test_append_cliques_exp(self):
        """Test that the correct circuit is generated."""
        p = [3, 4]
        c = {(1,): 2}
        for coeff in [3.4, "gamma", sympy.Symbol("symbol")]:
            expected_circuit = util.convert_to_tensor([
                util.exponential([unsigned.build_cliques_psum(p, c)], [coeff])
            ])
            cliques_exp_layer = unsigned.AppendCliquesExp(p, c, coeff)
            self.assertEqual(expected_circuit, cliques_exp_layer.exp_circuit)

    def test_append_cliques_exp(self):
        """Test that circuits are appended correctly."""
        p = [3]
        c = {(0,): 1}
        coeff = -2.2
        pre_circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))
        expected_circuit = util.convert_to_tensor([
            pre_circuit +
            util.exponential([unsigned.build_cliques_psum(p, c)], [coeff])
        ])
        cliques_exp_layer = unsigned.AppendCliquesExp(p, c, coeff)
        test_circuit = cliques_exp_layer(pre_circuit)
        self.assertEqual(expected_circuit, test_circuit)


class AppendMomentaExpTest(tf.test.TestCase):
    """Test the AppendMomentaExp class."""

    def test_append_momenta_exp_instantiate(self):
        """Test that an AppendMomentaExp layer can be instantiated."""
        p = [5]
        c = {(0,): 1}
        _ = unsigned.AppendMomentaExp(p, c)

    def test_append_momenta_exp_layer_error(self):
        """Test that layer instantiation raises error on bad input."""
        p = [5]
        c = {(0,): 1}
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = unsigned.AppendMomentaExp("junk", c)
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = unsigned.AppendMomentaExp(p, "junk")
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = unsigned.AppendMomentaExp(p, c, [])

    def test_append_momenta_exp_op_error(self):
        """Test that AppendMomentaExp will error inside of ops correctly."""
        p = [2, 5]
        c = {(1,): 1}
        circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="rank 1"):
            unsigned.AppendMomentaExp(p, c)([[circuit]])

    def test_append_momenta_exp(self):
        """Test that the correct circuit is generated."""
        p = [3, 4]
        c = {(1,): 2}
        r = unsigned.registers_from_precisions(p)
        qft_0 = cirq.Circuit(cirq.decompose(cirq.QFT(*r[0])))
        qft_1 = cirq.Circuit(cirq.decompose(cirq.QFT(*r[1])))
        for coeff in [3.4, "gamma", sympy.Symbol("symbol")]:
            expected_circuit = util.convert_to_tensor([
                qft_0 + qft_1 +
                util.exponential([unsigned.build_momenta_psum(p, c)], [coeff]) +
                qft_0**-1 + qft_1**-1
            ])
            momenta_exp_layer = unsigned.AppendMomentaExp(p, c, coeff)
            self.assertEqual(expected_circuit, momenta_exp_layer.exp_circuit)

    def test_append_momenta_exp(self):
        """Test that circuits are appended correctly."""
        p = [3]
        c = {(0,): 1}
        coeff = -2.2
        pre_circuit = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))
        r = unsigned.registers_from_precisions(p)
        qft_0 = cirq.Circuit(cirq.decompose(cirq.QFT(*r[0])))
        expected_circuit = util.convert_to_tensor([
            pre_circuit + qft_0 +
            util.exponential([unsigned.build_cliques_psum(p, c)], [coeff]) +
            qft_0**-1
        ])
        momenta_exp_layer = unsigned.AppendMomentaExp(p, c, coeff)
        test_circuit = momenta_exp_layer(pre_circuit)
        self.assertEqual(expected_circuit, test_circuit)


if __name__ == "__main__":
    tf.test.main()
