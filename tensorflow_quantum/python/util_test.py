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
"""Tests for TFQ utilities."""
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import sympy

import cirq
from tensorflow_quantum.core.serialize import serializer
from tensorflow_quantum.python import util
from tensorflow_quantum.python.layers.circuit_executors import expectation


def _single_to_tensor(item):
    if not isinstance(item, (cirq.PauliSum, cirq.PauliString, cirq.Circuit)):
        raise TypeError("Item must be a Circuit or PauliSum. Got {}.".format(
            type(item)))
    if isinstance(item, (cirq.PauliSum, cirq.PauliString)):
        return serializer.serialize_paulisum(item).SerializeToString()
    return serializer.serialize_circuit(item).SerializeToString()


def _exponential(theta, op):
    op_mat = cirq.unitary(op)
    return np.eye(op_mat.shape[0]) * np.cos(theta) - 1j * op_mat * np.sin(theta)


BITS = list(cirq.GridQubit.rect(1, 10))


def _items_to_tensorize():
    """Objects on which convert_to_tensor convert_from_tensor will be tested."""
    return [{
        'item': x
    } for x in (util.random_pauli_sums(BITS, 5, 5) + [
        cirq.PauliSum.from_pauli_strings([
            cirq.PauliString(),
            cirq.PauliString(cirq.Z(cirq.GridQubit(0, 0)))
        ])
    ] + [cirq.PauliString(), cirq.PauliString()] + [cirq.Circuit()] + [
        cirq.testing.random_circuit(BITS, 25, 0.9, util.get_supported_gates())
        for _ in range(5)
    ])]


class UtilFunctionsTest(tf.test.TestCase, parameterized.TestCase):
    """Test that utility functions work."""

    def test_get_supported_gates(self):
        """Confirm one of every gate is returned."""
        mapping_1 = util.get_supported_gates()
        self.assertEqual(len(mapping_1.keys()),
                         len(serializer.SERIALIZER.supported_gate_types()))

    @parameterized.parameters(_items_to_tensorize())
    def test_convert_to_tensor(self, item):
        """Test that the convert_to_tensor function works correctly by manually
        serializing flat and 2-deep nested lists of Circuits and PauliSums."""
        nested = [[item, item]] * 2
        nested_actual = util.convert_to_tensor(nested)
        nested_expected = np.array(
            [np.array([_single_to_tensor(x) for x in row]) for row in nested])
        self.assertAllEqual(nested_actual, nested_expected)
        flat = [item, item]
        flat_actual = util.convert_to_tensor(flat)
        flat_expected = np.array([_single_to_tensor(x) for x in flat])
        self.assertAllEqual(flat_actual, flat_expected)

    def test_convert_to_tensor_errors(self):
        """Test that convert_to_tensor fails when it should."""
        with self.assertRaisesRegex(TypeError, expected_regex="Incompatible"):
            util.convert_to_tensor("junk")
        with self.assertRaisesRegex(TypeError, expected_regex="Incompatible"):
            util.convert_to_tensor([1, cirq.Circuit()])
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='non-rectangular'):
            util.convert_to_tensor([[cirq.Circuit()], cirq.Circuit()])
        with self.assertRaisesRegex(TypeError, expected_regex="Incompatible"):
            util.convert_to_tensor(
                [cirq.Circuit(),
                 cirq.X(BITS[0]) + cirq.Y(BITS[1])])

    @parameterized.parameters(_items_to_tensorize())
    def test_from_tensor(self, item):
        """Check from_tensor assuming convert_to_tensor works."""

        item_nested_tensorized = util.convert_to_tensor([[item, item],
                                                         [item, item]])
        item_flat_tensorized = util.convert_to_tensor([item, item])
        item_nested_cycled = util.convert_to_tensor(
            util.from_tensor(item_nested_tensorized))

        self.assertAllEqual(item_nested_tensorized, item_nested_cycled)
        item_flat_cycled = util.convert_to_tensor(
            util.from_tensor(item_flat_tensorized))
        self.assertAllEqual(item_flat_tensorized, item_flat_cycled)

    def test_from_tensor_errors(self):
        """test that from_tensor fails when it should."""
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='Error decoding item'):
            util.from_tensor(
                tf.convert_to_tensor([
                    'bad',
                    serializer.serialize_circuit(
                        cirq.Circuit()).SerializeToString()
                ]))
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='Error decoding item'):
            util.from_tensor(
                tf.convert_to_tensor([
                    serializer.serialize_circuit(
                        cirq.Circuit()).SerializeToString() + b'bad'
                ]))
        with self.assertRaisesRegex(TypeError, expected_regex='single type'):
            util.from_tensor(
                tf.convert_to_tensor([
                    serializer.serialize_circuit(
                        cirq.Circuit()).SerializeToString(),
                    serializer.serialize_paulisum(
                        cirq.X(BITS[0]) + cirq.Y(BITS[1])).SerializeToString()
                ]))
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='recieved bad type'):
            util.from_tensor("junk")

    def test_cartesian_product(self):
        """Ensure cartesian_product works. inputs are any iterable you want."""
        result1 = list(util.kwargs_cartesian_product(a=[1, 2], b='hi'))
        self.assertEqual(result1, [{
            'a': 1,
            'b': 'h'
        }, {
            'a': 1,
            'b': 'i'
        }, {
            'a': 2,
            'b': 'h'
        }, {
            'a': 2,
            'b': 'i'
        }])

        result2 = list(
            util.kwargs_cartesian_product(**{
                'one': [1, 2, 3],
                'two': [4, 5]
            }))
        self.assertEqual(result2, [{
            'one': 1,
            'two': 4
        }, {
            'one': 1,
            'two': 5
        }, {
            'one': 2,
            'two': 4
        }, {
            'one': 2,
            'two': 5
        }, {
            'one': 3,
            'two': 4
        }, {
            'one': 3,
            'two': 5
        }])

        with self.assertRaisesRegex(ValueError, expected_regex='not iterable'):
            list(util.kwargs_cartesian_product(a=[1, 2], b=-1))

    def test_get_circuit_symbols(self):
        """Test that symbols can be extracted from circuits.
        This test will error out if get_supported_gates gets updated with new
        gates and the get_circuit function isn't updated.
        """
        expected_symbols = ['alpha', 'beta', 'gamma', 'omega']
        qubits = cirq.GridQubit.rect(1, 20)
        n_moments = 200
        for _ in range(5):
            test_circuit = util.random_symbol_circuit(qubits, expected_symbols,
                                                      n_moments)
            extracted_symbols = util.get_circuit_symbols(test_circuit)
            self.assertListEqual(sorted(extracted_symbols),
                                 sorted(expected_symbols))


class ExponentialUtilFunctionsTest(tf.test.TestCase):
    """Test that Exponential utility functions work."""

    def test_many_clifford_to_many_z(self):
        """Confirm correct basis transformations of input PauliSums."""
        q = cirq.GridQubit.rect(1, 4)
        test_term = 0.2277 * cirq.Z(q[1]) * cirq.X(q[2]) * cirq.Y(q[3])
        test_basis_gates = [cirq.H(q[2]), cirq.rx(np.pi / 2)(q[3])]
        test_conj_gates = [cirq.rx(-np.pi / 2)(q[3]), cirq.H(q[2])]

        gate_list, conj_gate_list = util._many_clifford_to_many_z(test_term)
        self.assertEqual(gate_list, test_basis_gates)
        self.assertEqual(conj_gate_list, test_conj_gates)

    def test_many_z_to_single_z(self):
        """Test many Z's to a single Z."""
        q = cirq.GridQubit.rect(1, 8)
        benchmark_term = 1.321 * cirq.Z(q[0]) * cirq.Z(q[3]) * cirq.Z(
            q[5]) * cirq.Z(q[7])
        # Assume the focal qubit is set to q[3].
        benchmark_gates_indices = [(q[7], q[3]), (q[5], q[3]), (q[0], q[3])]
        gates, _ = util._many_z_to_single_z(q[3], benchmark_term)
        for gate_op in gates:
            qubits = gate_op.qubits
            gate = gate_op.gate
            self.assertIsInstance(gate, cirq.CNotPowGate)
            self.assertIn(qubits, benchmark_gates_indices)
            benchmark_gates_indices.remove(qubits)
        self.assertEqual([], benchmark_gates_indices)

    def test_exponential_error(self):
        """Test exponential fails on bad inputs."""
        test_paulistring = cirq.X(cirq.GridQubit(0, 0))
        test_paulisum = cirq.X(cirq.GridQubit(0, 0)) + cirq.Z(
            cirq.GridQubit(0, 1))

        # bad operators
        with self.assertRaisesRegex(TypeError, expected_regex='not a list'):
            util.exponential('junk')
        for bad_op_list in [['junk'], [test_paulistring, 'junk'],
                            [test_paulistring, test_paulisum, 'junk']]:
            with self.assertRaisesRegex(TypeError,
                                        expected_regex='in operators'):
                util.exponential(bad_op_list)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="only supports real"):
            util.exponential([1j * test_paulistring])

        # bad coefficients
        with self.assertRaisesRegex(TypeError, expected_regex='not a list'):
            util.exponential([test_paulisum], coefficients='junk')

        for bad_coeff_list in [[None, 1.0], [['junk'], 1.0], [1.0, ['junk']],
                               [1.0, 1j]]:
            with self.assertRaisesRegex(TypeError,
                                        expected_regex='in coefficients'):
                util.exponential([test_paulistring, test_paulistring],
                                 coefficients=bad_coeff_list)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='should be the same as'):
            util.exponential([test_paulistring], coefficients=[1.0, 2.0])

    def test_exponential_simple(self):
        """Test exponential for a simple operator."""
        q = cirq.GridQubit(0, 0)
        for op in [cirq.X, cirq.Y, cirq.Z]:
            theta = np.random.random()
            circuit = util.exponential(operators=[theta * op(q)])
            ground_truth_unitary = _exponential(theta, op(q))
            self.assertAllClose(ground_truth_unitary, cirq.unitary(circuit))

    def test_allowed_cases(self):
        """Confirm all valid argument combinations are accepted."""
        t_pstr = cirq.X(cirq.GridQubit(0, 0))
        t_psum = cirq.X(cirq.GridQubit(0, 0)) + cirq.Z(cirq.GridQubit(0, 1))
        for op_list in [[t_pstr], [t_psum], (t_pstr,), (t_psum,)]:
            for coeff in ["test", sympy.Symbol("test"), 0.5]:
                for coeff_inp in [[coeff], (coeff,), np.array([coeff])]:
                    _ = util.exponential(op_list, coefficients=coeff_inp)

    def test_exponential_identity(self):
        """Test exponential for an identity."""
        theta = np.random.random()
        identity = cirq.PauliString({None: cirq.I})
        circuit = util.exponential(operators=[theta * identity])

        result_gates = []
        for moment in circuit:
            for gate_op in moment:
                result_gates.append(gate_op.gate)

        # Because it has no qubit in the total circuit, the default is set to
        # zeroth qubit.
        self.assertEqual(circuit.all_qubits(), frozenset({cirq.GridQubit(0,
                                                                         0)}))

        self.assertIsInstance(result_gates[0], cirq.XPowGate)
        self.assertIsInstance(result_gates[1], cirq.ZPowGate)
        self.assertIsInstance(result_gates[2], cirq.XPowGate)
        self.assertIsInstance(result_gates[3], cirq.ZPowGate)

        self.assertAllClose(
            np.eye(2) * np.exp(-1j * theta), cirq.unitary(circuit))

    def test_exponential_complex(self):
        """Test exponential for complex operators."""
        q = cirq.GridQubit.rect(1, 3)
        theta1 = np.random.random()
        theta2 = np.random.random()
        identity = cirq.PauliString({None: cirq.I})
        op1 = cirq.Z(q[1]) * cirq.Z(q[2])
        op2 = identity
        circuit = util.exponential(operators=[theta1 * op1, theta2 * op2])

        result_gates = []
        for moment in circuit:
            for gate_op in moment:
                result_gates.append(gate_op)

        self.assertIsInstance(result_gates[0].gate, cirq.CNotPowGate)
        self.assertIsInstance(result_gates[1].gate, cirq.ZPowGate)
        self.assertIsInstance(result_gates[2].gate, cirq.CNotPowGate)
        self.assertIsInstance(result_gates[3].gate, cirq.XPowGate)
        self.assertIsInstance(result_gates[4].gate, cirq.ZPowGate)
        self.assertIsInstance(result_gates[5].gate, cirq.XPowGate)
        self.assertIsInstance(result_gates[6].gate, cirq.ZPowGate)

        # The exponentiation of identity should not be on q[0], but on q[1].
        for i in range(3, 7):
            self.assertEqual(result_gates[i].qubits, (q[1],))

        ground_truth_unitary = _exponential(theta1, op1)
        result_unitary = cirq.unitary(circuit)
        global_phase = ground_truth_unitary[0][0] / result_unitary[0][0]
        result_unitary *= global_phase

        self.assertAllClose(ground_truth_unitary, result_unitary)

    def test_exponential_commutablility(self):
        """Test exponential for non-commutable operator."""
        q = cirq.GridQubit(0, 0)
        theta1 = np.random.random()
        theta2 = np.random.random()
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="non-commutable"):
            util.exponential(
                operators=[theta1 * cirq.X(q) + theta2 * cirq.Z(q)])

    def test_serializability(self):
        """Test exponential with serializer."""
        q = cirq.GridQubit.rect(1, 2)
        theta = np.random.random()
        identity = cirq.PauliString({None: cirq.I})
        op1 = theta * cirq.Z(q[0]) * cirq.Z(q[1])
        op2 = theta * identity
        circuit = util.exponential(operators=[op1, op2])
        util.convert_to_tensor([circuit])


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


# ==============================================================================
# Unsigned quantum integer tests.
# ==============================================================================


class ProjectorOnOneTest(tf.test.TestCase):
    """Test the projector_on_one function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""
        for bad_arg in [-5, 12, cirq.LineQubit(6), []]:
            with self.assertRaisesRegex(TypeError,
                                        expected_regex="cirq.GridQubit"):
                _ = util.projector_on_one(bad_arg)

    def test_return(self):
        """Confirm that GridQubit input returns correct PauliSum."""
        for i in range(12):
            for j in range(12):
                this_q = cirq.GridQubit(i, j)
                expected_psum = 0.5 * cirq.I(this_q) - 0.5 * cirq.Z(this_q)
                test_psum = util.projector_on_one(this_q)
                self.assertEqual(expected_psum, test_psum)


class UnsignedIntegerOperatorTest(tf.test.TestCase):
    """Test the unsigned_integer_operator function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""
        for bad_arg in [-7, 3, cirq.LineQubit(2), ["junk"]]:
            with self.assertRaisesRegex(TypeError,
                                        expected_regex="cirq.GridQubit"):
                _ = util.unsigned_integer_operator(bad_arg)

    def test_return(self):
        """Confirm that correct operators are created."""
        for i in range(5):
            qubits = cirq.GridQubit.rect(1, i + 1)
            expected_psum = (2**(i + 1) - 1) * 0.5 * cirq.I(qubits[0])
            for loc, q in enumerate(qubits):
                expected_psum -= 2**(i - loc) * 0.5 * cirq.Z(q)
            test_psum = util.unsigned_integer_operator(qubits)
            self.assertEqual(expected_psum, test_psum)


class RegistersFromPrecisionsTest(tf.test.TestCase):
    """Test the registers_from_precisions function."""

    def test_bad_inputs(self):
        """Confirm that function raises error on bad input."""
        for bad_arg in [-7, 3, cirq.LineQubit(2), [[]], ["junk"]]:
            with self.assertRaisesRegex(TypeError, expected_regex=""):
                _ = util.registers_from_precisions(bad_arg)

    def test_return(self):
        """Confirm that registers are built correctly."""
        precisions = [3, 2, 4]
        register_list = util.registers_from_precisions(precisions)
        test_set = set()
        for r, p in zip(register_list, precisions):
            self.assertEqual(len(r), p)
            for q in r:
                test_set.add(q)
                self.assertIsInstance(q, cirq.GridQubit)
        # Ensure all qubits are unique
        self.assertEqual(len(test_set), sum(precisions))


class BuildUnsignedCliquesPsumTest(tf.test.TestCase):
    """Test the build_unsigned_cliques_psum function."""

    def test_bad_inputs(self):
        """Confirm function raises error on bad inputs."""
        p = [3, 4]
        c = {(0,): 2, (1,): 3, (0, 1): 4}
        with self.assertRaisesRegex(TypeError, expected_regex="list"):
            _ = util.build_unsigned_cliques_psum("junk", c)
        with self.assertRaisesRegex(TypeError, expected_regex="integer"):
            _ = util.build_unsigned_cliques_psum(["junk"], c)
        with self.assertRaisesRegex(TypeError, expected_regex="greater than"):
            _ = util.build_unsigned_cliques_psum([2, 0], c)
        with self.assertRaisesRegex(ValueError, expected_regex="Cannot access"):
            # All labels must be in range
            _ = util.build_unsigned_cliques_psum(p, {(0, 1, 2): 3})
        with self.assertRaisesRegex(TypeError, expected_regex="dict"):
            _ = util.build_unsigned_cliques_psum(p, "junk")
        with self.assertRaisesRegex(TypeError, expected_regex="dict"):
            _ = util.build_unsigned_cliques_psum(p, [1])
        with self.assertRaisesRegex(TypeError, expected_regex="tuple"):
            # Most common error, where the key is not parsed as a tuple
            _ = util.build_unsigned_cliques_psum(p, {(0): 1})
        with self.assertRaisesRegex(TypeError, expected_regex="tuple"):
            # Most common error, where the key is not parsed as a tuple
            _ = util.build_unsigned_cliques_psum(p, {(0,): 1, (0): 1})
        with self.assertRaisesRegex(TypeError, expected_regex="tuple"):
            _ = util.build_unsigned_cliques_psum(p, {"junk": 1})
        with self.assertRaisesRegex(TypeError, expected_regex="integer"):
            _ = util.build_unsigned_cliques_psum(p, {(0, 1, "junk"): 1})
        with self.assertRaisesRegex(TypeError, expected_regex="non-negative"):
            _ = util.build_unsigned_cliques_psum(p, {(0, 1, -1): 1})
        with self.assertRaisesRegex(TypeError, expected_regex="integer"):
            _ = util.build_unsigned_cliques_psum(p, {(2.7,): 1})
        with self.assertRaisesRegex(TypeError, expected_regex="value"):
            _ = util.build_unsigned_cliques_psum(p, {(0,): "junk"})
        with self.assertRaisesRegex(TypeError, expected_regex="value"):
            _ = util.build_unsigned_cliques_psum(p, {(0,): 1, (1,): 1j})

    def test_psum_return(self):
        """Confirm operators are built correctly."""
        p = [3, 4]
        c = {(0,): 2, (1,): 3, (0, 1): 4, (): 7}
        test_psum = util.build_unsigned_cliques_psum(p, c)
        registers = util.registers_from_precisions(p)
        j0 = util.unsigned_integer_operator(registers[0])
        j1 = util.unsigned_integer_operator(registers[1])
        expected_psum = 2 * j0 + 3 * j1 + 4 * j0 * j1 + 7 * cirq.PauliString(
            cirq.I(registers[0][0]))
        self.assertEqual(expected_psum, test_psum)

    def test_counting(self):
        """Confirm we get numbers as expected from quantum integer psums."""
        precisions = [5, 3, 4]
        registers = util.registers_from_precisions(precisions)
        cliques = [{(0,): 1}, {(1,): 1}, {(2,): 1}]
        cliques_psums = [
            util.build_unsigned_cliques_psum(precisions, c) for c in cliques
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
        registers = util.registers_from_precisions(precisions)
        cliques = {(0, 0): 1}
        cliques_psums = util.build_unsigned_cliques_psum(precisions, cliques)
        exp_layer = expectation.Expectation()
        # Test that all counts increment correctly
        for i in range(2**precisions[0]):
            bit_circ_i = _append_register_bits(0, i, precisions, registers)
            test_val = exp_layer(bit_circ_i, operators=cliques_psums)
            self.assertAllClose([[i**2]], test_val.numpy(), atol=1e-5)

    def test_polynomial(self):
        """Confirm that expectations of polynomial with offset are correct."""
        precisions = [5]
        registers = util.registers_from_precisions(precisions)
        # test the polynomial y = 3 + 5.2x - 7.5x**2 + 0.3x**3
        cliques = {(): 3, (0,): 5.2, (0, 0): -7.5, (0, 0, 0): 0.3}
        cliques_psums = util.build_unsigned_cliques_psum(precisions, cliques)
        exp_layer = expectation.Expectation()
        # Test that all counts increment correctly
        for i in range(2**precisions[0]):
            bit_circ_i = _append_register_bits(0, i, precisions, registers)
            test_val = exp_layer(bit_circ_i, operators=cliques_psums)
            self.assertAllClose([[3 + 5.2 * i - 7.5 * i**2 + 0.3 * i**3]],
                                test_val.numpy(),
                                atol=1e-4)


class BuildUnsignedCliquesExpTest(tf.test.TestCase):
    """Test the build_unsigned_cliques_exp function."""

    def test_build_unsigned_cliques_exp_error(self):
        """Test that build_unsigned_cliques_exp raises error on bad input."""
        p = [5]
        c = {(0,): 1}
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = util.build_unsigned_cliques_exp("junk", c)
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = util.build_unsigned_cliques_exp(p, "junk")
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = util.build_unsigned_cliques_exp(p, c, [])

    def test_build_unsigned_cliques_exp_return(self):
        """Test that the correct circuit is generated."""
        p = [3, 4]
        c = {(1,): 2}
        for coeff in [3.4, "gamma", sympy.Symbol("symbol")]:
            expected_circuit = util.exponential(
                [util.build_unsigned_cliques_psum(p, c)], [coeff])
            test_circuit = util.build_unsigned_cliques_exp(p, c, coeff)
            self.assertTrue(
                cirq.protocols.approx_eq(expected_circuit, test_circuit))


class BuildUnsignedMomentaExpTest(tf.test.TestCase):
    """Test the build_unsigned_momenta_exp function."""

    def test_build_unsigned_momenta_exp_layer_error(self):
        """Test that build_unsigned_momenta_exp raises error on bad input."""
        p = [5]
        c = {(0,): 1}
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = util.build_unsigned_momenta_exp("junk", c)
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = util.build_unsigned_momenta_exp(p, "junk")
        with self.assertRaisesRegex(TypeError, expected_regex=""):
            _ = util.build_unsigned_momenta_exp(p, c, [])

    def test_build_unsigned_momenta_exp_return(self):
        """Test that the correct circuit is generated."""
        p = [3, 4]
        c = {(1,): 2}
        r = util.registers_from_precisions(p)
        qft_0 = cirq.Circuit(cirq.QFT(*r[0]))
        qft_1 = cirq.Circuit(cirq.QFT(*r[1]))
        convert = cirq.ConvertToCzAndSingleGates(allow_partial_czs=True)
        convert(qft_0)
        convert(qft_1)
        for coeff in [3.4, "gamma", sympy.Symbol("symbol")]:
            expected_circuit = (
                qft_0 + qft_1 +
                util.exponential([util.build_unsigned_cliques_psum(p, c)], [coeff]) +
                qft_1**-1 + qft_0**-1)
            test_circuit = util.build_unsigned_momenta_exp(p, c, coeff)
            self.assertTrue(
                cirq.protocols.approx_eq(expected_circuit, test_circuit))


if __name__ == "__main__":
    tf.test.main()
