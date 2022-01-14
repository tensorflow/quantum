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
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import sympy

import cirq
from tensorflow_quantum.core.serialize import serializer
from tensorflow_quantum.python import util


def _single_to_tensor(item):
    if not isinstance(item, (cirq.PauliSum, cirq.PauliString, cirq.Circuit)):
        raise TypeError("Item must be a Circuit or PauliSum. Got {}.".format(
            type(item)))
    if isinstance(item, (cirq.PauliSum, cirq.PauliString)):
        return serializer.serialize_paulisum(item).SerializeToString(
            deterministic=True)
    return serializer.serialize_circuit(item).SerializeToString(
        deterministic=True)


def _exponential(theta, op):
    op_mat = cirq.unitary(op)
    return np.eye(op_mat.shape[0]) * np.cos(theta) - 1j * op_mat * np.sin(theta)


BITS = list(cirq.GridQubit.rect(1, 10) + cirq.LineQubit.range(2))


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
        self.assertEqual(
            len(mapping_1.keys()),
            len(serializer.SERIALIZER.supported_gate_types()) -
            len(util.get_supported_channels()))

    def test_get_supported_channels(self):
        """Confirm one of every channel is returned."""
        mapping_1 = util.get_supported_channels()
        self.assertEqual(
            len(mapping_1.keys()),
            len(serializer.SERIALIZER.supported_gate_types()) -
            len(util.get_supported_gates()))

    @parameterized.parameters(_items_to_tensorize())
    def test_convert_to_tensor(self, item):
        """Test that the convert_to_tensor function works correctly by manually
        serializing flat and 2-deep nested lists of Circuits and PauliSums."""
        nested = [[item, item]] * 2
        nested_actual = util.convert_to_tensor(
            nested, deterministic_proto_serialize=True)
        nested_expected = np.array(
            [np.array([_single_to_tensor(x) for x in row]) for row in nested])
        self.assertAllEqual(nested_actual, nested_expected)
        flat = [item, item]
        flat_actual = util.convert_to_tensor(flat,
                                             deterministic_proto_serialize=True)
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

        item_nested_tensorized = util.convert_to_tensor(
            [[item, item], [item, item]], deterministic_proto_serialize=True)
        item_flat_tensorized = util.convert_to_tensor(
            [item, item], deterministic_proto_serialize=True)
        item_nested_cycled = util.convert_to_tensor(
            util.from_tensor(item_nested_tensorized),
            deterministic_proto_serialize=True)

        self.assertAllEqual(item_nested_tensorized, item_nested_cycled)
        item_flat_cycled = util.convert_to_tensor(
            util.from_tensor(item_flat_tensorized),
            deterministic_proto_serialize=True)
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
                                    expected_regex='received bad type'):
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

    def test_expression_approx_eq(self):
        """Test that coefficients and symbols are compared correctly."""
        # integers
        a = 1
        b = 1
        c = 2
        atol = 0.1
        self.assertTrue(util._expression_approx_eq(a, b, atol))
        self.assertFalse(util._expression_approx_eq(a, c, atol))
        self.assertTrue(util._expression_approx_eq(a, c, 2.0))

        # reals
        a = 1.1234
        b = 1.1231
        c = 1.1220
        atol = 5e-4
        self.assertTrue(util._expression_approx_eq(a, b, atol))
        self.assertFalse(util._expression_approx_eq(a, c, atol))
        self.assertTrue(util._expression_approx_eq(a, c, 0.01))

        # symbols
        a = sympy.Symbol("s")
        b = sympy.Symbol("s")
        c = sympy.Symbol("s_wrong")
        self.assertTrue(util._expression_approx_eq(a, b, atol))
        self.assertFalse(util._expression_approx_eq(a, c, atol))

        # number * symbol
        a = 3.5 * sympy.Symbol("s")
        b = 3.501 * sympy.Symbol("s")
        c = 3.5 * sympy.Symbol("s_wrong")
        atol = 1e-2
        self.assertTrue(util._expression_approx_eq(a, b, atol))
        self.assertFalse(util._expression_approx_eq(a, c, atol))
        c = 3.6 * sympy.Symbol("s")
        self.assertFalse(util._expression_approx_eq(a, c, atol))

        # symbol * number
        a = sympy.Symbol("s") * -1.7
        b = sympy.Symbol("s") * -1.701
        c = sympy.Symbol("s_wrong") * -1.7
        atol = 1e-2
        self.assertTrue(util._expression_approx_eq(a, b, atol))
        self.assertFalse(util._expression_approx_eq(a, c, atol))
        c = sympy.Symbol("s") * -1.8
        self.assertFalse(util._expression_approx_eq(a, c, atol))

        # other not equal
        atol = 1e-3
        self.assertFalse(util._expression_approx_eq(1, sympy.Symbol("s"), atol))
        self.assertFalse(util._expression_approx_eq(sympy.Symbol("s"), 1, atol))

    def test_expression_approx_eq_error(self):
        """Confirms that bad inputs raise appropriate errors."""
        # too complicated
        a = sympy.Symbol("s_1") * sympy.Symbol("s_2")
        b = 1.0 * a
        with self.assertRaisesRegex(ValueError, expected_regex='not supported'):
            _ = util._expression_approx_eq(a, a, 1e-3)
        with self.assertRaisesRegex(ValueError, expected_regex='not supported'):
            _ = util._expression_approx_eq(a, b, 1e-3)

        # junk
        with self.assertRaisesRegex(TypeError, expected_regex='Invalid input'):
            _ = util._expression_approx_eq(1, 'junk', 1e-3)
        with self.assertRaisesRegex(TypeError, expected_regex='Invalid input'):
            _ = util._expression_approx_eq('junk', 1, 1e-3)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='atol must be a real'):
            _ = util._expression_approx_eq(1, 1, 'junk')

    def test_gate_approx_eq(self):
        """Check valid TFQ gates for approximate equality."""
        atol = 1e-2
        exps_true = [
            3, 2.54, -1.7 * sympy.Symbol("s_1"),
            sympy.Symbol("s_2") * 4.3
        ]
        exps_eq = [
            3, 2.542, -1.705 * sympy.Symbol("s_1"),
            sympy.Symbol("s_2") * 4.305
        ]
        exps_not_eq = [
            4, 2.57, -1.5 * sympy.Symbol("s_1"),
            sympy.Symbol("s_2") * 4.4
        ]

        # Not a child class
        self.assertFalse(util.gate_approx_eq(cirq.X, cirq.Y))

        # Identity gate
        self.assertTrue(util.gate_approx_eq(cirq.I, cirq.I))

        # Parameterized gates
        for e_true, e_eq, e_not_eq in zip(exps_true, exps_eq, exps_not_eq):
            for g in serializer.EIGEN_GATES_DICT:
                g_true = g(exponent=e_true, global_shift=e_eq)
                g_eq = g(exponent=e_eq, global_shift=e_true)
                g_not_eq = g(exponent=e_not_eq, global_shift=e_not_eq)
                self.assertTrue(util.gate_approx_eq(g_true, g_eq, atol=atol))
                self.assertFalse(
                    util.gate_approx_eq(g_true, g_not_eq, atol=atol))
            for g in serializer.PHASED_EIGEN_GATES_DICT:
                g_true = g(exponent=e_true, phase_exponent=-1.0 * e_true)
                g_eq = g(exponent=e_eq, phase_exponent=-1.0 * e_eq)
                g_not_eq = g(exponent=e_not_eq, phase_exponent=-1.0 * e_not_eq)
                self.assertTrue(util.gate_approx_eq(g_true, g_eq, atol=atol))
                self.assertFalse(
                    util.gate_approx_eq(g_true, g_not_eq, atol=atol))
            g_true = cirq.FSimGate(theta=e_true, phi=e_eq)
            g_eq = cirq.FSimGate(theta=e_eq, phi=e_true)
            g_not_eq = cirq.FSimGate(theta=e_not_eq, phi=e_not_eq)
            self.assertTrue(util.gate_approx_eq(g_true, g_eq, atol=atol))
            self.assertFalse(util.gate_approx_eq(g_true, g_not_eq, atol=atol))

        # Controlled gates
        self.assertFalse(
            util.gate_approx_eq(
                cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2]),
                cirq.ops.ControlledGate(cirq.X, 2, [1, 1], [2, 2])))
        self.assertFalse(
            util.gate_approx_eq(
                cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2]),
                cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 1])))
        self.assertFalse(
            util.gate_approx_eq(
                cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2]),
                cirq.ops.ControlledGate(cirq.Y, 2, [1, 0], [2, 2])))
        self.assertTrue(
            util.gate_approx_eq(
                cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2]),
                cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2])))

        # Mixed gates
        self.assertFalse(
            util.gate_approx_eq(
                cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2]), cirq.X))
        self.assertFalse(
            util.gate_approx_eq(
                cirq.X, cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2])))

    def test_gate_approx_eq_error(self):
        """Confirms that bad inputs cause an error to be raised."""
        # junk
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="`gate_true` not a cirq"):
            _ = util.gate_approx_eq("junk", cirq.I)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="`gate_deser` not a cirq"):
            _ = util.gate_approx_eq(cirq.I, "junk")

        # Unsupported gates
        with self.assertRaisesRegex(
                ValueError, expected_regex="`gate_true` not a valid TFQ gate"):
            _ = util.gate_approx_eq(
                cirq.PhasedXZGate(x_exponent=1,
                                  z_exponent=1,
                                  axis_phase_exponent=1), cirq.I)
        with self.assertRaisesRegex(
                ValueError, expected_regex="`gate_deser` not a valid TFQ gate"):
            _ = util.gate_approx_eq(
                cirq.I,
                cirq.PhasedXZGate(x_exponent=1,
                                  z_exponent=1,
                                  axis_phase_exponent=1))
        # Unsupported gates inside a controlled gate
        with self.assertRaisesRegex(
                ValueError, expected_regex="`gate_true` not a valid TFQ gate"):
            _ = util.gate_approx_eq(
                cirq.ops.ControlledGate(
                    cirq.PhasedXZGate(x_exponent=1,
                                      z_exponent=1,
                                      axis_phase_exponent=1), 2, [1, 0],
                    [2, 2]), cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2]))
        with self.assertRaisesRegex(
                ValueError, expected_regex="`gate_deser` not a valid TFQ gate"):
            _ = util.gate_approx_eq(
                cirq.ops.ControlledGate(cirq.X, 2, [1, 0], [2, 2]),
                cirq.ops.ControlledGate(
                    cirq.PhasedXZGate(x_exponent=1,
                                      z_exponent=1,
                                      axis_phase_exponent=1), 2, [1, 0],
                    [2, 2]))

    def test_get_circuit_symbols(self):
        """Test that symbols can be extracted from circuits.
        This test will error out if get_supported_gates gets updated with new
        gates and the get_circuit function isn't updated.
        """
        expected_symbols = ['alpha', 'beta', 'gamma', 'omega']
        qubits = cirq.GridQubit.rect(1, 20)
        n_moments = 200
        for _ in range(5):
            test_circuit = util.random_symbol_circuit(qubits,
                                                      expected_symbols,
                                                      n_moments=n_moments)
            extracted_symbols = util.get_circuit_symbols(test_circuit)
            self.assertListEqual(sorted(extracted_symbols),
                                 sorted(expected_symbols))

    def test_get_circuit_symbols_all(self):
        """Confirm that circuits have all the requested symbols."""
        expected_symbols = ['alpha', 'beta', 'gamma', 'omega']
        qubits = cirq.GridQubit.rect(1, 2)
        n_moments = 1
        for _ in range(5):
            test_circuit = util.random_symbol_circuit(qubits,
                                                      expected_symbols,
                                                      n_moments=n_moments)
            extracted_symbols = util.get_circuit_symbols(test_circuit)
            self.assertListEqual(sorted(extracted_symbols),
                                 sorted(expected_symbols))

    def test_get_circuit_symbols_error(self):
        """Make sure that the method errors where it should."""
        for param in ['2', sympy.Symbol("X")]:
            # Passed an invalid parameter (not a cirq.Circuit).
            with self.assertRaisesRegex(TypeError,
                                        expected_regex='Expected a '
                                        'cirq.Circuit'):
                util.get_circuit_symbols(param)


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
        util.convert_to_tensor([circuit], deterministic_proto_serialize=True)


if __name__ == "__main__":
    tf.test.main()
