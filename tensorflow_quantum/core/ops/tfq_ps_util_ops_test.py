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
"""Test for ParameterShift specific C++ ops."""
import numpy as np
import tensorflow as tf
import sympy
import cirq

from tensorflow_quantum.core.ops import tfq_ps_util_ops
from tensorflow_quantum.python import util


def _complex_test_circuit():
    t = sympy.Symbol('t')
    r = sympy.Symbol('r')
    qubits = cirq.GridQubit.rect(1, 6)
    circuit_batch = [
        cirq.Circuit(
            cirq.Moment([cirq.H(q) for q in qubits]),
            cirq.Moment([
                cirq.X(qubits[4]),
                cirq.PhasedXPowGate(phase_exponent=np.random.random() * t).on(
                    qubits[5]),
                cirq.ISwapPowGate(exponent=np.random.random() * t).on(
                    qubits[0], qubits[1]),
                cirq.FSimGate(theta=np.random.random() * t,
                              phi=np.random.random() * r).on(
                                  qubits[2], qubits[3])
            ]), cirq.Moment([cirq.H(q) for q in qubits])),
        cirq.Circuit(
            cirq.FSimGate(theta=np.random.random() * t,
                          phi=np.random.random() * r).on(*qubits[:2]),
            cirq.FSimGate(theta=np.random.random() * r,
                          phi=np.random.random() * t).on(qubits[1], qubits[0])),
        cirq.Circuit(
            cirq.Moment([
                cirq.ISwapPowGate(exponent=np.random.random() *
                                  t).on(*qubits[:2]),
                cirq.PhasedXPowGate(phase_exponent=np.random.random() * r).on(
                    qubits[2]),
                cirq.ISwapPowGate(exponent=np.random.random() *
                                  r).on(*qubits[3:5])
            ]))
    ]
    return circuit_batch


class PSDecomposeTest(tf.test.TestCase):
    """Tests on tfq_ps_decompose"""

    def test_iswap_gate_test(self):
        """Test 1 ISwapPowGate decomposition."""
        t = sympy.Symbol('t')
        qubits = cirq.GridQubit.rect(1, 2)
        circuit = cirq.Circuit(
            cirq.ISwapPowGate(exponent=np.random.random() * t).on(*qubits))
        inputs = util.convert_to_tensor([circuit])
        outputs = tfq_ps_util_ops.tfq_ps_decompose(inputs)
        decomposed_programs = util.from_tensor(outputs)
        rand_resolver = {'t': np.random.random()}
        self.assertAllClose(cirq.unitary(
            cirq.resolve_parameters(circuit, rand_resolver)),
                            cirq.unitary(
                                cirq.resolve_parameters(decomposed_programs[0],
                                                        rand_resolver)),
                            atol=1e-5)

    def test_phased_x_pow_gate_test(self):
        """Test 1 PhasedXPowGate decomposition."""
        t = sympy.Symbol('t')
        r = sympy.Symbol('r')
        q = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=np.random.random() * r,
                                exponent=np.random.random() * t).on(q))
        inputs = util.convert_to_tensor([circuit])
        outputs = tfq_ps_util_ops.tfq_ps_decompose(inputs)
        decomposed_programs = util.from_tensor(outputs)
        rand_resolver = {'t': np.random.random(), 'r': np.random.random()}
        self.assertAllClose(cirq.unitary(
            cirq.resolve_parameters(circuit, rand_resolver)),
                            cirq.unitary(
                                cirq.resolve_parameters(decomposed_programs[0],
                                                        rand_resolver)),
                            atol=1e-5)

    def test_fsim_gate_test(self):
        """Test 1 FSimPowGate decomposition."""
        t = sympy.Symbol('t')
        r = sympy.Symbol('r')
        qubits = cirq.GridQubit.rect(1, 2)
        circuit = cirq.Circuit(
            cirq.FSimGate(theta=np.random.random() * r,
                          phi=np.random.random() * t).on(*qubits))
        inputs = util.convert_to_tensor([circuit])
        outputs = tfq_ps_util_ops.tfq_ps_decompose(inputs)
        decomposed_programs = util.from_tensor(outputs)
        rand_resolver = {'t': np.random.random(), 'r': np.random.random()}
        self.assertAllClose(cirq.unitary(
            cirq.resolve_parameters(circuit, rand_resolver)),
                            cirq.unitary(
                                cirq.resolve_parameters(decomposed_programs[0],
                                                        rand_resolver)),
                            atol=1e-5)

    def test_decompose_with_complex_circuit(self):
        """Test decompose with complex circuit."""
        names = ['CLAE', 'HRYV', 'IRKB', 'LKRV', 'PJOU', 'CJKX', 'NASW']
        # Test circuit has a Moment with 1) FSimGate & PhasedXPowGate,
        # 2) PhasedXPowGate & ISwapPowGate and 3) FSimGate & ISwapPowGate.
        # Be careful, they are not decomposed if not parameterized.
        circuit_batch = [
            cirq.Circuit([
                cirq.Moment(operations=[
                    cirq.FSimGate(theta=0.10338130973488413 *
                                  sympy.Symbol('CLAE'),
                                  phi=0.10338130973488413 *
                                  sympy.Symbol('IRKB')).
                    on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                    cirq.PhasedXPowGate(phase_exponent=1.0,
                                        exponent=0.86426029696045281 *
                                        sympy.Symbol('HRYV')).on(
                                            cirq.GridQubit(0, 1)),
                ]),
                cirq.Moment(operations=[
                    cirq.Y.on(cirq.GridQubit(0, 3)),
                    cirq.Z.on(cirq.GridQubit(0, 0)),
                    cirq.FSimGate(theta=1, phi=1).on(cirq.GridQubit(0, 1),
                                                     cirq.GridQubit(0, 2)),
                ]),
                cirq.Moment(operations=[
                    (cirq.CNOT**(0.92874230274398684 * sympy.Symbol('IRKB'))
                    ).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 2)),
                ]),
                cirq.Moment(operations=[
                    cirq.PhasedXPowGate(phase_exponent=sympy.Symbol('PJOU'),
                                        exponent=0.2081415255258906 *
                                        sympy.Symbol('LKRV')).on(
                                            cirq.GridQubit(0, 2)),
                    (cirq.ISWAP**(0.32860954996781722 * sympy.Symbol('PJOU'))
                    ).on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 3)),
                ]),
                cirq.Moment(operations=[
                    cirq.PhasedXPowGate(phase_exponent=sympy.Symbol('CJKX')).on(
                        cirq.GridQubit(0, 1)),
                    cirq.ZZ.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 3)),
                    (cirq.X**(0.6826594585474709 *
                              sympy.Symbol('HRYV'))).on(cirq.GridQubit(0, 2)),
                ]),
                cirq.Moment(operations=[
                    (cirq.ZZ**(0.18781276022427218 * sympy.Symbol('PJOU'))
                    ).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 3)),
                ]),
                cirq.Moment(operations=[
                    cirq.Y.on(cirq.GridQubit(0, 0)),
                ]),
                cirq.Moment(operations=[
                    cirq.FSimGate(theta=0.13793763138552417 *
                                  sympy.Symbol('CJKX'),
                                  phi=0.13793763138552417 *
                                  sympy.Symbol('PJOU')).
                    on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                    (cirq.ISWAP**(0.028165738453673095 * sympy.Symbol('NASW'))
                    ).on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                ]),
                cirq.Moment(operations=[
                    cirq.FSimGate(theta=0.74356520426349459 *
                                  sympy.Symbol('CJKX'),
                                  phi=0.74356520426349459 *
                                  sympy.Symbol('NASW')).
                    on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 0)),
                ]),
                cirq.Moment(operations=[
                    cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 2)),
                    cirq.SWAP.on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 1)),
                ]),
                cirq.Moment(operations=[
                    cirq.H.on(cirq.GridQubit(0, 3)),
                    cirq.H.on(cirq.GridQubit(0, 2)),
                    cirq.CNOT.on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)),
                ]),
                cirq.Moment(operations=[
                    cirq.CNOT.on(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)),
                    cirq.YY.on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                ]),
                cirq.Moment(operations=[
                    cirq.CZ.on(cirq.GridQubit(0, 1), cirq.GridQubit(0, 0)),
                    cirq.CNOT.on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3)),
                ]),
                cirq.Moment(operations=[
                    cirq.FSimGate(theta=1, phi=1).on(cirq.GridQubit(0, 0),
                                                     cirq.GridQubit(0, 2)),
                    cirq.CNOT.on(cirq.GridQubit(0, 3), cirq.GridQubit(0, 1)),
                ]),
                cirq.Moment(operations=[
                    cirq.FSimGate(theta=1, phi=1).on(cirq.GridQubit(0, 0),
                                                     cirq.GridQubit(0, 3)),
                    cirq.SWAP.on(cirq.GridQubit(0, 2), cirq.GridQubit(0, 1)),
                ]),
                cirq.Moment(operations=[
                    cirq.Y.on(cirq.GridQubit(0, 0)),
                    cirq.PhasedXPowGate(
                        phase_exponent=1.0).on(cirq.GridQubit(0, 2)),
                    cirq.FSimGate(theta=1, phi=1).on(cirq.GridQubit(0, 1),
                                                     cirq.GridQubit(0, 3)),
                ]),
            ])
        ]

        # Decompose programs.
        inputs = util.convert_to_tensor(circuit_batch)
        outputs = tfq_ps_util_ops.tfq_ps_decompose(inputs)
        decomposed_programs = util.from_tensor(outputs)
        self.assertEqual(len(decomposed_programs), len(circuit_batch))

        # Original programs has parameterized ISP, PXP, FSIM, but this result
        # has no such gates at all. All parameterized gates have at most two
        # eigenvalues. There are still ISwap and PhasedX(1.0) because they are
        # not parameterized, which doesn't affect ParameterShift differentiation
        # at all.
        for program in decomposed_programs:
            for moment in program:
                for gate_op in moment:
                    # Consider parameterized gates only
                    if cirq.is_parameterized(gate_op.gate):
                        # Check I. The gate should have _eigen_components.
                        self.assertTrue(
                            hasattr(gate_op.gate, '_eigen_components'))
                        # Check II. The gate should have two eigen values.
                        self.assertEqual(len(gate_op.gate._eigen_components()),
                                         2, gate_op.gate)
        # Now all programs don't have ISWAP & PhasedXPowGate because ISWAP has
        # 3 eigenvalues and PhasedXPowGate doesn't have _eigen_components.
        # Check if two programs are identical.
        rand_resolver = {name: np.random.random() for name in names}
        self.assertAllClose(cirq.unitary(
            cirq.resolve_parameters(circuit_batch[0], rand_resolver)),
                            cirq.unitary(
                                cirq.resolve_parameters(decomposed_programs[0],
                                                        rand_resolver)),
                            atol=1e-5)

    def test_moment_preservation(self):
        """Test Moment-structure preservation."""
        t = sympy.Symbol('t')
        r = sympy.Symbol('r')
        qubits = cirq.GridQubit.rect(1, 6)
        circuit_batch = [
            cirq.Circuit(
                cirq.Moment([cirq.H(q) for q in qubits]),
                cirq.Moment([
                    cirq.X(qubits[4]),
                    cirq.PhasedXPowGate(phase_exponent=np.random.random() *
                                        t).on(qubits[5]),
                    cirq.ISwapPowGate(exponent=np.random.random() * t).on(
                        qubits[0], qubits[1]),
                    cirq.FSimGate(theta=np.random.random() * t,
                                  phi=np.random.random() * r).on(
                                      qubits[2], qubits[3])
                ]), cirq.Moment([cirq.H(q) for q in qubits]))
        ]
        inputs = util.convert_to_tensor(circuit_batch)
        outputs = tfq_ps_util_ops.tfq_ps_decompose(inputs)
        decomposed_programs = util.from_tensor(outputs)
        # Now all programs don't have ISWAP & PhasedXPowGate because ISWAP has
        # 3 eigenvalues and PhasedXPowGate doesn't have _eigen_components.
        # Check if two programs are identical.
        rand_resolver = {'t': np.random.random(), 'r': np.random.random()}
        self.assertAllClose(cirq.unitary(
            cirq.resolve_parameters(circuit_batch[0], rand_resolver)),
                            cirq.unitary(
                                cirq.resolve_parameters(decomposed_programs[0],
                                                        rand_resolver)),
                            atol=1e-5)
        # Check if the Moments are conserved.
        max_decomposed_length = 3
        n_non_decomposed_moments = 2
        self.assertEqual(len(decomposed_programs[0]),
                         n_non_decomposed_moments + max_decomposed_length)
        # Total length of Moments = 5
        # The non-decomposed moments should be the same.
        self.assertEqual(decomposed_programs[0][0], circuit_batch[0][0])
        self.assertEqual(decomposed_programs[0][-1], circuit_batch[0][-1])
        # Check paralellized decompose gates in Moment[1]~[3].
        # The target ops are replaced by the first decomposition gates. It means
        # the first Moment has exactly the same number of gate ops.
        self.assertEqual(len(decomposed_programs[0][1]),
                         len(circuit_batch[0][1]))
        # From the second Moments, the Moments only have decomposition gates.
        # In this example, two ISwapPowGate & one PhasedXPowGate are located.
        # Since PhasedXPowGate, ISwapPowGate, FSimGate has 3, 2, 3 result gates
        # Moment[2] have 3 gate ops and Moment[3] have 2 gate ops.
        self.assertEqual(len(decomposed_programs[0][2]), 3)
        self.assertEqual(len(decomposed_programs[0][3]), 2)

    def test_more_complex_moment_preservation(self):
        """Test Moment-structure preservation."""
        circuit_batch = _complex_test_circuit()
        inputs = util.convert_to_tensor(circuit_batch)
        outputs = tfq_ps_util_ops.tfq_ps_decompose(inputs)
        decomposed_programs = util.from_tensor(outputs)
        # Now all programs don't have ISWAP & PhasedXPowGate because ISWAP has
        # 3 eigenvalues and PhasedXPowGate doesn't have _eigen_components.
        # Check if two programs are identical.
        rand_resolver = {'t': np.random.random(), 'r': np.random.random()}
        for i in range(3):
            self.assertAllClose(cirq.unitary(
                cirq.resolve_parameters(circuit_batch[i], rand_resolver)),
                                cirq.unitary(
                                    cirq.resolve_parameters(
                                        decomposed_programs[i], rand_resolver)),
                                atol=1e-5)
        # Check if the Moments are conserved.
        # Circuit 1.
        max_decomposed_length = 3
        n_non_decomposed_moments = 2
        self.assertEqual(len(decomposed_programs[0]),
                         n_non_decomposed_moments + max_decomposed_length)
        # Total length of Moments = 5
        # The non-decomposed moments should be the same.
        self.assertEqual(decomposed_programs[0][0], circuit_batch[0][0])
        self.assertEqual(decomposed_programs[0][-1], circuit_batch[0][-1])
        # Check paralellized decompose gates in Moment[1]~[3].
        # The target ops are replaced by the first decomposition gates. It means
        # the first Moment has exactly the same number of gate ops.
        self.assertEqual(len(decomposed_programs[0][1]),
                         len(circuit_batch[0][1]))
        # From the second Moments, the Moments only have decomposition gates.
        # In this example, two ISwapPowGate & one PhasedXPowGate are located.
        # Since PhasedXPowGate, ISwapPowGate, FSimGate has 3, 2, 3 result gates
        # Moment[2] have 3 gate ops and Moment[3] have 2 gate ops.
        self.assertEqual(len(decomposed_programs[0][2]), 3)
        self.assertEqual(len(decomposed_programs[0][3]), 2)

        # Circuit 2. two FSimGates.
        self.assertEqual(len(decomposed_programs[1]), 2 * max_decomposed_length)

        # Circuit 3. one PXP between two ISwapPowGates.
        self.assertEqual(len(decomposed_programs[2]), max_decomposed_length)


class PSSymbolReplaceTest(tf.test.TestCase):
    """Tests tfq_ps_symbol_replace."""

    def test_simple_case(self):
        """Test trivial case."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
        )
        inputs = util.convert_to_tensor([circuit])
        symbols = tf.convert_to_tensor(['alpha'])
        new = tf.convert_to_tensor(['new'])
        res = tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, new)
        output = util.from_tensor(res)
        correct_00 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('new'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
        )
        correct_01 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('new'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
        )
        correct_02 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('new'),
        )
        self.assertEqual(correct_00, output[0][0][0])
        self.assertEqual(correct_01, output[0][0][1])
        self.assertEqual(correct_02, output[0][0][2])

    def test_error(self):
        """Ensure that errors happen with bad inputs."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(cirq.X(bit)**(sympy.Symbol('alpha') * 2))
        inputs = util.convert_to_tensor([[circuit]])
        symbols = tf.convert_to_tensor(['test'])
        replacements = tf.convert_to_tensor(['nothing'])
        with self.assertRaisesRegex(Exception,
                                    expected_regex='rank 1. Got rank 2.'):
            tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, replacements)

        inputs = tf.convert_to_tensor(['junk'])
        with self.assertRaisesRegex(Exception,
                                    expected_regex='Unparseable proto:'):
            tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, replacements)

        inputs = util.convert_to_tensor([circuit])
        symbols = tf.convert_to_tensor([['test']])
        replacements = tf.convert_to_tensor(['nothing'])
        with self.assertRaisesRegex(Exception,
                                    expected_regex='rank 1. Got rank 2.'):
            tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, replacements)

        symbols = tf.convert_to_tensor(['test'])
        replacements = tf.convert_to_tensor([['nothing']])
        with self.assertRaisesRegex(Exception,
                                    expected_regex='rank 1. Got rank 2.'):
            tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, replacements)

        symbols = tf.convert_to_tensor(['test'])
        replacements = tf.convert_to_tensor(['nothing', 'too long'])
        with self.assertRaisesRegex(
                Exception,
                expected_regex=
                'symbols.shape is not equal to replacement_symbols.shape'):
            tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, replacements)

    def test_weight_coefficient(self):
        """Test that scalar multiples of trivial case work."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(
            cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
            cirq.Y(bit)**(sympy.Symbol('alpha') * 3.0),
            cirq.Z(bit)**(sympy.Symbol('alpha') * 4.0),
        )
        inputs = util.convert_to_tensor([circuit])
        symbols = tf.convert_to_tensor(['alpha'])
        new = tf.convert_to_tensor(['new'])
        res = tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, new)
        output = util.from_tensor(res)
        correct_00 = cirq.Circuit(
            cirq.X(bit)**(sympy.Symbol('new') * 2.0),
            cirq.Y(bit)**(sympy.Symbol('alpha') * 3.0),
            cirq.Z(bit)**(sympy.Symbol('alpha') * 4.0),
        )
        correct_01 = cirq.Circuit(
            cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
            cirq.Y(bit)**(sympy.Symbol('new') * 3.0),
            cirq.Z(bit)**(sympy.Symbol('alpha') * 4.0),
        )
        correct_02 = cirq.Circuit(
            cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
            cirq.Y(bit)**(sympy.Symbol('alpha') * 3.0),
            cirq.Z(bit)**(sympy.Symbol('new') * 4.0),
        )
        self.assertEqual(correct_00, output[0][0][0])
        self.assertEqual(correct_01, output[0][0][1])
        self.assertEqual(correct_02, output[0][0][2])

    def test_simple_pad(self):
        """Test simple padding."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
        )
        circuit2 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('beta'),
            cirq.Y(bit)**sympy.Symbol('beta'),
            cirq.Z(bit)**sympy.Symbol('beta'),
        )
        circuit3 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
        )
        inputs = util.convert_to_tensor([circuit, circuit2, circuit3])
        symbols = tf.convert_to_tensor(['alpha', 'beta', 'gamma'])
        new = tf.convert_to_tensor(['new', 'old', 'nothing'])
        res = tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, new)
        output = util.from_tensor(res)

        correct_00 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('new'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
        )
        correct_01 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('new'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
        )
        correct_02 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('new'),
        )
        self.assertEqual(correct_00, output[0][0][0])
        self.assertEqual(correct_01, output[0][0][1])
        self.assertEqual(correct_02, output[0][0][2])

        self.assertEqual(correct_00, output[2][0][0])
        self.assertEqual(correct_01, output[2][0][1])
        self.assertEqual(correct_02, output[2][0][2])

        correct_10 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('old'),
            cirq.Y(bit)**sympy.Symbol('beta'),
            cirq.Z(bit)**sympy.Symbol('beta'),
        )
        correct_11 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('beta'),
            cirq.Y(bit)**sympy.Symbol('old'),
            cirq.Z(bit)**sympy.Symbol('beta'),
        )
        correct_12 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('beta'),
            cirq.Y(bit)**sympy.Symbol('beta'),
            cirq.Z(bit)**sympy.Symbol('old'),
        )
        self.assertEqual(correct_10, output[1][1][0])
        self.assertEqual(correct_11, output[1][1][1])
        self.assertEqual(correct_12, output[1][1][2])

        correct_20 = cirq.Circuit()
        correct_21 = cirq.Circuit()
        correct_22 = cirq.Circuit()
        self.assertEqual(correct_20, output[2][2][0])
        self.assertEqual(correct_21, output[2][2][1])
        self.assertEqual(correct_22, output[2][2][2])

        correct = cirq.Circuit()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and (not (i == 2 and j == 0)):
                        self.assertEqual(correct, output[i][j][k])

    def test_complex_pad(self):
        """Test trickier padding."""
        bit = cirq.GridQubit(0, 0)
        bit2 = cirq.GridQubit(0, 1)
        circuit = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        circuit2 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('beta'),
            cirq.Y(bit)**sympy.Symbol('beta'),
            cirq.Z(bit)**sympy.Symbol('beta'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        circuit3 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        inputs = util.convert_to_tensor([circuit, circuit2, circuit3])
        symbols = tf.convert_to_tensor(['alpha', 'beta', 'gamma'])
        new = tf.convert_to_tensor(['new', 'old', 'nothing'])
        res = tfq_ps_util_ops.tfq_ps_symbol_replace(inputs, symbols, new)
        output = util.from_tensor(res)

        correct_000 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('new'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        correct_001 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('new'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        correct_002 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('new'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        correct_003 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('alpha'),
            cirq.Y(bit)**sympy.Symbol('alpha'),
            cirq.Z(bit)**sympy.Symbol('alpha'),
            cirq.XX(bit, bit2)**sympy.Symbol('new'))

        self.assertEqual(correct_000, output[0][0][0])
        self.assertEqual(correct_001, output[0][0][1])
        self.assertEqual(correct_002, output[0][0][2])
        self.assertEqual(correct_003, output[0][0][3])

        self.assertEqual(correct_000, output[2][0][0])
        self.assertEqual(correct_001, output[2][0][1])
        self.assertEqual(correct_002, output[2][0][2])
        self.assertEqual(correct_003, output[2][0][3])

        correct_110 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('old'),
            cirq.Y(bit)**sympy.Symbol('beta'),
            cirq.Z(bit)**sympy.Symbol('beta'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        correct_111 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('beta'),
            cirq.Y(bit)**sympy.Symbol('old'),
            cirq.Z(bit)**sympy.Symbol('beta'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        correct_112 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('beta'),
            cirq.Y(bit)**sympy.Symbol('beta'),
            cirq.Z(bit)**sympy.Symbol('old'),
            cirq.XX(bit, bit2)**sympy.Symbol('alpha'))
        correct_113 = cirq.Circuit()

        self.assertEqual(correct_110, output[1][1][0])
        self.assertEqual(correct_111, output[1][1][1])
        self.assertEqual(correct_112, output[1][1][2])
        self.assertEqual(correct_113, output[1][1][3])

        correct_100 = cirq.Circuit(
            cirq.X(bit)**sympy.Symbol('beta'),
            cirq.Y(bit)**sympy.Symbol('beta'),
            cirq.Z(bit)**sympy.Symbol('beta'),
            cirq.XX(bit, bit2)**sympy.Symbol('new'))
        correct_101 = cirq.Circuit()
        correct_102 = cirq.Circuit()
        correct_103 = cirq.Circuit()

        self.assertEqual(correct_100, output[1][0][0])
        self.assertEqual(correct_101, output[1][0][1])
        self.assertEqual(correct_102, output[1][0][2])
        self.assertEqual(correct_103, output[1][0][3])

        correct_220 = cirq.Circuit()
        correct_221 = cirq.Circuit()
        correct_222 = cirq.Circuit()
        correct_223 = cirq.Circuit()

        self.assertEqual(correct_220, output[2][2][0])
        self.assertEqual(correct_221, output[2][2][1])
        self.assertEqual(correct_222, output[2][2][2])
        self.assertEqual(correct_223, output[2][2][3])

        correct = cirq.Circuit()
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if i != j and (not (i == 2 and j == 0)) \
                        and (not (i == 1 and j == 0)):
                        self.assertEqual(correct, output[i][j][k])


class PSWeightsFromSymbolTest(tf.test.TestCase):
    """Tests tfq_ps_weights_from_symbols."""

    def test_simple(self):
        """Ensure that weight extraction works."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(cirq.X(bit)**(sympy.Symbol('alpha') * 2))
        inputs = util.convert_to_tensor([circuit])
        symbols = tf.convert_to_tensor(['alpha'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(res, np.array([[[2.0]]]))

    def test_empty(self):
        """Test empty circuit. and symbol free circuit. does nothing."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(cirq.X(bit))
        circuit2 = cirq.Circuit()
        inputs = util.convert_to_tensor([circuit, circuit2])
        symbols = tf.convert_to_tensor(['alpha'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(res, np.array([[[]], [[]]]))

    def test_rotation_gates(self):
        """Test that rotation gates work."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(cirq.Rx(sympy.Symbol('alpha') * 5.0)(bit))
        inputs = util.convert_to_tensor([circuit])
        symbols = tf.convert_to_tensor(['alpha'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(res, np.array([[[5.0 / np.pi]]]))

    def test_error(self):
        """Ensure if a symbol can't be found the op errors."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(cirq.X(bit)**(sympy.Symbol('delta') * 2))
        inputs = util.convert_to_tensor([circuit])
        symbols = tf.convert_to_tensor(['alpha', 'delta'])
        tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        symbols = tf.convert_to_tensor(['alpha'])
        with self.assertRaisesRegex(Exception, expected_regex='sympy.Symbol'):
            tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)

        symbols = tf.convert_to_tensor([['delta']])
        with self.assertRaisesRegex(Exception,
                                    expected_regex='rank 1. Got rank 2.'):
            tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)

        inputs = tf.convert_to_tensor(['junk'])
        symbols = tf.convert_to_tensor(['delta'])
        with self.assertRaisesRegex(Exception,
                                    expected_regex='Unparseable proto:'):
            tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)

        inputs = util.convert_to_tensor([[circuit]])
        with self.assertRaisesRegex(Exception,
                                    expected_regex='rank 1. Got rank 2.'):
            tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)

    def test_many_values(self):
        """Ensure that padding with few symbols and many values works."""
        bit = cirq.GridQubit(0, 0)
        circuits = [
            cirq.Circuit(
                cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
                cirq.Y(bit)**(sympy.Symbol('alpha') * 3.0),
                cirq.Z(bit)**(sympy.Symbol('alpha')),
                cirq.X(bit)**(sympy.Symbol('alpha') * 4.0)),
            cirq.Circuit(cirq.X(bit)**(sympy.Symbol('alpha') * 9.0)),
            cirq.Circuit(cirq.X(bit)**sympy.Symbol('beta'))
        ]
        inputs = util.convert_to_tensor(circuits)
        symbols = tf.convert_to_tensor(['alpha', 'beta'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(
            res,
            np.array([[[2.0, 3.0, 1.0, 4.0], [0.0, 0.0, 0.0, 0.0]],
                      [[9.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                      [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]))

    def test_many_symbols(self):
        """Ensure that padding with few values and many symbols works."""
        bit = cirq.GridQubit(0, 0)
        circuits = [
            cirq.Circuit(cirq.X(bit)**(sympy.Symbol('alpha') * 2.0)),
            cirq.Circuit(cirq.X(bit)**(sympy.Symbol('beta') * 6)),
            cirq.Circuit(cirq.X(bit)**(sympy.Symbol('alpha') * 5.0)),
            cirq.Circuit(cirq.X(bit)**(sympy.Symbol('gamma') * 8)),
            cirq.Circuit(cirq.X(bit)**(sympy.Symbol('delta') * 9))
        ]
        inputs = util.convert_to_tensor(circuits)
        symbols = tf.convert_to_tensor(['alpha', 'beta', 'gamma', 'delta'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(
            res,
            np.array([[[2.0], [0.0], [0.0], [0.0]], [[0.0], [6.0], [0.0],
                                                     [0.0]],
                      [[5.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [8.0],
                                                     [0.0]],
                      [[0.0], [0.0], [0.0], [9.0]]]))

    def test_out_of_order(self):
        """Test that discovery order of symbols in circuits doesn't matter."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(
            cirq.X(bit)**(sympy.Symbol('alpha') * 2),
            cirq.Y(bit)**(sympy.Symbol('beta') * 3))
        inputs = util.convert_to_tensor([circuit])
        symbols = tf.convert_to_tensor(['alpha', 'beta'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(res, np.array([[[2.0], [3.0]]]))
        symbols = tf.convert_to_tensor(['beta', 'alpha'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(res, np.array([[[3.0], [2.0]]]))

    def test_padding(self):
        """Ensure that the padding is correct in a complex example."""
        bit = cirq.GridQubit(0, 0)
        circuits = [
            cirq.Circuit(
                cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
                cirq.Y(bit)**(sympy.Symbol('alpha') * 3.0),
                cirq.Z(bit)**(sympy.Symbol('beta') * 4.0),
            ),
            cirq.Circuit(
                cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
                cirq.Y(bit)**(sympy.Symbol('beta') * 3.0),
                cirq.Z(bit)**(sympy.Symbol('beta') * 4.0),
            ),
            cirq.Circuit(
                cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
                cirq.Y(bit)**(sympy.Symbol('beta') * 3.0),
                cirq.Z(bit)**(sympy.Symbol('gamma') * 4.0),
            )
        ]
        inputs = util.convert_to_tensor(circuits)
        symbols = tf.convert_to_tensor(['alpha', 'beta', 'gamma'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(
            res,
            np.array([[[2.0, 3.0], [4.0, 0.0], [0.0, 0.0]],
                      [[2.0, 0.0], [3.0, 4.0], [0.0, 0.0]],
                      [[2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]))

    def test_padding_with_non_parameterized_gates(self):
        """Ensure that the padding is correct in a complex example."""
        bit = cirq.GridQubit(0, 0)
        circuits = [
            cirq.Circuit(
                cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
                cirq.Y(bit)**3.0,
                cirq.Z(bit)**(sympy.Symbol('beta') * 4.0),
            ),
            cirq.Circuit(
                cirq.X(bit)**(sympy.Symbol('alpha') * 2.0),
                cirq.Y(bit)**(sympy.Symbol('beta') * 3.0),
                cirq.Z(bit)**4.0,
            ),
            cirq.Circuit(
                cirq.X(bit)**2.0,
                cirq.Y(bit)**(sympy.Symbol('beta') * 3.0),
                cirq.Z(bit)**(sympy.Symbol('gamma') * 4.0),
            )
        ]
        inputs = util.convert_to_tensor(circuits)
        symbols = tf.convert_to_tensor(['alpha', 'beta', 'gamma'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        self.assertAllClose(
            res,
            np.array([[[2.0], [4.0], [0.0]], [[2.0], [3.0], [0.0]],
                      [[0.0], [3.0], [4.0]]]))

    def test_ignorance(self):
        """Test ignorance of ISP, PXP, FSIM gates."""
        circuit_batch = _complex_test_circuit()
        inputs = util.convert_to_tensor(circuit_batch)
        symbols = tf.convert_to_tensor(['r', 't'])
        res = tfq_ps_util_ops.tfq_ps_weights_from_symbols(inputs, symbols)
        # Because there are no weights to be gathered, the last dimension = 0
        self.assertAllClose(tf.shape(res), [len(circuit_batch), 2, 0])


if __name__ == "__main__":
    tf.test.main()
