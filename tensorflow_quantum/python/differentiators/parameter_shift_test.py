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
"""Basic tests for the ParameterShift differentiator"""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys

NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import sympy
import cirq

from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import parameter_shift
from tensorflow_quantum.core.ops import circuit_execution_ops


def _simple_op_inputs():
    qubit = cirq.GridQubit(0, 0)
    symbol = 'alpha'
    circuit = cirq.Circuit(cirq.Y(qubit)**sympy.Symbol(symbol))
    op = cirq.X(qubit)
    value = 0.3
    n_samples = 2000

    # Return inputs prepped for expectation ops.
    # circuit, symbol_names, values, ops, n_samples
    # along with expected feedforward expectation
    # and expected gradient.
    return (util.convert_to_tensor([circuit]), tf.convert_to_tensor([symbol]),
            tf.convert_to_tensor([[value]]), util.convert_to_tensor([[op]]),
            tf.convert_to_tensor([[n_samples]]),
            tf.convert_to_tensor([[np.sin(np.pi * value)]]),
            tf.convert_to_tensor([[np.pi * np.cos(np.pi * value)]]))


class ParameterShiftTest(tf.test.TestCase, parameterized.TestCase):
    """Test the ParameterShift Differentiator will run end to end."""

    def test_parameter_shift_analytic(self):
        """Test if ParameterShift.differentiate_analytical doesn't crash before
        running."""
        programs, names, values, ops, _, true_f, true_g = \
        _simple_op_inputs()

        ps = parameter_shift.ParameterShift()
        op = ps.generate_differentiable_op(
            analytic_op=circuit_execution_ops.get_expectation_op())

        with tf.GradientTape() as g:
            g.watch(values)
            expectations = op(programs, names, values, ops)
        grads = g.gradient(expectations, values)
        self.assertAllClose(expectations, true_f, atol=1e-2, rtol=1e-2)
        self.assertAllClose(grads, true_g, atol=1e-2, rtol=1e-2)

    def test_parameter_shift_sampled(self):
        """Test if ParameterShift.differentiate_sampled doesn't crash before
        running."""
        programs, names, values, ops, n_samples, true_f, true_g = \
        _simple_op_inputs()
        ps = parameter_shift.ParameterShift()
        op = ps.generate_differentiable_op(
            sampled_op=circuit_execution_ops.get_sampled_expectation_op())

        with tf.GradientTape() as g:
            g.watch(values)
            expectations = op(programs, names, values, ops, n_samples)
        grads = g.gradient(expectations, values)
        self.assertAllClose(expectations, true_f, atol=1e-1, rtol=1e-1)
        self.assertAllClose(grads, true_g, atol=1e-1, rtol=1e-1)

    def test_get_gradient_circuits(self):
        """Test that the correct objects are returned."""

        diff = parameter_shift.ParameterShift()

        # Circuits to differentiate.
        symbols = [sympy.Symbol("s0"), sympy.Symbol("s1")]
        q0 = cirq.GridQubit(0, 0)
        q1 = cirq.GridQubit(1, 2)
        input_programs = util.convert_to_tensor([
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q0)**symbols[0],
                cirq.ry(symbols[1])(q1)),
            cirq.Circuit(cirq.Y(q1)**symbols[1]),
        ])
        input_symbol_names = tf.constant([str(s) for s in symbols])
        input_symbol_values = tf.constant([[1.5, -2.7], [-0.3, 0.9]])

        # First, for each symbol `s`, check how many times `s` appears in each
        # program `p`, `n_ps`. Let `n_param_gates` be the maximum of `n_ps` over
        # all symbols and programs. Then, the shape of `batch_programs` will be
        # [n_programs, n_symbols * n_param_gates * n_shifts], where `n_shifts`
        # is 2 because we decompose into gates with 2 eigenvalues. For row index
        # `p` we have for column indices between `i * n_param_gates * n_shifts`
        # and `(i + 1) * n_param_gates * n_shifts`, the first `n_pi * 2`
        # programs are parameter shifted versions of `input_programs[p]` and the
        # remaining programs are empty.
        # Here, `n_param_gates` is 2.
        impurity_symbol_name = "_impurity_for_param_shift"
        impurity_symbol = sympy.Symbol(impurity_symbol_name)
        expected_batch_programs_0 = util.convert_to_tensor([
            cirq.Circuit(
                cirq.X(q0)**impurity_symbol,
                cirq.Y(q0)**symbols[0],
                cirq.ry(symbols[1])(q1)),
            cirq.Circuit(
                cirq.X(q0)**impurity_symbol,
                cirq.Y(q0)**symbols[0],
                cirq.ry(symbols[1])(q1)),
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q0)**impurity_symbol,
                cirq.ry(symbols[1])(q1)),
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q0)**impurity_symbol,
                cirq.ry(symbols[1])(q1)),
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q0)**symbols[0],
                cirq.ry(impurity_symbol)(q1)),
            cirq.Circuit(
                cirq.X(q0)**symbols[0],
                cirq.Y(q0)**symbols[0],
                cirq.ry(impurity_symbol)(q1)),
            cirq.Circuit(),
            cirq.Circuit()
        ])
        expected_batch_programs_1 = util.convert_to_tensor([
            cirq.Circuit(),
            cirq.Circuit(),
            cirq.Circuit(),
            cirq.Circuit(),
            cirq.Circuit(cirq.Y(q1)**impurity_symbol),
            cirq.Circuit(cirq.Y(q1)**impurity_symbol),
            cirq.Circuit(),
            cirq.Circuit()
        ])
        expected_batch_programs = tf.stack(
            [expected_batch_programs_0, expected_batch_programs_1])

        # The new symbols are the old ones, with an extra used for shifting.
        expected_new_symbol_names = tf.concat(
            [input_symbol_names,
             tf.constant([impurity_symbol_name])], 0)

        # The batch symbol values are the input symbol values, tiled and with
        # shifted values appended. Locations that have empty programs should
        # also have zero for the shift.
        # The shifted values are the original value plus 1/2 divided by the
        # `exponent_scalar` of the gate.
        expected_batch_symbol_values = tf.constant(
            [[[1.5, -2.7, 1.5 + 0.5], [1.5, -2.7, 1.5 - 0.5],
              [1.5, -2.7, 1.5 + 0.5], [1.5, -2.7, 1.5 - 0.5],
              [1.5, -2.7, -2.7 + np.pi / 2], [1.5, -2.7, -2.7 - np.pi / 2],
              [1.5, -2.7, -2.7], [1.5, -2.7, -2.7]],
             [[-0.3, 0.9, -0.3], [-0.3, 0.9, -0.3], [-0.3, 0.9, -0.3],
              [-0.3, 0.9, -0.3], [-0.3, 0.9, 0.9 + 0.5], [-0.3, 0.9, 0.9 - 0.5],
              [-0.3, 0.9, 0.9], [-0.3, 0.9, 0.9]]])

        # Empty program locations are given zero weight.
        expected_batch_weights = tf.constant(
            [[[np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2],
              [0.5, -0.5, 0.0, 0.0]],
             [[0.0, 0.0, 0.0, 0.0], [np.pi / 2, -np.pi / 2, 0.0, 0.0]]])

        expected_batch_mapper = tf.constant([[[0, 1, 2, 3], [4, 5, 6, 7]],
                                             [[0, 1, 2, 3], [4, 5, 6, 7]]])

        (test_batch_programs, test_new_symbol_names, test_batch_symbol_values,
         test_batch_weights, test_batch_mapper) = diff.get_gradient_circuits(
             input_programs, input_symbol_names, input_symbol_values)
        for i in range(tf.shape(input_programs)[0]):
            self.assertAllEqual(util.from_tensor(expected_batch_programs[i]),
                                util.from_tensor(test_batch_programs[i]))
        self.assertAllEqual(expected_new_symbol_names, test_new_symbol_names)
        self.assertAllClose(expected_batch_symbol_values,
                            test_batch_symbol_values,
                            atol=1e-5)
        self.assertAllClose(expected_batch_weights,
                            test_batch_weights,
                            atol=1e-5)
        self.assertAllEqual(expected_batch_mapper, test_batch_mapper)

    def test_parameter_shift_pisp_shared_moment(self):
        """Test analytic gradients when PISP precedes PXP and FSIM in Moment."""
        qubits = cirq.LineQubit.range(5)
        names = ['s0', 's1', 's2', 's3', 's4']
        s0, s1, s2, s3, s4 = sympy.symbols(names)
        prep = cirq.Moment(
            cirq.rx(0.73)(qubits[0]),
            cirq.ry(-0.91)(qubits[1]),
            cirq.rx(0.64)(qubits[2]),
            cirq.ry(1.13)(qubits[3]),
            cirq.rx(-1.07)(qubits[4]),
        )
        pisp = cirq.PhasedISwapPowGate(phase_exponent=1.3 * s1,
                                       exponent=0.7 * s0).on(
                                           qubits[0], qubits[1])
        pxp = cirq.PhasedXPowGate(phase_exponent=0.9 * s2,
                                  exponent=0.5 * s0).on(qubits[2])
        fsim = cirq.FSimGate(theta=0.8 * s3,
                             phi=1.1 * s4).on(qubits[3], qubits[4])
        joint_circuit = cirq.Circuit(prep, cirq.Moment(pisp, pxp, fsim))
        split_circuit = cirq.Circuit(prep, cirq.Moment(pisp), cirq.Moment(pxp),
                                     cirq.Moment(fsim))
        psum = (
            0.7 * cirq.X(qubits[0]) * cirq.Z(qubits[1]) * cirq.Y(qubits[2]) +
            0.9 * cirq.Y(qubits[0]) * cirq.X(qubits[3]) +
            0.8 * cirq.Z(qubits[1]) * cirq.X(qubits[2]) * cirq.Y(qubits[4]))
        raw_vals = np.array([0.37, -0.41, 0.63, -0.52, 0.49], dtype=np.float64)

        ps_op = parameter_shift.ParameterShift().generate_differentiable_op(
            analytic_op=circuit_execution_ops.get_expectation_op())
        programs = util.convert_to_tensor([joint_circuit, split_circuit])
        symbol_names = tf.constant(names)
        ops = util.convert_to_tensor([[psum], [psum]])
        joint_vals = tf.constant([raw_vals], dtype=tf.float32)
        split_vals = tf.constant([raw_vals], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch([joint_vals, split_vals])
            joint_exp = ps_op(programs[:1], symbol_names, joint_vals, ops[:1])
            split_exp = ps_op(programs[1:], symbol_names, split_vals, ops[1:])
        joint_grad, split_grad = tape.gradient([joint_exp, split_exp],
                                               [joint_vals, split_vals])

        qubit_map = {q: i for i, q in enumerate(qubits)}

        def eval_exp(vec):
            resolved = cirq.resolve_parameters(joint_circuit,
                                               dict(zip(names, vec)))
            state = cirq.Simulator(dtype=np.complex128).simulate(
                resolved, qubit_order=qubits).final_state_vector
            return psum.expectation_from_state_vector(state, qubit_map).real

        eps = 1e-5
        ref_grad = np.array([(eval_exp(raw_vals + eps * basis) -
                              eval_exp(raw_vals - eps * basis)) / (2.0 * eps)
                             for basis in np.eye(len(names))])
        self.assertGreater(np.min(np.abs(ref_grad)), 1e-2)
        self.assertAllClose(joint_exp, [[eval_exp(raw_vals)]], atol=1e-5)
        self.assertAllClose(joint_grad, split_grad, atol=1e-5, rtol=1e-5)
        self.assertAllClose(joint_grad, [ref_grad], atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    tf.test.main()
