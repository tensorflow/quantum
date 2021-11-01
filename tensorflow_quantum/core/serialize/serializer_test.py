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
"""Module to test serialization core."""
import copy
import numpy as np
import sympy
import tensorflow as tf

import cirq
from absl.testing import parameterized
from tensorflow_quantum.core.proto import pauli_sum_pb2
from tensorflow_quantum.core.proto import program_pb2
from tensorflow_quantum.core.serialize import serializer


def _build_op_proto(gate_id, arg_names, arg_vals, qubit_ids):
    """Helper function to generate proto for a given circuit spec.

    Understand how it works from this example:

    _build_op_proto("HP",
                      ['exponent', 'global_shift'],
                      ['alpha', 0.0],
                      ['0_0'])

    would produce the following:

    language {
      gate_set: "tfq_gate_set"
    }
    circuit {
      scheduling_strategy: MOMENT_BY_MOMENT
      moments {
        operations {
          gate {
            id: "HP"
          }
          args {
            key: "global_shift"
            value {
              arg_value {
                float_value: 0.0
              }
            }
          }
          args {
            key: "exponent"
            value {
              symbol: "alpha"
            }
          }
          args {
            key: "control_qubits"
            value {
              arg_value: ""
            }
          }
          args {
            key: "control_values"
            value {
              arg_value: ""
            }
          }
          qubits {
            id: "0_0"
          }
        }
      }
    }
    """
    program_proto = program_pb2.Program()
    program_proto.language.gate_set = 'tfq_gate_set'

    circuit_proto = program_proto.circuit
    circuit_proto.scheduling_strategy = circuit_proto.MOMENT_BY_MOMENT
    circuit_proto.moments.add(operations=[program_pb2.Operation(
        gate = program_pb2.Gate(id=gate_id),
        args = {arg_names[i]: (program_pb2.Arg(symbol=arg_vals[i]) \
        if isinstance(arg_vals[i], str) else \
            program_pb2.Arg(
                arg_value=program_pb2.ArgValue(
                    float_value=np.round(float(arg_vals[i]), 6)))) \
                for i in range(len(arg_vals))},
        qubits=[program_pb2.Qubit(
            id=q_id) for q_id in qubit_ids])])

    # Add in empty control information
    t = program_proto.circuit.moments[0].operations[0]
    t.args['control_qubits'].arg_value.string_value = ''
    t.args['control_values'].arg_value.string_value = ''

    return program_proto


def _make_controlled_gate_proto(program_proto, control_qubits, control_values):
    """Turn a gate proto (from above) into a controlled gate proto.

    inserts control_qubits and control_values into gate args map.
    """
    t = program_proto.circuit.moments[0].operations[0]
    t.args['control_qubits'].arg_value.string_value = control_qubits
    t.args['control_values'].arg_value.string_value = control_values
    return program_proto


def _make_controlled_circuit(circuit, control_qubits, control_values):
    new_circuit = cirq.Circuit()
    for moment in circuit:
        for op in moment:
            new_op = op
            for qb, v in zip(control_qubits[::-1], control_values[::-1]):
                new_op = new_op.controlled_by(qb, control_values=[v])
            new_circuit += new_op
    return new_circuit


def _get_circuit_proto_pairs(qubit_type='grid'):
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    q0_str = '0_0'
    q1_str = '0_1'

    if qubit_type == 'line':
        q0 = cirq.LineQubit(0)
        q1 = cirq.LineQubit(1)
        q0_str = '0'
        q1_str = '1'

    pairs = [
        # HPOW and aliases.
        (cirq.Circuit(cirq.HPowGate(exponent=0.3)(q0)),
         _build_op_proto("HP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str])),
        (cirq.Circuit(cirq.HPowGate(exponent=sympy.Symbol('alpha'))(q0)),
         _build_op_proto("HP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str])),
        (cirq.Circuit(cirq.HPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0)),
         _build_op_proto("HP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str])),
        (cirq.Circuit(cirq.H(q0)),
         _build_op_proto("HP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str])),

        # XPOW and aliases.
        (cirq.Circuit(cirq.XPowGate(exponent=0.3)(q0)),
         _build_op_proto("XP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str])),
        (cirq.Circuit(cirq.XPowGate(exponent=sympy.Symbol('alpha'))(q0)),
         _build_op_proto("XP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str])),
        (cirq.Circuit(cirq.XPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0)),
         _build_op_proto("XP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str])),
        (cirq.Circuit(cirq.X(q0)),
         _build_op_proto("XP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str])),

        # YPOW and aliases
        (cirq.Circuit(cirq.YPowGate(exponent=0.3)(q0)),
         _build_op_proto("YP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str])),
        (cirq.Circuit(cirq.YPowGate(exponent=sympy.Symbol('alpha'))(q0)),
         _build_op_proto("YP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str])),
        (cirq.Circuit(cirq.YPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0)),
         _build_op_proto("YP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str])),
        (cirq.Circuit(cirq.Y(q0)),
         _build_op_proto("YP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str])),

        # ZPOW and aliases.
        (cirq.Circuit(cirq.ZPowGate(exponent=0.3)(q0)),
         _build_op_proto("ZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str])),
        (cirq.Circuit(cirq.ZPowGate(exponent=sympy.Symbol('alpha'))(q0)),
         _build_op_proto("ZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str])),
        (cirq.Circuit(cirq.ZPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0)),
         _build_op_proto("ZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str])),
        (cirq.Circuit(cirq.Z(q0)),
         _build_op_proto("ZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str])),

        # XXPow and aliases
        (cirq.Circuit(cirq.XXPowGate(exponent=0.3)(q0, q1)),
         _build_op_proto("XXP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.XXPowGate(exponent=sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("XXP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.XXPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("XXP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.XX(q0, q1)),
         _build_op_proto("XXP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str, q1_str])),

        # YYPow and aliases
        (cirq.Circuit(cirq.YYPowGate(exponent=0.3)(q0, q1)),
         _build_op_proto("YYP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.YYPowGate(exponent=sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("YYP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.YYPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("YYP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.YY(q0, q1)),
         _build_op_proto("YYP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str, q1_str])),

        # ZZPow and aliases
        (cirq.Circuit(cirq.ZZPowGate(exponent=0.3)(q0, q1)),
         _build_op_proto("ZZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.ZZPowGate(exponent=sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("ZZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.ZZPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("ZZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.ZZ(q0, q1)),
         _build_op_proto("ZZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str, q1_str])),

        # CZPow and aliases
        (cirq.Circuit(cirq.CZPowGate(exponent=0.3)(q0, q1)),
         _build_op_proto("CZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.CZPowGate(exponent=sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("CZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.CZPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("CZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.CZ(q0, q1)),
         _build_op_proto("CZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str, q1_str])),

        # CNOTPow and aliases
        (cirq.Circuit(cirq.CNotPowGate(exponent=0.3)(q0, q1)),
         _build_op_proto("CNP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.CNotPowGate(exponent=sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("CNP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.CNotPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("CNP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.CNOT(q0, q1)),
         _build_op_proto("CNP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str, q1_str])),

        # SWAPPow and aliases
        (cirq.Circuit(cirq.SwapPowGate(exponent=0.3)(q0, q1)),
         _build_op_proto("SP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.SwapPowGate(exponent=sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("SP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.SwapPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("SP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.SWAP(q0, q1)),
         _build_op_proto("SP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str, q1_str])),

        # ISWAPPow and aliases
        (cirq.Circuit(cirq.ISwapPowGate(exponent=0.3)(q0, q1)),
         _build_op_proto("ISP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [0.3, 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.ISwapPowGate(exponent=sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("ISP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 1.0, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.ISwapPowGate(exponent=3.1 * sympy.Symbol('alpha'))(q0, q1)),
         _build_op_proto("ISP", ['exponent', 'exponent_scalar', 'global_shift'],
                         ['alpha', 3.1, 0.0], [q0_str, q1_str])),
        (cirq.Circuit(cirq.ISWAP(q0, q1)),
         _build_op_proto("ISP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, 0.0], [q0_str, q1_str])),

        # PhasedXPow and aliases
        (cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=0.9,
                                exponent=0.3,
                                global_shift=0.2)(q0)),
         _build_op_proto("PXP", [
             'phase_exponent', 'phase_exponent_scalar', 'exponent',
             'exponent_scalar', 'global_shift'
         ], [0.9, 1.0, 0.3, 1.0, 0.2], [q0_str])),
        (cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=sympy.Symbol('alpha'),
                                exponent=0.3)(q0)),
         _build_op_proto("PXP", [
             'phase_exponent', 'phase_exponent_scalar', 'exponent',
             'exponent_scalar', 'global_shift'
         ], ['alpha', 1.0, 0.3, 1.0, 0.0], [q0_str])),
        (cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=3.1 * sympy.Symbol('alpha'),
                                exponent=0.3)(q0)),
         _build_op_proto("PXP", [
             'phase_exponent', 'phase_exponent_scalar', 'exponent',
             'exponent_scalar', 'global_shift'
         ], ['alpha', 3.1, 0.3, 1.0, 0.0], [q0_str])),
        (cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=0.9,
                                exponent=sympy.Symbol('beta'))(q0)),
         _build_op_proto("PXP", [
             'phase_exponent', 'phase_exponent_scalar', 'exponent',
             'exponent_scalar', 'global_shift'
         ], [0.9, 1.0, 'beta', 1.0, 0.0], [q0_str])),
        (cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=0.9,
                                exponent=5.1 * sympy.Symbol('beta'))(q0)),
         _build_op_proto("PXP", [
             'phase_exponent', 'phase_exponent_scalar', 'exponent',
             'exponent_scalar', 'global_shift'
         ], [0.9, 1.0, 'beta', 5.1, 0.0], [q0_str])),
        (cirq.Circuit(
            cirq.PhasedXPowGate(phase_exponent=3.1 * sympy.Symbol('alpha'),
                                exponent=5.1 * sympy.Symbol('beta'))(q0)),
         _build_op_proto("PXP", [
             'phase_exponent', 'phase_exponent_scalar', 'exponent',
             'exponent_scalar', 'global_shift'
         ], ['alpha', 3.1, 'beta', 5.1, 0.0], [q0_str])),

        # RX, RY, RZ with symbolization is tested in special cases as the
        # string comparison of the float converted sympy.pi does not happen
        # smoothly. See: test_serialize_deserialize_special_case_one_qubit
        (cirq.Circuit(cirq.rx(np.pi)(q0)),
         _build_op_proto("XP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, -0.5], [q0_str])),
        (cirq.Circuit(cirq.ry(np.pi)(q0)),
         _build_op_proto("YP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, -0.5], [q0_str])),
        (cirq.Circuit(cirq.rz(np.pi)(q0)),
         _build_op_proto("ZP", ['exponent', 'exponent_scalar', 'global_shift'],
                         [1.0, 1.0, -0.5], [q0_str])),

        # Identity
        (cirq.Circuit(cirq.I(q0)),
         _build_op_proto("I", ['unused'], [True], [q0_str])),

        # FSimGate
        (cirq.Circuit(cirq.FSimGate(theta=0.1, phi=0.2)(q0, q1)),
         _build_op_proto("FSIM", ['theta', 'theta_scalar', 'phi', 'phi_scalar'],
                         [0.1, 1.0, 0.2, 1.0], [q0_str, q1_str])),
        (cirq.Circuit(
            cirq.FSimGate(theta=2.1 * sympy.Symbol("alpha"),
                          phi=1.3 * sympy.Symbol("beta"))(q0, q1)),
         _build_op_proto("FSIM", ['theta', 'theta_scalar', 'phi', 'phi_scalar'],
                         ['alpha', 2.1, 'beta', 1.3], [q0_str, q1_str])),
    ]

    return pairs


def _get_controlled_circuit_proto_pairs(qubit_type='grid'):

    q0 = cirq.GridQubit(5, 6)
    q1 = cirq.GridQubit(7, 8)
    q2 = cirq.GridQubit(9, 10)
    q0_str = '5_6'
    q1_str = '7_8'
    q2_str = '9_10'

    if qubit_type == 'line':
        q0 = cirq.LineQubit(6)
        q1 = cirq.LineQubit(7)
        q2 = cirq.LineQubit(10)
        q0_str = '6'
        q1_str = '7'
        q2_str = '10'

    return [(_make_controlled_circuit(a, [q0, q1, q2], [1, 1, 0]),
             _make_controlled_gate_proto(b, ','.join([q0_str, q1_str, q2_str]),
                                         '1,1,0'))
            for a, b in _get_circuit_proto_pairs(qubit_type=qubit_type)]


def _get_valid_pauli_proto_pairs(qubit_type='grid'):
    """Generate valid paulisum proto pairs."""
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(1, 0)
    q0_str = '0_0'
    q1_str = '1_0'

    if qubit_type == 'line':
        q0 = cirq.LineQubit(0)
        q1 = cirq.LineQubit(1)
        q0_str = '0'
        q1_str = '1'

    pairs = [
        (cirq.PauliSum.from_pauli_strings((2.1 + 0.2j) * cirq.Z(q0)),
         _build_pauli_proto([2.1 + 0.2j], [['Z']], [[q0_str]])),
        (cirq.PauliSum.from_pauli_strings((1.0 + 0.0j) * cirq.X(q0)),
         _build_pauli_proto([1.0 + 0.0j], [['X']], [[q0_str]])),
        (cirq.PauliSum.from_pauli_strings((0.0 + 1.0j) * cirq.Y(q0)),
         _build_pauli_proto([0.0 + 1.0j], [['Y']], [[q0_str]])),
        ((0.0 + 1.0j) * cirq.Y(q0) + 1.0 * cirq.Z(q1),
         _build_pauli_proto([0.0 + 1.0j, 1.0 + 0.0j], [['Y'], ['Z']],
                            [[q0_str], [q1_str]])),
        (2.0 * cirq.Y(q1) + 1.0 * cirq.Z(q0) + cirq.X(q0) * cirq.X(q1),
         _build_pauli_proto([2.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j],
                            [['Y'], ['Z'], ['X', 'X']],
                            [[q1_str], [q0_str], [q0_str, q1_str]])),
    ]

    return pairs


def _get_noise_proto_pairs(qubit_type='grid'):
    q0 = cirq.GridQubit(0, 0)
    q0_str = '0_0'

    if qubit_type == 'line':
        q0 = cirq.LineQubit(0)
        q0_str = '0'

    pairs = [
        # Depolarization.
        (cirq.Circuit(cirq.depolarize(p=0.3)(q0)),
         _build_op_proto("DP", ['p'], [0.3], [q0_str])),

        # Asymmetric depolarization.
        (cirq.Circuit(
            cirq.asymmetric_depolarize(p_x=0.1, p_y=0.2, p_z=0.3)(q0)),
         _build_op_proto("ADP", ['p_x', 'p_y', 'p_z'], [0.1, 0.2, 0.3],
                         [q0_str])),

        # Generalized Amplitude damp.
        (cirq.Circuit(cirq.generalized_amplitude_damp(p=0.1, gamma=0.2)(q0)),
         _build_op_proto("GAD", ['p', 'gamma'], [0.1, 0.2], [q0_str])),

        # Amplitude damp.
        (cirq.Circuit(cirq.amplitude_damp(gamma=0.1)(q0)),
         _build_op_proto("AD", ['gamma'], [0.1], [q0_str])),

        # Reset.
        (cirq.Circuit(cirq.reset(q0)), _build_op_proto("RST", [], [],
                                                       [q0_str])),

        # Phase damp.
        (cirq.Circuit(cirq.phase_damp(gamma=0.1)(q0)),
         _build_op_proto("PD", ['gamma'], [0.1], [q0_str])),

        # Phase flip.
        (cirq.Circuit(cirq.phase_flip(p=0.1)(q0)),
         _build_op_proto("PF", ['p'], [0.1], [q0_str])),

        # Bit flip.
        (cirq.Circuit(cirq.bit_flip(p=0.1)(q0)),
         _build_op_proto("BF", ['p'], [0.1], [q0_str]))
    ]
    return pairs


def _build_pauli_proto(coefs, ops, qubit_ids):
    """Construct pauli_sum proto explicitly."""
    terms = []
    for i in range(len(coefs)):
        term = pauli_sum_pb2.PauliTerm()
        term.coefficient_real = coefs[i].real
        term.coefficient_imag = coefs[i].imag
        for j in range(len(qubit_ids[i])):
            term.paulis.add(qubit_id=qubit_ids[i][j], pauli_type=ops[i][j])

        terms.append(term)

    a = pauli_sum_pb2.PauliSum()
    a.terms.extend(terms)
    return a


class SerializerTest(tf.test.TestCase, parameterized.TestCase):
    """Tests basic serializer functionality"""

    @parameterized.parameters(
        [{
            'circ_proto_pair': v
        } for v in _get_controlled_circuit_proto_pairs(qubit_type='grid') +
         _get_controlled_circuit_proto_pairs(qubit_type='line') +
         _get_circuit_proto_pairs(qubit_type='grid') +
         _get_circuit_proto_pairs(qubit_type='line') +
         _get_noise_proto_pairs(qubit_type='grid') +
         _get_noise_proto_pairs(qubit_type='line')])
    def test_serialize_circuit_valid(self, circ_proto_pair):
        """Test conversion of cirq Circuits to tfq_gate_set proto."""
        self.assertProtoEquals(serializer.serialize_circuit(circ_proto_pair[0]),
                               circ_proto_pair[1])

    @parameterized.parameters(
        [{
            'circ_proto_pair': v
        } for v in _get_controlled_circuit_proto_pairs(qubit_type='grid') +
         _get_controlled_circuit_proto_pairs(qubit_type='line') +
         _get_circuit_proto_pairs(qubit_type='grid') +
         _get_circuit_proto_pairs(qubit_type='line') +
         _get_noise_proto_pairs(qubit_type='grid') +
         _get_noise_proto_pairs(qubit_type='line')])
    def test_deserialize_circuit_valid(self, circ_proto_pair):
        """Test deserialization of protos in tfq_gate_set."""

        # String casting is done here to round floating point values.
        # cirq.testing.assert_same_circuits will call  break and think
        # cirq.Z^0.30000001 is different from cirq.Z^0.3
        self.assertEqual(circ_proto_pair[0],
                         serializer.deserialize_circuit(circ_proto_pair[1]))

    @parameterized.parameters(
        [{
            'circ_proto_pair': v
        } for v in _get_controlled_circuit_proto_pairs(qubit_type='grid') +
         _get_controlled_circuit_proto_pairs(qubit_type='line') +
         _get_circuit_proto_pairs(qubit_type='grid') +
         _get_circuit_proto_pairs(qubit_type='line') +
         _get_noise_proto_pairs(qubit_type='grid') +
         _get_noise_proto_pairs(qubit_type='line')])
    def test_serialize_deserialize_circuit_consistency(self, circ_proto_pair):
        """Ensure that serializing followed by deserializing works."""

        # String casting is done here to round floating point values.
        # cirq.testing.assert_same_circuits will call  break and think
        # cirq.Z^0.30000001 is different from cirq.Z^0.3
        self.assertProtoEquals(
            serializer.serialize_circuit(
                serializer.deserialize_circuit(circ_proto_pair[1])),
            circ_proto_pair[1])
        self.assertEqual(
            serializer.deserialize_circuit(
                serializer.serialize_circuit(circ_proto_pair[0])),
            circ_proto_pair[0])

    def test_serialize_circuit_unsupported_gate(self):
        """Ensure we error on unsupported gates."""
        q0 = cirq.GridQubit(0, 0)
        q1 = cirq.GridQubit(0, 1)
        unsupported_circuit = cirq.Circuit(cirq.qft(q0, q1))

        with self.assertRaises(ValueError):
            serializer.serialize_circuit(unsupported_circuit)

    def test_serialize_circuit_with_large_identity(self):
        """Ensure that multi qubit identity errors correctly."""
        q0 = cirq.GridQubit(0, 0)
        q1 = cirq.GridQubit(0, 1)
        unsupported_circuit = cirq.Circuit(
            cirq.IdentityGate(num_qubits=2)(q0, q1))

        with self.assertRaisesRegex(ValueError, expected_regex="cirq.I"):
            serializer.serialize_circuit(unsupported_circuit)

    @parameterized.parameters([
        {
            "gate_with_param": g(p)
        }
        # Use a gate from each category of serializer
        for g in [
            # eigen
            lambda p: cirq.Circuit(
                cirq.HPowGate(exponent=p, global_shift=p)
                (cirq.GridQubit(0, 0))),
            # phased eigen
            lambda p: cirq.Circuit(
                cirq.PhasedXPowGate(
                    phase_exponent=p, exponent=p, global_shift=p)
                (cirq.GridQubit(0, 0))),
            # fsim
            lambda p: cirq.Circuit(
                cirq.FSimGate(theta=p, phi=p)
                (cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))),
        ]
        # Attempt parameterization with a variety of numeric types
        for p in
        [0.35, float(0.35), 35e-2,
         np.float32(0.35),
         np.float64(0.35), 7]
        for (q0, q1) in [(
            cirq.GridQubit(0, 1),
            cirq.GridQubit(0, 0)), (cirq.LineQubit(0), cirq.LineQubit(1))]
    ])
    def test_serialize_circuit_valid_number_types(self, gate_with_param):
        """Tests number datatype support by our serializer."""
        self.assertAllClose(
            gate_with_param.unitary(),
            serializer.deserialize_circuit(
                serializer.serialize_circuit(gate_with_param)).unitary())

    def test_serialize_circuit_unsupported_value(self):
        """Ensure we error on unsupported arithmetic expressions and qubits."""
        q0 = cirq.GridQubit(0, 0)
        unsupported_circuit = cirq.Circuit(
            cirq.HPowGate()(q0)**(sympy.Symbol('alpha') + 1))

        q1 = cirq.NamedQubit('wont work')
        unsupported_circuit2 = cirq.Circuit(cirq.H(q1))

        with self.assertRaises(ValueError):
            serializer.serialize_circuit(unsupported_circuit)

        with self.assertRaises(ValueError):
            serializer.serialize_circuit(unsupported_circuit2)

    def test_serialize_controlled_circuit_unsupported_value(self):
        """Ensure serializing invalid controlled gates fails gracefully."""
        qubits = cirq.GridQubit.rect(1, 2)
        invalid_symbol = cirq.Circuit((cirq.HPowGate()(
            qubits[0])**(sympy.Symbol('alpha') + 1)).controlled_by(qubits[1]))
        with self.assertRaises(ValueError):
            serializer.serialize_circuit(invalid_symbol)

    def test_serialize_noise_channel_unsupported_value(self):
        """Ensure serializing invalid channels fails gracefully."""
        qubit = cirq.NamedQubit('wont work')
        simple_circuit = cirq.Circuit(cirq.depolarize(0.3)(qubit))
        with self.assertRaises(ValueError):
            serializer.serialize_circuit(simple_circuit)

    @parameterized.parameters([{'inp': v} for v in ['wrong', 1.0, None, []]])
    def test_serialize_circuit_wrong_type(self, inp):
        """Attempt to serialize invalid objects types."""
        with self.assertRaises(TypeError):
            serializer.serialize_circuit(input)

    @parameterized.parameters([{'inp': v} for v in ['wrong', 1.0, None, []]])
    def test_deserialize_circuit_wrong_type(self, inp):
        """Attempt to deserialize invalid objects types."""
        with self.assertRaises(TypeError):
            serializer.deserialize_circuit(input)

    @parameterized.parameters([{'inp': v} for v in ['wrong', 1.0, None, []]])
    def test_serialize_paulisum_wrong_type(self, inp):
        """Attempt to serialize invalid object types."""
        with self.assertRaises(TypeError):
            serializer.serialize_paulisum(inp)

    @parameterized.parameters([{'inp': v} for v in ['wrong', 1.0, None, []]])
    def test_deserialize_paulisum_wrong_type(self, inp):
        """Attempt to deserialize invalid object types."""
        with self.assertRaises(TypeError):
            serializer.deserialize_paulisum(inp)

    def test_serialize_paulisum_invalid(self):
        """Ensure we don't support anything but GridQubits."""
        q0 = cirq.NamedQubit('wont work')
        a = 3.0 * cirq.Z(q0) - 2.0 * cirq.X(q0)
        with self.assertRaises(ValueError):
            serializer.serialize_paulisum(a)

    @parameterized.parameters([{
        'sum_proto_pair': v
    } for v in _get_valid_pauli_proto_pairs(qubit_type='grid') +
                               _get_valid_pauli_proto_pairs(qubit_type='line')])
    def test_serialize_paulisum_simple(self, sum_proto_pair):
        """Ensure serialization is correct."""
        self.assertProtoEquals(sum_proto_pair[1],
                               serializer.serialize_paulisum(sum_proto_pair[0]))

    @parameterized.parameters([{
        'sum_proto_pair': v
    } for v in _get_valid_pauli_proto_pairs(qubit_type='grid') +
                               _get_valid_pauli_proto_pairs(qubit_type='line')])
    def test_deserialize_paulisum_simple(self, sum_proto_pair):
        """Ensure deserialization is correct."""
        self.assertEqual(serializer.deserialize_paulisum(sum_proto_pair[1]),
                         sum_proto_pair[0])

    @parameterized.parameters([{
        'sum_proto_pair': v
    } for v in _get_valid_pauli_proto_pairs(qubit_type='grid') +
                               _get_valid_pauli_proto_pairs(qubit_type='line')])
    def test_serialize_deserialize_paulisum_consistency(self, sum_proto_pair):
        """Serialize and deserialize and ensure nothing changed."""
        self.assertEqual(
            serializer.serialize_paulisum(
                serializer.deserialize_paulisum(sum_proto_pair[1])),
            sum_proto_pair[1])

        self.assertEqual(
            serializer.deserialize_paulisum(
                serializer.serialize_paulisum(sum_proto_pair[0])),
            sum_proto_pair[0])

    @parameterized.parameters([
        {
            'gate': cirq.rx(3.0 * sympy.Symbol('alpha'))
        },
        {
            'gate': cirq.ry(-1.0 * sympy.Symbol('alpha'))
        },
        {
            'gate': cirq.rz(sympy.Symbol('alpha'))
        },
    ])
    def test_serialize_deserialize_special_case_one_qubit(self, gate):
        """Check output state equality."""
        q0 = cirq.GridQubit(0, 0)
        c = cirq.Circuit(gate(q0))

        c = cirq.resolve_parameters(c, cirq.ParamResolver({"alpha": 0.1234567}))
        before = c.unitary()
        c2 = serializer.deserialize_circuit(serializer.serialize_circuit(c))
        after = c2.unitary()
        self.assertAllClose(before, after)

    def test_terminal_measurement_support(self):
        """Test that non-terminal measurements error during serialization."""
        q0 = cirq.GridQubit(0, 0)
        q1 = cirq.GridQubit(0, 1)
        simple_circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0), cirq.H(q1),
                                      cirq.Z(q1), cirq.measure(q1))

        simple_circuit_before_call = copy.deepcopy(simple_circuit)

        expected_circuit = cirq.Circuit(cirq.Moment([cirq.H(q0),
                                                     cirq.H(q1)]),
                                        cirq.Moment([cirq.Z(q1)]),
                                        cirq.Moment([]))

        self.assertEqual(serializer.serialize_circuit(simple_circuit),
                         serializer.serialize_circuit(expected_circuit))

        # Check that serialization didn't modify existing circuit.
        self.assertEqual(simple_circuit, simple_circuit_before_call)

        invalid_circuit = cirq.Circuit(cirq.H(q0), cirq.measure(q0),
                                       cirq.measure(q0))

        with self.assertRaisesRegex(ValueError, expected_regex="non-terminal"):
            serializer.serialize_circuit(invalid_circuit)

    def test_serialize_deserialize_identity(self):
        """Confirm that identity gates can be serialized and deserialized."""
        q0 = cirq.GridQubit(0, 0)
        q1 = cirq.GridQubit(0, 1)
        paulisum_with_identity = cirq.PauliSum.from_pauli_strings([
            cirq.PauliString(cirq.I(q0)),
            cirq.PauliString(cirq.Z(q0), cirq.Z(q1)),
        ])
        self.assertEqual(
            paulisum_with_identity,
            serializer.deserialize_paulisum(
                serializer.serialize_paulisum(paulisum_with_identity)))


if __name__ == "__main__":
    tf.test.main()
