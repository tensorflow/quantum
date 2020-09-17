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
"""A basic serializer used to serialize/deserialize Cirq circuits for tfq."""
# TODO(pmassey / anyone): determine if this should be kept as globals.
import copy
import numbers
import sympy
import numpy as np

import cirq
import cirq.google.api.v2 as v2
from tensorflow_quantum.core.proto import pauli_sum_pb2

# Needed to allow autograph to crawl AST without erroring.
_CONSTANT_TRUE = lambda x: True


def _round(x):
    return np.round(x, 6) if isinstance(x, float) else x


def _parse_mul(expr):
    """Returns the lhs and rhs of a sympy.Mul. This is written
    to prevent autograph from going into sympy library code and having
    conflicts with the @cacheit decorator."""
    if len(expr.args) == 1:
        return sympy.S.One, expr.args[0]
    if len(expr.args) == 2:
        return expr.args[0], expr.args[1]

    raise ValueError("Arithmetic expression outside of simple "
                     "scalar multiplication is currently not "
                     "supported. See serializer.py for more "
                     "information.")


def _scalar_extractor(x):
    """This is a workaround to support symbol scalar multiplication.
    In the future we should likely get rid of this in favor of proper
    expression parsing once cirq supports it. See cirq.op_serializer
    and cirq's program protobuf for details. This is needed for things
    like cirq.rx('alpha').
    """
    if not isinstance(x, (numbers.Real, sympy.Expr)):
        raise TypeError("Invalid input argument for exponent.")

    if isinstance(x, (numbers.Real, sympy.Symbol)):
        return 1.0

    expr = x.evalf()
    if isinstance(expr, sympy.mul.Mul):
        lhs_eval, rhs_eval = _parse_mul(expr)

        if isinstance(lhs_eval, sympy.Symbol) and isinstance(
                rhs_eval, (sympy.numbers.Float, sympy.numbers.Integer)):
            # lhs contains symbol rhs contains number.
            return _round(float(rhs_eval))

        if isinstance(rhs_eval, sympy.Symbol) and isinstance(
                lhs_eval, (sympy.numbers.Float, sympy.numbers.Integer)):
            # lhs contains number.
            return _round(float(lhs_eval))

    raise ValueError("Arithmetic expression outside of simple "
                     "scalar multiplication is currently not "
                     "supported. See serializer.py for more "
                     "information.")


def _symbol_extractor(x):
    """This is the second extractor for above."""
    if not isinstance(x, (numbers.Real, sympy.Expr)):
        raise TypeError("Invalid input argument for exponent.")

    if isinstance(x, numbers.Real):
        return _round(float(x))
    if isinstance(x, sympy.Symbol):
        return x

    expr = x.evalf()
    if isinstance(expr, sympy.mul.Mul):
        lhs_eval, rhs_eval = _parse_mul(expr)

        if isinstance(lhs_eval, sympy.Symbol) and isinstance(
                rhs_eval, (sympy.numbers.Float, sympy.numbers.Integer)):
            # lhs contains symbol rhs contains number.
            return lhs_eval

        if isinstance(rhs_eval, sympy.Symbol) and isinstance(
                lhs_eval, (sympy.numbers.Float, sympy.numbers.Integer)):
            # lhs contains number.
            return rhs_eval

    raise ValueError("Arithmetic expression outside of simple "
                     "scalar multiplication is currently not "
                     "supported. See serializer.py for more "
                     "information.")


def _eigen_gate_serializer(gate_type, serialized_id):
    """Make standard serializer for eigen gates."""

    args = [
        cirq.google.SerializingArg(
            serialized_name="exponent",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.exponent)),
        cirq.google.SerializingArg(
            serialized_name="exponent_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.exponent)),
        cirq.google.SerializingArg(
            serialized_name="global_shift",
            serialized_type=float,
            op_getter=lambda x: float(x.gate._global_shift))
    ]
    return cirq.google.GateOpSerializer(gate_type=gate_type,
                                        serialized_gate_id=serialized_id,
                                        args=args,
                                        can_serialize_predicate=_CONSTANT_TRUE)


def _eigen_gate_deserializer(gate_type, serialized_id):
    """Make standard deserializer for eigen gates."""

    def _scalar_combiner(exponent, global_shift, exponent_scalar):
        """This is a workaround to support symbol scalar multiplication.
        In the future we should likely get rid of this in favor of proper
        expression parsing once cirq supports it. See cirq.op_serializer
        and cirq's program protobuf for details. This is needed for things
        like cirq.rx('alpha').
        """
        if exponent_scalar == 1.0:
            return gate_type(exponent=_round(exponent),
                             global_shift=_round(global_shift))
        return gate_type(exponent=_round(exponent) * _round(exponent_scalar),
                         global_shift=_round(global_shift))

    args = [
        cirq.google.DeserializingArg(serialized_name="exponent",
                                     constructor_arg_name="exponent"),
        cirq.google.DeserializingArg(serialized_name="global_shift",
                                     constructor_arg_name="global_shift"),
        cirq.google.DeserializingArg(serialized_name="exponent_scalar",
                                     constructor_arg_name="exponent_scalar")
    ]
    return cirq.google.GateOpDeserializer(serialized_gate_id=serialized_id,
                                          gate_constructor=_scalar_combiner,
                                          args=args)


def _fsim_gate_serializer():
    """Make standard serializer for fsim gate."""

    args = [
        cirq.google.SerializingArg(
            serialized_name="theta",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.theta)),
        cirq.google.SerializingArg(
            serialized_name="phi",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.phi)),
        cirq.google.SerializingArg(
            serialized_name="theta_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.theta)),
        cirq.google.SerializingArg(
            serialized_name="phi_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.phi)),
    ]
    return cirq.google.GateOpSerializer(gate_type=cirq.FSimGate,
                                        serialized_gate_id="FSIM",
                                        args=args,
                                        can_serialize_predicate=_CONSTANT_TRUE)


def _fsim_gate_deserializer():
    """Make standard deserializer for fsim gate."""

    def _scalar_combiner(theta, theta_scalar, phi, phi_scalar):
        """This is a workaround to support symbol scalar multiplication.
        See `_eigen_gate_deserializer` for details.
        """
        return cirq.FSimGate(theta=_round(theta) * _round(theta_scalar),
                             phi=_round(phi) * _round(phi_scalar))

    args = [
        cirq.google.DeserializingArg(serialized_name="theta",
                                     constructor_arg_name="theta"),
        cirq.google.DeserializingArg(serialized_name="phi",
                                     constructor_arg_name="phi"),
        cirq.google.DeserializingArg(serialized_name="theta_scalar",
                                     constructor_arg_name="theta_scalar"),
        cirq.google.DeserializingArg(serialized_name="phi_scalar",
                                     constructor_arg_name="phi_scalar"),
    ]
    return cirq.google.GateOpDeserializer(serialized_gate_id="FSIM",
                                          gate_constructor=_scalar_combiner,
                                          args=args)


def _identity_gate_serializer():
    """Make a standard serializer for the single qubit identity."""

    def _identity_check(x):
        if x.gate.num_qubits() != 1:
            raise ValueError("Multi-Qubit identity gate not supported."
                             "Given: {}. To work around this, use "
                             "cirq.I.on_each instead.".format(str(x)))
        return True

    # Here `args` is used for two reasons. 1. GateOpSerializer doesn't work well
    # with empty arg lists. 2. It is a nice way to check identity gate size.
    args = [
        cirq.google.SerializingArg(serialized_name="unused",
                                   serialized_type=bool,
                                   op_getter=_identity_check)
    ]
    return cirq.google.GateOpSerializer(gate_type=cirq.IdentityGate,
                                        serialized_gate_id="I",
                                        args=args,
                                        can_serialize_predicate=_CONSTANT_TRUE)


def _identity_gate_deserializer():
    """Make a standard deserializer for the single qubit identity."""
    args = [
        cirq.google.DeserializingArg(serialized_name="unused",
                                     constructor_arg_name="unused")
    ]

    def _cirq_i_workaround(unused):
        return cirq.I

    return cirq.google.GateOpDeserializer(serialized_gate_id="I",
                                          gate_constructor=_cirq_i_workaround,
                                          args=args)


def _phased_eigen_gate_serializer(gate_type, serialized_id):
    """Make a standard serializer for phased eigen gates."""

    args = [
        cirq.google.SerializingArg(
            serialized_name="phase_exponent",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.phase_exponent)),
        cirq.google.SerializingArg(
            serialized_name="phase_exponent_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.phase_exponent)),
        cirq.google.SerializingArg(
            serialized_name="exponent",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.exponent)),
        cirq.google.SerializingArg(
            serialized_name="exponent_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.exponent)),
        cirq.google.SerializingArg(
            serialized_name="global_shift",
            serialized_type=float,
            op_getter=lambda x: float(x.gate.global_shift))
    ]
    return cirq.google.GateOpSerializer(gate_type=gate_type,
                                        serialized_gate_id=serialized_id,
                                        args=args,
                                        can_serialize_predicate=_CONSTANT_TRUE)


def _phased_eigen_gate_deserializer(gate_type, serialized_id):
    """Make a standard deserializer for phased eigen gates."""

    def _scalar_combiner(exponent, global_shift, exponent_scalar,
                         phase_exponent, phase_exponent_scalar):
        """This is a workaround to support symbol scalar multiplication.
        In the future we should likely get rid of this in favor of proper
        expression parsing once cirq supports it. See cirq.op_serializer
        and cirq's program protobuf for details. This is needed for things
        like cirq.rx('alpha').
        """
        exponent = _round(exponent)
        phase_exponent = _round(phase_exponent)
        exponent = exponent if exponent_scalar == 1.0 \
            else exponent * _round(exponent_scalar)
        phase_exponent = phase_exponent if phase_exponent_scalar == 1.0 \
            else phase_exponent * _round(phase_exponent_scalar)
        if global_shift != 0:
            # needed in case this specific phasedeigengate doesn't
            # have a global_phase in constructor.
            return gate_type(exponent=exponent,
                             global_shift=_round(global_shift),
                             phase_exponent=phase_exponent)
        return gate_type(exponent=exponent, phase_exponent=phase_exponent)

    args = [
        cirq.google.DeserializingArg(serialized_name="phase_exponent",
                                     constructor_arg_name="phase_exponent"),
        cirq.google.DeserializingArg(
            serialized_name="phase_exponent_scalar",
            constructor_arg_name="phase_exponent_scalar"),
        cirq.google.DeserializingArg(serialized_name="exponent",
                                     constructor_arg_name="exponent"),
        cirq.google.DeserializingArg(serialized_name="exponent_scalar",
                                     constructor_arg_name="exponent_scalar"),
        cirq.google.DeserializingArg(serialized_name="global_shift",
                                     constructor_arg_name="global_shift"),
    ]
    return cirq.google.GateOpDeserializer(serialized_gate_id=serialized_id,
                                          gate_constructor=_scalar_combiner,
                                          args=args)


EIGEN_GATES_DICT = {
    cirq.XPowGate: "XP",
    cirq.XXPowGate: "XXP",
    cirq.YPowGate: "YP",
    cirq.YYPowGate: "YYP",
    cirq.ZPowGate: "ZP",
    cirq.ZZPowGate: "ZZP",
    cirq.HPowGate: "HP",
    cirq.CZPowGate: "CZP",
    cirq.CNotPowGate: "CNP",
    cirq.SwapPowGate: "SP",
    cirq.ISwapPowGate: "ISP",
}

PHASED_EIGEN_GATES_DICT = {
    cirq.PhasedXPowGate: "PXP",
    cirq.PhasedISwapPowGate: "PISP",
}

SERIALIZERS = [
    _eigen_gate_serializer(g, g_name) for g, g_name in EIGEN_GATES_DICT.items()
] + [
    _fsim_gate_serializer(),
] + [
    _identity_gate_serializer(),
] + [
    _phased_eigen_gate_serializer(g, g_name)
    for g, g_name in PHASED_EIGEN_GATES_DICT.items()
]

DESERIALIZERS = [
    _eigen_gate_deserializer(g, g_name)
    for g, g_name in EIGEN_GATES_DICT.items()
] + [
    _fsim_gate_deserializer(),
] + [
    _identity_gate_deserializer(),
] + [
    _phased_eigen_gate_deserializer(g, g_name)
    for g, g_name in PHASED_EIGEN_GATES_DICT.items()
]

SERIALIZER = cirq.google.SerializableGateSet(gate_set_name="tfq_gate_set",
                                             serializers=SERIALIZERS,
                                             deserializers=DESERIALIZERS)


def serialize_circuit(circuit_inp):
    """Returns a `cirq.Program` proto representing the `cirq.Circuit`.

    Note that the circuit must use gates valid in the tfq_gate_set.
    Currently we only support scalar multiplication of symbols and
    no other more complex arithmetic expressions. This means
    we can support things like X**(3*alpha), and Rx(alpha). Because
    we use the `cirq.Program` proto, we only support `cirq.GridQubit` instances
    during serialization of circuits.

    Note: once serialized terminal measurements are removed.

    Args:
        circuit_inp: A `cirq.Circuit`.

    Returns:
        A `cirq.google.api.v2.Program` proto.
    """
    circuit = copy.deepcopy(circuit_inp)
    if not isinstance(circuit, cirq.Circuit):
        raise TypeError("serialize requires cirq.Circuit objects."
                        " Given: " + str(type(circuit)))

    # This code is intentionally written to avoid using cirq functions
    # as this get analyzed by tensorflow-autograph.

    # Gives a map from moment index to measure qubits in moment
    measured_moments = dict()

    # Tracks qubits that have been measured already.
    all_measured_qubits = set()
    for i, moment in enumerate(circuit.moments):
        measured_qubits = set()
        for op in moment:
            for qubit in op.qubits:
                if not isinstance(qubit, cirq.GridQubit):
                    raise ValueError(
                        "Attempted to serialize circuit that don't use "
                        "only cirq.GridQubits.")

            if isinstance(op.gate, cirq.MeasurementGate):
                for qubit in op.qubits:
                    if qubit in all_measured_qubits:
                        raise ValueError("Serialization of circuit failed. "
                                         "Circuits with non-terminal "
                                         "measurement operations are not "
                                         "supported.")
                    measured_qubits.add(qubit)
                    all_measured_qubits.add(qubit)

        if len(measured_qubits) > 0:
            measured_moments[i] = measured_qubits

    # Remove terminal measurements.
    for moment_ind in measured_moments:
        old_moment = circuit[moment_ind]
        measured_qubits = measured_moments[moment_ind]
        new_moment = cirq.Moment(
            filter(lambda x: not any(y in measured_qubits for y in x.qubits),
                   old_moment.operations))
        circuit[moment_ind] = new_moment

    return SERIALIZER.serialize(circuit)


def deserialize_circuit(proto):
    """Constructs a `cirq.Circuit` from a `cirq.Program` proto.

    Note that the proto must use gates valid in the tfq_gate_set.

    Args:
        proto: A `cirq.google.api.v2.Program` proto

    Returns:
        A `cirq.Circuit`.
    """
    if not isinstance(proto, cirq.google.api.v2.program_pb2.Program):
        raise TypeError("deserialize requires "
                        "cirq.google.api.v2.program_pb2.Program object."
                        " Given: " + str(type(proto)))

    return SERIALIZER.deserialize(proto)


def serialize_paulisum(paulisum):
    """Constructs a pauli_sum proto from `cirq.PauliSum` or `cirq.PauliString`.

    Args:
        paulisum: A `cirq.PauliSum` object.

    Returns:
        A pauli_sum proto object.
    """
    if isinstance(paulisum, cirq.PauliString):
        paulisum = cirq.PauliSum.from_pauli_strings(paulisum)

    if not isinstance(paulisum, cirq.PauliSum):
        raise TypeError("serialize requires a cirq.PauliSum object."
                        " Given: " + str(type(paulisum)))

    if any(not isinstance(qubit, cirq.GridQubit) for qubit in paulisum.qubits):
        raise ValueError("Attempted to serialize a paulisum that doesn't use "
                         "only cirq.GridQubits.")

    paulisum_proto = pauli_sum_pb2.PauliSum()
    for term in paulisum:
        pauliterm_proto = pauli_sum_pb2.PauliTerm()

        pauliterm_proto.coefficient_real = term.coefficient.real
        pauliterm_proto.coefficient_imag = term.coefficient.imag
        for t in sorted(term.items()):  # sort to keep qubits ordered.
            pauliterm_proto.paulis.add(
                qubit_id=v2.qubit_to_proto_id(t[0]),
                pauli_type=str(t[1]),
            )
        paulisum_proto.terms.extend([pauliterm_proto])

    return paulisum_proto


def deserialize_paulisum(proto):
    """Constructs a `cirq.PauliSum` from pauli_sum proto.

    Args:
        proto: A pauli_sum proto object.

    Returns:
        A `cirq.PauliSum` object.
    """
    if not isinstance(proto, pauli_sum_pb2.PauliSum):
        raise TypeError("deserialize requires a pauli_sum_pb2 object."
                        " Given: " + str(type(proto)))

    res = cirq.PauliSum()
    for term_proto in proto.terms:
        coef = float(_round(term_proto.coefficient_real)) + \
            1.0j * float(_round(term_proto.coefficient_imag))
        term = coef * cirq.PauliString()
        for pauli_qubit_pair in term_proto.paulis:
            op = _process_pauli_type(pauli_qubit_pair.pauli_type)
            term *= op(v2.grid_qubit_from_proto_id(pauli_qubit_pair.qubit_id))
        res += term

    return res


def _process_pauli_type(char):
    if char == 'Z':
        return cirq.Z
    if char == 'X':
        return cirq.X
    if char == 'Y':
        return cirq.Y
    raise ValueError("Invalid pauli type.")
