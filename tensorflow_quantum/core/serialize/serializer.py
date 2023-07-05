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
from tensorflow_quantum.core.serialize import op_serializer, op_deserializer, \
    serializable_gate_set
from tensorflow_quantum.core.proto import pauli_sum_pb2
from tensorflow_quantum.core.proto import program_pb2
from tensorflow_quantum.core.proto import projector_sum_pb2

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
    if isinstance(expr, sympy.core.Mul):
        lhs_eval, rhs_eval = _parse_mul(expr)

        if isinstance(lhs_eval, sympy.Symbol) and isinstance(
                rhs_eval,
            (sympy.core.numbers.Float, sympy.core.numbers.Integer)):
            # lhs contains symbol rhs contains number.
            return _round(float(rhs_eval))

        if isinstance(rhs_eval, sympy.Symbol) and isinstance(
                lhs_eval,
            (sympy.core.numbers.Float, sympy.core.numbers.Integer)):
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
    if isinstance(expr, sympy.core.Mul):
        lhs_eval, rhs_eval = _parse_mul(expr)

        if isinstance(lhs_eval, sympy.Symbol) and isinstance(
                rhs_eval,
            (sympy.core.numbers.Float, sympy.core.numbers.Integer)):
            # lhs contains symbol rhs contains number.
            return lhs_eval

        if isinstance(rhs_eval, sympy.Symbol) and isinstance(
                lhs_eval,
            (sympy.core.numbers.Float, sympy.core.numbers.Integer)):
            # lhs contains number.
            return rhs_eval

    raise ValueError("Arithmetic expression outside of simple "
                     "scalar multiplication is currently not "
                     "supported. See serializer.py for more "
                     "information.")


def _serialize_controls(gate):
    """Helper to serialize control qubits if applicable."""
    if hasattr(gate, '_tfq_control_qubits'):
        return ','.join(
            op_serializer.qubit_to_proto(q) for q in gate._tfq_control_qubits)
    return ''


def _serialize_control_vals(gate):
    """Helper to serialize control values if applicable.."""
    if hasattr(gate, '_tfq_control_values'):
        return ','.join(str(v[0]) for v in gate._tfq_control_values)
    return ''


class DelayedAssignmentGate(cirq.Gate):
    """Class to do control qubit assignment before sub_gate qubit assignment."""

    def __init__(self, gate_callable, control_qubits, control_values):
        self._gate_callable = gate_callable
        self._control_qubits = control_qubits
        self._control_values = control_values

    def _qid_shape_(self):
        raise ValueError("Called qid_shape on workaround class.")

    # pylint: disable=invalid-name
    def on(self, *qubits):
        """Returns gate_callable on qubits controlled by contol_qubits."""
        gate = self._gate_callable(*qubits)
        # TODO(tonybruguier,#636): Here we call the parent's class controlled_by
        # because Cirq's breaking change #4167 created 3-qubit gates that cannot
        # be serialized yet. Instead, support 3-qubit gates and revert the
        # work-around.
        if len(self._control_qubits) == 0:
            return gate
        return cirq.ControlledOperation(self._control_qubits,
                                        gate,
                                        control_values=self._control_values)

    # pylint: enable=invalid-name


def _optional_control_promote(gate, qubits_message, values_message):
    """Optionally promote to controlled gate based on serialized control msg."""
    if qubits_message == '' and values_message == '':
        return gate
    qbs = [
        op_deserializer.qubit_from_proto(qb) for qb in qubits_message.split(',')
    ]
    vals = [int(cv) for cv in values_message.split(',')]

    return DelayedAssignmentGate(gate, qbs, vals)


# Channels.
def _asymmetric_depolarize_serializer():
    """Make standard serializer for asymmetric depolarization channel."""
    args = [
        # cirq channels can't contain symbols.
        op_serializer.SerializingArg(serialized_name="p_x",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.p_x),
        op_serializer.SerializingArg(serialized_name="p_y",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.p_y),
        op_serializer.SerializingArg(serialized_name="p_z",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.p_z),
        op_serializer.SerializingArg(serialized_name="control_qubits",
                                     serialized_type=str,
                                     op_getter=lambda x: ''),
        op_serializer.SerializingArg(serialized_name="control_values",
                                     serialized_type=str,
                                     op_getter=lambda x: '')
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.AsymmetricDepolarizingChannel,
        serialized_gate_id="ADP",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _asymmetric_depolarize_deserializer():
    """Make standard deserializer for asymmetric depolarization channel."""
    args = [
        op_deserializer.DeserializingArg(serialized_name="p_x",
                                         constructor_arg_name="p_x"),
        op_deserializer.DeserializingArg(serialized_name="p_y",
                                         constructor_arg_name="p_y"),
        op_deserializer.DeserializingArg(serialized_name="p_z",
                                         constructor_arg_name="p_z")
    ]
    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="ADP",
        gate_constructor=cirq.AsymmetricDepolarizingChannel,
        args=args)


def _depolarize_channel_serializer():
    """Make standard serializer for depolarization channel."""

    args = [
        # cirq channels can't contain symbols.
        op_serializer.SerializingArg(serialized_name="p",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.p),
        op_serializer.SerializingArg(serialized_name="control_qubits",
                                     serialized_type=str,
                                     op_getter=lambda x: ''),
        op_serializer.SerializingArg(serialized_name="control_values",
                                     serialized_type=str,
                                     op_getter=lambda x: '')
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.DepolarizingChannel,
        serialized_gate_id="DP",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _depolarize_channel_deserializer():
    """Make standard deserializer for depolarization channel."""

    args = [
        op_deserializer.DeserializingArg(serialized_name="p",
                                         constructor_arg_name="p")
    ]
    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="DP",
        gate_constructor=cirq.DepolarizingChannel,
        args=args)


def _gad_channel_serializer():
    """Make standard serializer for GeneralizedAmplitudeDamping."""

    args = [
        # cirq channels can't contain symbols.
        op_serializer.SerializingArg(serialized_name="p",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.p),
        op_serializer.SerializingArg(serialized_name="gamma",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.gamma),
        op_serializer.SerializingArg(serialized_name="control_qubits",
                                     serialized_type=str,
                                     op_getter=lambda x: ''),
        op_serializer.SerializingArg(serialized_name="control_values",
                                     serialized_type=str,
                                     op_getter=lambda x: '')
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.GeneralizedAmplitudeDampingChannel,
        serialized_gate_id="GAD",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _gad_channel_deserializer():
    """Make standard deserializer for GeneralizedAmplitudeDamping."""

    args = [
        op_deserializer.DeserializingArg(serialized_name="p",
                                         constructor_arg_name="p"),
        op_deserializer.DeserializingArg(serialized_name="gamma",
                                         constructor_arg_name="gamma")
    ]
    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="GAD",
        gate_constructor=cirq.GeneralizedAmplitudeDampingChannel,
        args=args)


def _amplitude_damp_channel_serializer():
    """Make standard serializer for AmplitudeDamp channel."""

    args = [
        # cirq channels can't contain symbols.
        op_serializer.SerializingArg(serialized_name="gamma",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.gamma),
        op_serializer.SerializingArg(serialized_name="control_qubits",
                                     serialized_type=str,
                                     op_getter=lambda x: ''),
        op_serializer.SerializingArg(serialized_name="control_values",
                                     serialized_type=str,
                                     op_getter=lambda x: '')
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.AmplitudeDampingChannel,
        serialized_gate_id="AD",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _amplitude_damp_channel_deserializer():
    """Make standard deserializer for depolarization channel."""

    args = [
        op_deserializer.DeserializingArg(serialized_name="gamma",
                                         constructor_arg_name="gamma")
    ]
    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="AD",
        gate_constructor=cirq.AmplitudeDampingChannel,
        args=args)


def _reset_channel_serializer():
    """Make standard serializer for reset channel."""

    args = [
        # cirq channels can't contain symbols.
        op_serializer.SerializingArg(serialized_name="control_qubits",
                                     serialized_type=str,
                                     op_getter=lambda x: ''),
        op_serializer.SerializingArg(serialized_name="control_values",
                                     serialized_type=str,
                                     op_getter=lambda x: '')
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.ResetChannel,
        serialized_gate_id="RST",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _reset_channel_deserializer():
    """Make standard deserializer for reset channel."""

    args = []
    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="RST", gate_constructor=cirq.ResetChannel, args=args)


def _phase_damp_channel_serializer():
    """Make standard serializer for PhaseDamp channel."""
    args = [
        # cirq channels can't contain symbols.
        op_serializer.SerializingArg(serialized_name="gamma",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.gamma),
        op_serializer.SerializingArg(serialized_name="control_qubits",
                                     serialized_type=str,
                                     op_getter=lambda x: ''),
        op_serializer.SerializingArg(serialized_name="control_values",
                                     serialized_type=str,
                                     op_getter=lambda x: '')
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.PhaseDampingChannel,
        serialized_gate_id="PD",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _phase_damp_channel_deserializer():
    """Make standard deserializer for PhaseDamp channel."""
    args = [
        op_deserializer.DeserializingArg(serialized_name="gamma",
                                         constructor_arg_name="gamma")
    ]
    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="PD",
        gate_constructor=cirq.PhaseDampingChannel,
        args=args)


def _phase_flip_channel_serializer():
    """Make standard serializer for PhaseFlip channel."""
    args = [
        # cirq channels can't contain symbols.
        op_serializer.SerializingArg(serialized_name="p",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.p),
        op_serializer.SerializingArg(serialized_name="control_qubits",
                                     serialized_type=str,
                                     op_getter=lambda x: ''),
        op_serializer.SerializingArg(serialized_name="control_values",
                                     serialized_type=str,
                                     op_getter=lambda x: '')
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.PhaseFlipChannel,
        serialized_gate_id="PF",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _phase_flip_channel_deserializer():
    """Make standard deserializer for PhaseFlip channel."""

    args = [
        op_deserializer.DeserializingArg(serialized_name="p",
                                         constructor_arg_name="p")
    ]
    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="PF",
        gate_constructor=cirq.PhaseFlipChannel,
        args=args)


def _bit_flip_channel_serializer():
    """Make standard serializer for BitFlip channel."""
    args = [
        # cirq channels can't contain symbols.
        op_serializer.SerializingArg(serialized_name="p",
                                     serialized_type=float,
                                     op_getter=lambda x: x.gate.p),
        op_serializer.SerializingArg(serialized_name="control_qubits",
                                     serialized_type=str,
                                     op_getter=lambda x: ''),
        op_serializer.SerializingArg(serialized_name="control_values",
                                     serialized_type=str,
                                     op_getter=lambda x: '')
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.BitFlipChannel,
        serialized_gate_id="BF",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _bit_flip_channel_deserializer():
    """Make standard deserializer for BitFlip channel."""
    args = [
        op_deserializer.DeserializingArg(serialized_name="p",
                                         constructor_arg_name="p")
    ]
    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="BF",
        gate_constructor=cirq.BitFlipChannel,
        args=args)


# Gates.
def _eigen_gate_serializer(gate_type, serialized_id):
    """Make standard serializer for eigen gates."""

    args = [
        op_serializer.SerializingArg(
            serialized_name="exponent",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.exponent)),
        op_serializer.SerializingArg(
            serialized_name="exponent_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.exponent)),
        op_serializer.SerializingArg(
            serialized_name="global_shift",
            serialized_type=float,
            op_getter=lambda x: float(x.gate._global_shift)),
        op_serializer.SerializingArg(
            serialized_name="control_qubits",
            serialized_type=str,
            op_getter=lambda x: _serialize_controls(x)),
        op_serializer.SerializingArg(
            serialized_name="control_values",
            serialized_type=str,
            op_getter=lambda x: _serialize_control_vals(x))
    ]
    return op_serializer.GateOpSerializer(
        gate_type=gate_type,
        serialized_gate_id=serialized_id,
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _eigen_gate_deserializer(gate_type, serialized_id):
    """Make standard deserializer for eigen gates."""

    def _scalar_combiner(exponent, global_shift, exponent_scalar,
                         control_qubits, control_values):
        """This is a workaround to support symbol scalar multiplication.
        In the future we should likely get rid of this in favor of proper
        expression parsing once cirq supports it. See cirq.op_serializer
        and cirq's program protobuf for details. This is needed for things
        like cirq.rx('alpha').
        """
        if exponent_scalar == 1.0:
            return _optional_control_promote(
                gate_type(exponent=_round(exponent),
                          global_shift=_round(global_shift)), control_qubits,
                control_values)
        return _optional_control_promote(
            gate_type(exponent=_round(exponent) * _round(exponent_scalar),
                      global_shift=_round(global_shift)), control_qubits,
            control_values)

    args = [
        op_deserializer.DeserializingArg(serialized_name="exponent",
                                         constructor_arg_name="exponent"),
        op_deserializer.DeserializingArg(serialized_name="global_shift",
                                         constructor_arg_name="global_shift"),
        op_deserializer.DeserializingArg(
            serialized_name="exponent_scalar",
            constructor_arg_name="exponent_scalar"),
        op_deserializer.DeserializingArg(serialized_name="control_qubits",
                                         constructor_arg_name="control_qubits"),
        op_deserializer.DeserializingArg(serialized_name="control_values",
                                         constructor_arg_name="control_values")
    ]
    return op_deserializer.GateOpDeserializer(serialized_gate_id=serialized_id,
                                              gate_constructor=_scalar_combiner,
                                              args=args)


def _fsim_gate_serializer():
    """Make standard serializer for fsim gate."""

    args = [
        op_serializer.SerializingArg(
            serialized_name="theta",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.theta)),
        op_serializer.SerializingArg(
            serialized_name="phi",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.phi)),
        op_serializer.SerializingArg(
            serialized_name="theta_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.theta)),
        op_serializer.SerializingArg(
            serialized_name="phi_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.phi)),
        op_serializer.SerializingArg(
            serialized_name="control_qubits",
            serialized_type=str,
            op_getter=lambda x: _serialize_controls(x)),
        op_serializer.SerializingArg(
            serialized_name="control_values",
            serialized_type=str,
            op_getter=lambda x: _serialize_control_vals(x))
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.FSimGate,
        serialized_gate_id="FSIM",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _fsim_gate_deserializer():
    """Make standard deserializer for fsim gate."""

    def _scalar_combiner(theta, theta_scalar, phi, phi_scalar, control_qubits,
                         control_values):
        """This is a workaround to support symbol scalar multiplication.
        See `_eigen_gate_deserializer` for details.
        """
        return _optional_control_promote(
            cirq.FSimGate(theta=_round(theta) * _round(theta_scalar),
                          phi=_round(phi) * _round(phi_scalar)), control_qubits,
            control_values)

    args = [
        op_deserializer.DeserializingArg(serialized_name="theta",
                                         constructor_arg_name="theta"),
        op_deserializer.DeserializingArg(serialized_name="phi",
                                         constructor_arg_name="phi"),
        op_deserializer.DeserializingArg(serialized_name="theta_scalar",
                                         constructor_arg_name="theta_scalar"),
        op_deserializer.DeserializingArg(serialized_name="phi_scalar",
                                         constructor_arg_name="phi_scalar"),
        op_deserializer.DeserializingArg(serialized_name="control_qubits",
                                         constructor_arg_name="control_qubits"),
        op_deserializer.DeserializingArg(serialized_name="control_values",
                                         constructor_arg_name="control_values")
    ]
    return op_deserializer.GateOpDeserializer(serialized_gate_id="FSIM",
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
        op_serializer.SerializingArg(serialized_name="unused",
                                     serialized_type=bool,
                                     op_getter=_identity_check),
        op_serializer.SerializingArg(
            serialized_name="control_qubits",
            serialized_type=str,
            op_getter=lambda x: _serialize_controls(x)),
        op_serializer.SerializingArg(
            serialized_name="control_values",
            serialized_type=str,
            op_getter=lambda x: _serialize_control_vals(x))
    ]
    return op_serializer.GateOpSerializer(
        gate_type=cirq.IdentityGate,
        serialized_gate_id="I",
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _identity_gate_deserializer():
    """Make a standard deserializer for the single qubit identity."""
    args = [
        op_deserializer.DeserializingArg(serialized_name="unused",
                                         constructor_arg_name="unused"),
        op_deserializer.DeserializingArg(serialized_name="control_qubits",
                                         constructor_arg_name="control_qubits"),
        op_deserializer.DeserializingArg(serialized_name="control_values",
                                         constructor_arg_name="control_values")
    ]

    def _cirq_i_workaround(unused, control_qubits, control_values):
        return _optional_control_promote(cirq.I, control_qubits, control_values)

    return op_deserializer.GateOpDeserializer(
        serialized_gate_id="I", gate_constructor=_cirq_i_workaround, args=args)


def _phased_eigen_gate_serializer(gate_type, serialized_id):
    """Make a standard serializer for phased eigen gates."""

    args = [
        op_serializer.SerializingArg(
            serialized_name="phase_exponent",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.phase_exponent)),
        op_serializer.SerializingArg(
            serialized_name="phase_exponent_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.phase_exponent)),
        op_serializer.SerializingArg(
            serialized_name="exponent",
            serialized_type=float,
            op_getter=lambda x: _symbol_extractor(x.gate.exponent)),
        op_serializer.SerializingArg(
            serialized_name="exponent_scalar",
            serialized_type=float,
            op_getter=lambda x: _scalar_extractor(x.gate.exponent)),
        op_serializer.SerializingArg(
            serialized_name="global_shift",
            serialized_type=float,
            op_getter=lambda x: float(x.gate.global_shift)),
        op_serializer.SerializingArg(
            serialized_name="control_qubits",
            serialized_type=str,
            op_getter=lambda x: _serialize_controls(x)),
        op_serializer.SerializingArg(
            serialized_name="control_values",
            serialized_type=str,
            op_getter=lambda x: _serialize_control_vals(x))
    ]
    return op_serializer.GateOpSerializer(
        gate_type=gate_type,
        serialized_gate_id=serialized_id,
        args=args,
        can_serialize_predicate=_CONSTANT_TRUE)


def _phased_eigen_gate_deserializer(gate_type, serialized_id):
    """Make a standard deserializer for phased eigen gates."""

    def _scalar_combiner(exponent, global_shift, exponent_scalar,
                         phase_exponent, phase_exponent_scalar, control_qubits,
                         control_values):
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
            return _optional_control_promote(
                gate_type(exponent=exponent,
                          global_shift=_round(global_shift),
                          phase_exponent=phase_exponent), control_qubits,
                control_values)
        return _optional_control_promote(
            gate_type(exponent=exponent, phase_exponent=phase_exponent),
            control_qubits, control_values)

    args = [
        op_deserializer.DeserializingArg(serialized_name="phase_exponent",
                                         constructor_arg_name="phase_exponent"),
        op_deserializer.DeserializingArg(
            serialized_name="phase_exponent_scalar",
            constructor_arg_name="phase_exponent_scalar"),
        op_deserializer.DeserializingArg(serialized_name="exponent",
                                         constructor_arg_name="exponent"),
        op_deserializer.DeserializingArg(
            serialized_name="exponent_scalar",
            constructor_arg_name="exponent_scalar"),
        op_deserializer.DeserializingArg(serialized_name="global_shift",
                                         constructor_arg_name="global_shift"),
        op_deserializer.DeserializingArg(serialized_name="control_qubits",
                                         constructor_arg_name="control_qubits"),
        op_deserializer.DeserializingArg(serialized_name="control_values",
                                         constructor_arg_name="control_values")
    ]
    return op_deserializer.GateOpDeserializer(serialized_gate_id=serialized_id,
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
    _phased_eigen_gate_serializer(g, g_name)
    for g, g_name in PHASED_EIGEN_GATES_DICT.items()
] + [
    _amplitude_damp_channel_serializer(),
    _asymmetric_depolarize_serializer(),
    _bit_flip_channel_serializer(),
    _depolarize_channel_serializer(),
    _fsim_gate_serializer(),
    _gad_channel_serializer(),
    _identity_gate_serializer(),
    _phase_damp_channel_serializer(),
    _reset_channel_serializer(),
    _phase_flip_channel_serializer()
]

DESERIALIZERS = [
    _eigen_gate_deserializer(g, g_name)
    for g, g_name in EIGEN_GATES_DICT.items()
] + [
    _phased_eigen_gate_deserializer(g, g_name)
    for g, g_name in PHASED_EIGEN_GATES_DICT.items()
] + [
    _amplitude_damp_channel_deserializer(),
    _asymmetric_depolarize_deserializer(),
    _bit_flip_channel_deserializer(),
    _depolarize_channel_deserializer(),
    _fsim_gate_deserializer(),
    _gad_channel_deserializer(),
    _identity_gate_deserializer(),
    _phase_damp_channel_deserializer(),
    _reset_channel_deserializer(),
    _phase_flip_channel_deserializer()
]

SERIALIZER = serializable_gate_set.SerializableGateSet(
    gate_set_name="tfq_gate_set",
    serializers=SERIALIZERS,
    deserializers=DESERIALIZERS)


def serialize_circuit(circuit_inp):
    """Returns a `cirq.Program` proto representing the `cirq.Circuit`.

    Note that the circuit must use gates valid in the tfq_gate_set.
    Currently we only support scalar multiplication of symbols and
    no other more complex arithmetic expressions. This means
    we can support things like X**(3*alpha), and Rx(alpha). Because
    we use the `cirq.Program` proto, we only support `cirq.GridQubit`
    and `cirq.LineQubit` instances during serialization of circuits.

    Note: once serialized terminal measurements are removed.

    Args:
        circuit_inp: A `cirq.Circuit`.

    Returns:
        A `tfq.proto.Program` proto.
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
                if not isinstance(qubit, (cirq.GridQubit, cirq.LineQubit)):
                    raise ValueError(
                        "Attempted to serialize circuit that don't use "
                        "only cirq.GridQubits or cirq.LineQubits.")

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

    # Demote cirq.controlled_operations (controlled gates) to their sub_gate
    # types with _tfq_control_qubits and _tfq_control_values fields so that
    # the gates can still get picked up by the serializer. There would be no way
    # to discern controlledgates from one another otherwise. This
    # "momentary demotion" occurs with the help of the DelayedAssignmentGate.
    for i, moment in enumerate(circuit):
        controlled_ops = [
            op for op in moment if isinstance(op, cirq.ControlledOperation)
        ]
        new_ops = dict()
        for op in controlled_ops:
            tfq_compatible = op.sub_operation
            tfq_compatible._tfq_control_qubits = op.controls
            tfq_compatible._tfq_control_values = op.control_values
            new_ops[op.qubits] = tfq_compatible

        circuit[i] = cirq.Moment(
            new_ops[op.qubits] if op.qubits in new_ops else op for op in moment)

    return SERIALIZER.serialize(circuit)


def deserialize_circuit(proto):
    """Constructs a `cirq.Circuit` from a `cirq.Program` proto.

    Note that the proto must use gates valid in the tfq_gate_set.

    Args:
        proto: A `tfq.proto.Program` proto

    Returns:
        A `cirq.Circuit`.
    """
    if not isinstance(proto, program_pb2.Program):
        raise TypeError("deserialize requires "
                        "tfq.proto.Program object."
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

    if any(not isinstance(qubit, (cirq.LineQubit, cirq.GridQubit))
           for qubit in paulisum.qubits):
        raise ValueError("Attempted to serialize a paulisum that doesn't use "
                         "only cirq.GridQubits or cirq.LineQubits.")

    paulisum_proto = pauli_sum_pb2.PauliSum()
    for term in paulisum:
        pauliterm_proto = pauli_sum_pb2.PauliTerm()

        pauliterm_proto.coefficient_real = term.coefficient.real
        pauliterm_proto.coefficient_imag = term.coefficient.imag
        for t in sorted(term.items()):  # sort to keep qubits ordered.
            pauliterm_proto.paulis.add(
                qubit_id=op_serializer.qubit_to_proto(t[0]),
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
            term *= op(
                op_deserializer.qubit_from_proto(pauli_qubit_pair.qubit_id))
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


def serialize_projectorsum(projectorsum):
    """Constructs a projector_sum proto from `cirq.ProjectorSum`.

    Args:
        projectorsum: A `cirq.ProjectorSum` or `cirq.ProjectorString` object.

    Returns:
        A projector_sum proto object.
    """
    if isinstance(projectorsum, cirq.ProjectorString):
        projectorsum = cirq.ProjectorSum.from_pauli_strings(projectorsum)

    if not isinstance(projectorsum, cirq.ProjectorSum):
        raise TypeError("serialize requires a cirq.ProjectorSum object."
                        " Given: " + str(type(projectorsum)))

    if any(not isinstance(qubit, (cirq.GridQubit, cirq.LineQubit))
           for qubit in projectorsum.qubits):
        raise ValueError("Attempted to serialize a paulisum that doesn't use "
                         "only cirq.GridQubit or cirq.LineQubit.")

    projectorsum_proto = projector_sum_pb2.ProjectorSum()
    for term in projectorsum:
        projectorterm_proto = projector_sum_pb2.ProjectorTerm()

        projectorterm_proto.coefficient_real = term.coefficient.real
        projectorterm_proto.coefficient_imag = term.coefficient.imag
        for qubit, basis_state in sorted(
                term.projector_dict.items()):  # sort to keep qubits ordered
            projectorterm_proto.projector_dict.add(
                qubit_id=op_serializer.qubit_to_proto(qubit),
                basis_state=basis_state)

        projectorsum_proto.terms.extend([projectorterm_proto])

    return projectorsum_proto


def deserialize_projectorsum(proto):
    """Constructs a `cirq.ProjectorSum` from projector_sum proto.

    Args:
        proto: A projector_sum proto object.

    Returns:
        A `cirq.ProjectorSum` object.
    """
    if not isinstance(proto, projector_sum_pb2.ProjectorSum):
        raise TypeError("deserialize requires a projector_sum_pb2 object."
                        " Given: " + str(type(proto)))

    res = cirq.ProjectorSum()
    for term_proto in proto.terms:
        coef = float(_round(term_proto.coefficient_real)) + \
            1.0j * float(_round(term_proto.coefficient_imag))
        projector_dict = {}
        for projector_dict_entry in term_proto.projector_dict:
            qubit = op_deserializer.qubit_from_proto(
                projector_dict_entry.qubit_id)
            projector_dict[qubit] = 1 if projector_dict_entry.basis_state else 0
        res += cirq.ProjectorString(projector_dict, coef)

    return res
