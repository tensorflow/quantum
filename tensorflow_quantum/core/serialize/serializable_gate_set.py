# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Support for serializing and deserializing cirq_google.api.v2 protos."""

import cirq
from tensorflow_quantum.core.proto import program_pb2

LANGUAGE_ORDER = [
    '',
    'linear',
    'exp',
]


def _max_lang(langs):
    i = max((LANGUAGE_ORDER.index(e) for e in langs), default=0)
    return LANGUAGE_ORDER[i]


def _infer_function_language_from_circuit(value):
    return _max_lang({
        e for moment in value.moments for op in moment.operations
        for e in _function_languages_from_operation(op)
    })


def _function_languages_from_operation(value):
    for arg in value.args.values():
        yield from _function_languages_from_arg(arg)


def _function_languages_from_arg(arg_proto):
    which = arg_proto.WhichOneof('arg')
    if which == 'func':
        if arg_proto.func.type in ['add', 'mul']:
            yield 'linear'
            for a in arg_proto.func.args:
                yield from _function_languages_from_arg(a)
        if arg_proto.func.type in ['pow']:
            yield 'exp'
            for a in arg_proto.func.args:
                yield from _function_languages_from_arg(a)


class SerializableGateSet:
    """A class for serializing and deserializing programs and operations.

    This class is for cirq_google.api.v2. protos.
    """

    def __init__(self, gate_set_name, serializers, deserializers):
        """Construct the gate set.

        Args:
            gate_set_name: The name used to identify the gate set.
            serializers: The GateOpSerializers to use for serialization.
                Multiple serializers for a given gate type are allowed and
                will be checked for a given type in the order specified here.
                This allows for a given gate type to be serialized into
                different serialized form depending on the parameters of the
                gate.
            deserializers: The GateOpDeserializers to convert serialized
                forms of gates to GateOperations.
        """
        self.gate_set_name = gate_set_name
        self.serializers = {}
        for s in serializers:
            self.serializers.setdefault(s.gate_type, []).append(s)
        self.deserializers = {d.serialized_gate_id: d for d in deserializers}

    def with_added_gates(
            self,
            *,
            gate_set_name=None,
            serializers=(),
            deserializers=(),
    ):
        """Creates a new gateset with additional (de)serializers.

        Args:
            gate_set_name: Optional new name of the gateset. If not given, use
                the same name as this gateset.
            serializers: Serializers to add to those in this gateset.
            deserializers: Deserializers to add to those in this gateset.
        """
        # Iterate over all serializers in this gateset.
        curr_serializers = (serializer
                            for serializers in self.serializers.values()
                            for serializer in serializers)
        return SerializableGateSet(
            gate_set_name or self.gate_set_name,
            serializers=[*curr_serializers, *serializers],
            deserializers=[*self.deserializers.values(), *deserializers])

    def supported_gate_types(self):
        """Tuple of support gate types."""
        return tuple(self.serializers.keys())

    def is_supported_operation(self, op):
        """Whether or not the given gate can be serialized by this gate set."""
        gate = op.gate
        for gate_type_mro in type(gate).mro():
            if gate_type_mro in self.serializers:
                for serializer in self.serializers[gate_type_mro]:
                    if serializer.can_serialize_operation(op):
                        return True
        return False

    def serialize(self, program, msg=None, *, arg_function_language=None):
        """Serialize a Circuit to cirq_google.api.v2.Program proto.

        Args:
            program: The Circuit to serialize.
        """
        if msg is None:
            msg = program_pb2.Program()
        msg.language.gate_set = self.gate_set_name
        if isinstance(program, cirq.Circuit):
            self._serialize_circuit(program,
                                    msg.circuit,
                                    arg_function_language=arg_function_language)
            if arg_function_language is None:
                arg_function_language = (_infer_function_language_from_circuit(
                    msg.circuit))
        else:
            raise NotImplementedError(
                f'Unrecognized program type: {type(program)}')
        msg.language.arg_function_language = arg_function_language
        return msg

    def serialize_op(
            self,
            op,
            msg=None,
            *,
            arg_function_language='',
    ):
        """Serialize an Operation to cirq_google.api.v2.Operation proto.

        Args:
            op: The operation to serialize.

        Returns:
            A dictionary corresponds to the cirq_google.api.v2.Operation proto.
        """
        gate_type = type(op.gate)
        for gate_type_mro in gate_type.mro():
            # Check all super classes in method resolution order.
            if gate_type_mro in self.serializers:
                # Check each serializer in turn, if serializer proto returns
                # None, then skip.
                for serializer in self.serializers[gate_type_mro]:
                    proto_msg = serializer.to_proto(
                        op, msg, arg_function_language=arg_function_language)
                    if proto_msg is not None:
                        return proto_msg
        raise ValueError('Cannot serialize op {!r} of type {}'.format(
            op, gate_type))

    def deserialize(self, proto, device=None):
        """Deserialize a Circuit from a cirq_google.api.v2.Program.

        Args:
            proto: A dictionary representing a cirq_google.api.v2.Program proto.
            device: If the proto is for a schedule, a device is required
                Otherwise optional.

        Returns:
            The deserialized Circuit, with a device if device was
            not None.
        """
        if not proto.HasField('language') or not proto.language.gate_set:
            raise ValueError('Missing gate set specification.')
        if proto.language.gate_set != self.gate_set_name:
            raise ValueError('Gate set in proto was {} but expected {}'.format(
                proto.language.gate_set, self.gate_set_name))
        which = proto.WhichOneof('program')
        if which == 'circuit':
            circuit = self._deserialize_circuit(
                proto.circuit,
                arg_function_language=proto.language.arg_function_language)
            return circuit if device is None else circuit.with_device(device)

        raise NotImplementedError('Program proto does not contain a circuit.')

    def deserialize_op(
            self,
            operation_proto,
            *,
            arg_function_language='',
    ):
        """Deserialize an Operation from a cirq_google.api.v2.Operation.

        Args:
            operation_proto: A dictionary representing a
                cirq_google.api.v2.Operation proto.

        Returns:
            The deserialized Operation.
        """
        if not operation_proto.gate.id:
            raise ValueError('Operation proto does not have a gate.')

        gate_id = operation_proto.gate.id
        if gate_id not in self.deserializers.keys():
            raise ValueError('Unsupported serialized gate with id "{}".'
                             '\n\noperation_proto:\n{}'.format(
                                 gate_id, operation_proto))

        return self.deserializers[gate_id].from_proto(
            operation_proto, arg_function_language=arg_function_language)

    def _serialize_circuit(self, circuit, msg, *, arg_function_language):
        msg.scheduling_strategy = program_pb2.Circuit.MOMENT_BY_MOMENT
        for moment in circuit:
            moment_proto = msg.moments.add()
            for op in moment:
                self.serialize_op(op,
                                  moment_proto.operations.add(),
                                  arg_function_language=arg_function_language)

    def _deserialize_circuit(
            self,
            circuit_proto,
            *,
            arg_function_language,
    ):
        moments = []
        for i, moment_proto in enumerate(circuit_proto.moments):
            moment_ops = []
            for op in moment_proto.operations:
                try:
                    moment_ops.append(
                        self.deserialize_op(
                            op, arg_function_language=arg_function_language))
                except ValueError as ex:
                    raise ValueError(f'Failed to deserialize circuit. '
                                     f'There was a problem in moment {i} '
                                     f'handling an operation with the '
                                     f'following proto:\n{op}') from ex
            moments.append(cirq.Moment(moment_ops))
        return cirq.Circuit(moments)
