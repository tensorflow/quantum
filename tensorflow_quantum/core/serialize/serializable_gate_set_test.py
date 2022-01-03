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
"""Test serializable_gat_set.py functionality."""
# Remove PYTHONPATH collisions for protobuf.
import sys
new_path = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = new_path

import tensorflow as tf

import cirq
from google.protobuf import json_format
from tensorflow_quantum.core.serialize import op_serializer, op_deserializer, \
    serializable_gate_set
from tensorflow_quantum.core.proto import program_pb2

X_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=cirq.XPowGate,
    serialized_gate_id='x_pow',
    args=[
        op_serializer.SerializingArg(
            serialized_name='half_turns',
            serialized_type=float,
            op_getter='exponent',
        )
    ],
)

X_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='x_pow',
    gate_constructor=cirq.XPowGate,
    args=[
        op_deserializer.DeserializingArg(
            serialized_name='half_turns',
            constructor_arg_name='exponent',
        )
    ],
)

Y_SERIALIZER = op_serializer.GateOpSerializer(
    gate_type=cirq.YPowGate,
    serialized_gate_id='y_pow',
    args=[
        op_serializer.SerializingArg(
            serialized_name='half_turns',
            serialized_type=float,
            op_getter='exponent',
        )
    ],
)

Y_DESERIALIZER = op_deserializer.GateOpDeserializer(
    serialized_gate_id='y_pow',
    gate_constructor=cirq.YPowGate,
    args=[
        op_deserializer.DeserializingArg(
            serialized_name='half_turns',
            constructor_arg_name='exponent',
        )
    ],
)

MY_GATE_SET = serializable_gate_set.SerializableGateSet(
    gate_set_name='my_gate_set',
    serializers=[X_SERIALIZER],
    deserializers=[X_DESERIALIZER],
)


def op_proto(json):
    """Json to proto."""
    op = program_pb2.Operation()
    json_format.ParseDict(json, op)
    return op


class SerializableGateSetTest(tf.test.TestCase):
    """Test SerializableGateSet and associated deserialize functionality."""

    def test_supported_gate_types(self):
        """Report correct gate types."""
        self.assertEqual(MY_GATE_SET.supported_gate_types(), (cirq.XPowGate,))

    def test_is_supported_operation(self):
        """Ensure supported operations are correct."""
        q = cirq.GridQubit(1, 1)
        self.assertTrue(MY_GATE_SET.is_supported_operation(cirq.XPowGate()(q)))
        self.assertTrue(MY_GATE_SET.is_supported_operation(cirq.X(q)))
        self.assertFalse(MY_GATE_SET.is_supported_operation(cirq.ZPowGate()(q)))

    def test_is_supported_operation_can_serialize_predicate(self):
        """Test can_serialize predicate for operations."""
        q = cirq.GridQubit(1, 2)
        serializer = op_serializer.GateOpSerializer(
            gate_type=cirq.XPowGate,
            serialized_gate_id='x_pow',
            args=[
                op_serializer.SerializingArg(
                    serialized_name='half_turns',
                    serialized_type=float,
                    op_getter='exponent',
                )
            ],
            can_serialize_predicate=lambda x: x.gate.exponent == 1.0)
        gate_set = serializable_gate_set.SerializableGateSet(
            gate_set_name='my_gate_set',
            serializers=[serializer],
            deserializers=[X_DESERIALIZER])
        self.assertTrue(gate_set.is_supported_operation(cirq.XPowGate()(q)))
        self.assertFalse(
            gate_set.is_supported_operation(cirq.XPowGate()(q)**0.5))
        self.assertTrue(gate_set.is_supported_operation(cirq.X(q)))

    def test_serialize_deserialize_circuit(self):
        """Verify one to one serialize deserialize consistency."""
        q0 = cirq.GridQubit(1, 1)
        q1 = cirq.GridQubit(1, 2)
        circuit = cirq.Circuit(cirq.X(q0), cirq.X(q1), cirq.X(q0))

        proto = program_pb2.Program(
            language=program_pb2.Language(arg_function_language='',
                                          gate_set='my_gate_set'),
            circuit=program_pb2.Circuit(
                scheduling_strategy=program_pb2.Circuit.MOMENT_BY_MOMENT,
                moments=[
                    program_pb2.Moment(operations=[
                        X_SERIALIZER.to_proto(cirq.X(q0)),
                        X_SERIALIZER.to_proto(cirq.X(q1))
                    ]),
                    program_pb2.Moment(
                        operations=[X_SERIALIZER.to_proto(cirq.X(q0))]),
                ]))
        self.assertEqual(proto, MY_GATE_SET.serialize(circuit))
        self.assertEqual(MY_GATE_SET.deserialize(proto), circuit)

    def test_deserialize_bad_operation_id(self):
        """Ensure error is raised when deserializing bad operation."""
        proto = program_pb2.Program(
            language=program_pb2.Language(arg_function_language='',
                                          gate_set='my_gate_set'),
            circuit=program_pb2.Circuit(
                scheduling_strategy=program_pb2.Circuit.MOMENT_BY_MOMENT,
                moments=[
                    program_pb2.Moment(operations=[]),
                    program_pb2.Moment(operations=[
                        program_pb2.Operation(
                            gate=program_pb2.Gate(id='UNKNOWN_GATE'),
                            args={
                                'half_turns':
                                    program_pb2.Arg(
                                        arg_value=program_pb2.ArgValue(
                                            float_value=1.0))
                            },
                            qubits=[program_pb2.Qubit(id='1_1')])
                    ]),
                ]))
        with self.assertRaisesRegex(
                ValueError,
                expected_regex='problem in moment 1 handling an '
                'operation with the following'):
            MY_GATE_SET.deserialize(proto)

    def test_serialize_deserialize_empty_circuit(self):
        """Verify empty case serialize deserialize works."""
        circuit = cirq.Circuit()

        proto = program_pb2.Program(
            language=program_pb2.Language(arg_function_language='',
                                          gate_set='my_gate_set'),
            circuit=program_pb2.Circuit(
                scheduling_strategy=program_pb2.Circuit.MOMENT_BY_MOMENT,
                moments=[]))
        self.assertEqual(proto, MY_GATE_SET.serialize(circuit))
        self.assertEqual(MY_GATE_SET.deserialize(proto), circuit)

    def test_deserialize_empty_moment(self):
        """Ensure deserialize empty moment works."""
        circuit = cirq.Circuit([cirq.Moment()])

        proto = program_pb2.Program(
            language=program_pb2.Language(arg_function_language='',
                                          gate_set='my_gate_set'),
            circuit=program_pb2.Circuit(
                scheduling_strategy=program_pb2.Circuit.MOMENT_BY_MOMENT,
                moments=[
                    program_pb2.Moment(),
                ]))
        self.assertEqual(MY_GATE_SET.deserialize(proto), circuit)

    def test_serialize_unrecognized(self):
        """Error on uncrecognized serialization."""
        with self.assertRaisesRegex(NotImplementedError,
                                    expected_regex='program type'):
            MY_GATE_SET.serialize("not quite right")

    def test_serialize_deserialize_op(self):
        """Simple serialize and deserialize back test."""
        q0 = cirq.GridQubit(1, 1)
        proto = op_proto({
            'gate': {
                'id': 'x_pow'
            },
            'args': {
                'half_turns': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                },
            },
            'qubits': [{
                'id': '1_1'
            }]
        })
        self.assertEqual(
            proto, MY_GATE_SET.serialize_op(cirq.XPowGate(exponent=0.125)(q0)))
        self.assertEqual(MY_GATE_SET.deserialize_op(proto),
                         cirq.XPowGate(exponent=0.125)(q0))

    def test_serialize_deserialize_op_subclass(self):
        """Verify subclasses can serialize and deserialize back."""
        q0 = cirq.GridQubit(1, 1)
        proto = op_proto({
            'gate': {
                'id': 'x_pow'
            },
            'args': {
                'half_turns': {
                    'arg_value': {
                        'float_value': 1.0
                    }
                },
            },
            'qubits': [{
                'id': '1_1'
            }]
        })
        # cirq.X is a subclass of XPowGate.
        self.assertEqual(proto, MY_GATE_SET.serialize_op(cirq.X(q0)))
        self.assertEqual(MY_GATE_SET.deserialize_op(proto), cirq.X(q0))

    def test_multiple_serializers(self):
        """Compound serialization."""
        serializer1 = op_serializer.GateOpSerializer(
            gate_type=cirq.XPowGate,
            serialized_gate_id='x_pow',
            args=[
                op_serializer.SerializingArg(serialized_name='half_turns',
                                             serialized_type=float,
                                             op_getter='exponent')
            ],
            can_serialize_predicate=lambda x: x.gate.exponent != 1)
        serializer2 = op_serializer.GateOpSerializer(
            gate_type=cirq.XPowGate,
            serialized_gate_id='x',
            args=[
                op_serializer.SerializingArg(serialized_name='half_turns',
                                             serialized_type=float,
                                             op_getter='exponent')
            ],
            can_serialize_predicate=lambda x: x.gate.exponent == 1)
        gate_set = serializable_gate_set.SerializableGateSet(
            gate_set_name='my_gate_set',
            serializers=[serializer1, serializer2],
            deserializers=[])
        q0 = cirq.GridQubit(1, 1)
        self.assertEqual(gate_set.serialize_op(cirq.X(q0)).gate.id, 'x')
        self.assertEqual(
            gate_set.serialize_op(cirq.X(q0)**0.5).gate.id, 'x_pow')

    def test_gateset_with_added_gates(self):
        """Test adding new gates to gateset."""
        q = cirq.GridQubit(1, 1)
        x_gateset = serializable_gate_set.SerializableGateSet(
            gate_set_name='x',
            serializers=[X_SERIALIZER],
            deserializers=[X_DESERIALIZER],
        )
        xy_gateset = x_gateset.with_added_gates(
            gate_set_name='xy',
            serializers=[Y_SERIALIZER],
            deserializers=[Y_DESERIALIZER],
        )
        self.assertEqual(x_gateset.gate_set_name, 'x')
        self.assertTrue(x_gateset.is_supported_operation(cirq.X(q)))
        self.assertFalse(x_gateset.is_supported_operation(cirq.Y(q)))
        self.assertEqual(xy_gateset.gate_set_name, 'xy')
        self.assertTrue(xy_gateset.is_supported_operation(cirq.X(q)))
        self.assertTrue(xy_gateset.is_supported_operation(cirq.Y(q)))

        # test serialization and deserialization
        proto = op_proto({
            'gate': {
                'id': 'y_pow'
            },
            'args': {
                'half_turns': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                },
            },
            'qubits': [{
                'id': '1_1'
            }]
        })

        expected_gate = cirq.YPowGate(exponent=0.125)(cirq.GridQubit(1, 1))
        self.assertEqual(xy_gateset.serialize_op(expected_gate), proto)
        self.assertEqual(xy_gateset.deserialize_op(proto), expected_gate)

    def test_gateset_with_added_gates_again(self):
        """Verify that adding a serializer twice doesn't mess anything up."""
        q = cirq.GridQubit(2, 2)
        x_gateset = serializable_gate_set.SerializableGateSet(
            gate_set_name='x',
            serializers=[X_SERIALIZER],
            deserializers=[X_DESERIALIZER],
        )
        xx_gateset = x_gateset.with_added_gates(
            gate_set_name='xx',
            serializers=[X_SERIALIZER],
            deserializers=[X_DESERIALIZER],
        )

        self.assertEqual(xx_gateset.gate_set_name, 'xx')
        self.assertTrue(xx_gateset.is_supported_operation(cirq.X(q)))
        self.assertFalse(xx_gateset.is_supported_operation(cirq.Y(q)))

        # test serialization and deserialization
        proto = op_proto({
            'gate': {
                'id': 'x_pow'
            },
            'args': {
                'half_turns': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                },
            },
            'qubits': [{
                'id': '1_1'
            }]
        })

        expected_gate = cirq.XPowGate(exponent=0.125)(cirq.GridQubit(1, 1))
        self.assertEqual(xx_gateset.serialize_op(expected_gate), proto)
        self.assertEqual(xx_gateset.deserialize_op(proto), expected_gate)

    def test_deserialize_op_invalid_gate(self):
        """Ensure deserialize invalid gates errors."""
        proto = op_proto({
            'gate': {},
            'args': {
                'half_turns': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                },
            },
            'qubits': [{
                'id': '1_1'
            }]
        })
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='does not have a gate'):
            MY_GATE_SET.deserialize_op(proto)

        proto = op_proto({
            'args': {
                'half_turns': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                },
            },
            'qubits': [{
                'id': '1_1'
            }]
        })
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='does not have a gate'):
            MY_GATE_SET.deserialize_op(proto)

    def test_deserialize_unsupported_gate_type(self):
        """Ensure deserializing unsupported types errors."""
        proto = op_proto({
            'gate': {
                'id': 'no_pow'
            },
            'args': {
                'half_turns': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                },
            },
            'qubits': [{
                'id': '1_1'
            }]
        })
        with self.assertRaisesRegex(ValueError, expected_regex='no_pow'):
            MY_GATE_SET.deserialize_op(proto)

    def test_serialize_op_unsupported_type(self):
        """Serialize unsopported types should error."""
        q0 = cirq.GridQubit(1, 1)
        with self.assertRaisesRegex(ValueError, expected_regex='YPowGate'):
            MY_GATE_SET.serialize_op(cirq.YPowGate()(q0))

    def test_deserialize_invalid_gate_set(self):
        """Deserializing an invalid gate set should error if element not in."""
        proto = program_pb2.Program(
            language=program_pb2.Language(gate_set='not_my_gate_set'),
            circuit=program_pb2.Circuit(
                scheduling_strategy=program_pb2.Circuit.MOMENT_BY_MOMENT,
                moments=[]))
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='not_my_gate_set'):
            MY_GATE_SET.deserialize(proto)

        proto.language.gate_set = ''
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='Missing gate set'):
            MY_GATE_SET.deserialize(proto)

        proto = program_pb2.Program(circuit=program_pb2.Circuit(
            scheduling_strategy=program_pb2.Circuit.MOMENT_BY_MOMENT,
            moments=[]))
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='Missing gate set'):
            MY_GATE_SET.deserialize(proto)


if __name__ == "__main__":
    tf.test.main()
