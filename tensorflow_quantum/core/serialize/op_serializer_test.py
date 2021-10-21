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
"""Adaption of op_serializer_test.py from Cirq 0.9.0."""

from typing import List
import numpy as np
import tensorflow as tf

import cirq
import sympy
from absl.testing import parameterized
from google.protobuf import json_format
from tensorflow_quantum.core.proto import program_pb2
from tensorflow_quantum.core.serialize import op_serializer


def op_proto(json):
    """Concert json to a proto."""
    op = program_pb2.Operation()
    json_format.ParseDict(json, op)
    return op


class GateWithAttribute(cirq.SingleQubitGate):
    """GateAttribute helper class."""

    def __init__(self, val):
        self.val = val


class GateWithProperty(cirq.SingleQubitGate):
    """GateProperty helper class."""

    def __init__(self, val, not_req=None):
        self._val = val
        self._not_req = not_req

    @property
    def val(self):
        """get val."""
        return self._val


class GateWithMethod(cirq.SingleQubitGate):
    """GateMethod helper class."""

    def __init__(self, val):
        self._val = val

    def get_val(self):
        """get val."""
        return self._val


class SubclassGate(GateWithAttribute):
    """EmptyGate helper class."""


def get_val(op):
    """Get value of op."""
    return op.gate.get_val()


TEST_CASES = [
    (float, 1.0, {
        'arg_value': {
            'float_value': 1.0
        }
    }),
    (str, 'abc', {
        'arg_value': {
            'string_value': 'abc'
        }
    }),
    (float, 1, {
        'arg_value': {
            'float_value': 1.0
        }
    }),
    (List[bool], [True, False], {
        'arg_value': {
            'bool_values': {
                'values': [True, False]
            }
        }
    }),
    (List[bool], (True, False), {
        'arg_value': {
            'bool_values': {
                'values': [True, False]
            }
        }
    }),
    (List[bool], np.array([True, False], dtype=np.bool), {
        'arg_value': {
            'bool_values': {
                'values': [True, False]
            }
        }
    }),
    (sympy.Symbol, sympy.Symbol('x'), {
        'symbol': 'x'
    }),
    (float, sympy.Symbol('x'), {
        'symbol': 'x'
    }),
    (float, sympy.Symbol('x') - sympy.Symbol('y'), {
        'func': {
            'type':
                'add',
            'args': [{
                'symbol': 'x'
            }, {
                'func': {
                    'type':
                        'mul',
                    'args': [{
                        'arg_value': {
                            'float_value': -1.0
                        }
                    }, {
                        'symbol': 'y'
                    }]
                }
            }]
        }
    }),
]


class OpSerializerTest(tf.test.TestCase, parameterized.TestCase):
    """Test OpSerializer functions correctly."""

    @parameterized.parameters([
        CASE + (x,)
        for CASE in TEST_CASES
        for x in [cirq.GridQubit(1, 2), cirq.LineQubit(4)]
    ])
    def test_to_proto_attribute(self, val_type, val, arg_value, q):
        """Test proto attribute serialization works."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithAttribute,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=val_type,
                                             op_getter='val')
            ])
        result = serializer.to_proto(GateWithAttribute(val)(q),
                                     arg_function_language='linear')
        expected = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': arg_value
            },
            'qubits': [{
                'id': '1_2' if isinstance(q, cirq.GridQubit) else '4'
            }]
        })
        self.assertEqual(result, expected)

    @parameterized.parameters([
        CASE + (x,)
        for CASE in TEST_CASES
        for x in [cirq.GridQubit(1, 2), cirq.LineQubit(4)]
    ])
    def test_to_proto_property(self, val_type, val, arg_value, q):
        """Test proto property serialization works."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithProperty,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=val_type,
                                             op_getter='val')
            ])
        result = serializer.to_proto(GateWithProperty(val)(q),
                                     arg_function_language='linear')
        expected = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': arg_value
            },
            'qubits': [{
                'id': '1_2' if isinstance(q, cirq.GridQubit) else '4'
            }]
        })
        self.assertEqual(result, expected)

    @parameterized.parameters([
        CASE + (x,)
        for CASE in TEST_CASES
        for x in [cirq.GridQubit(1, 2), cirq.LineQubit(4)]
    ])
    def test_to_proto_callable(self, val_type, val, arg_value, q):
        """Test callable serialization works."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithMethod,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=val_type,
                                             op_getter=get_val)
            ])
        result = serializer.to_proto(GateWithMethod(val)(q),
                                     arg_function_language='linear')
        expected = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': arg_value
            },
            'qubits': [{
                'id': '1_2' if isinstance(q, cirq.GridQubit) else '4'
            }]
        })
        self.assertEqual(result, expected)

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_to_proto_gate_predicate(self, q):
        """Test can_serialize works."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithAttribute,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=float,
                                             op_getter='val')
            ],
            can_serialize_predicate=lambda x: x.gate.val == 1)
        self.assertIsNone(serializer.to_proto(GateWithAttribute(0)(q)))
        self.assertIsNotNone(serializer.to_proto(GateWithAttribute(1)(q)))
        self.assertFalse(
            serializer.can_serialize_operation(GateWithAttribute(0)(q)))
        self.assertTrue(
            serializer.can_serialize_operation(GateWithAttribute(1)(q)))

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_to_proto_gate_mismatch(self, q):
        """Test proto gate mismatch errors."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithProperty,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=float,
                                             op_getter='val')
            ])
        with self.assertRaisesRegex(
                ValueError,
                expected_regex='GateWithAttribute.*GateWithProperty'):
            serializer.to_proto(GateWithAttribute(1.0)(q))

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_to_proto_unsupported_type(self, q):
        """Test proto unsupported types errors."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithProperty,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=bytes,
                                             op_getter='val')
            ])
        with self.assertRaisesRegex(ValueError, expected_regex='bytes'):
            serializer.to_proto(GateWithProperty(b's')(q))

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_to_proto_required_but_not_present(self, q):
        """Test required and missing args errors."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithProperty,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=float,
                                             op_getter=lambda x: None)
            ])
        with self.assertRaisesRegex(ValueError, expected_regex='required'):
            serializer.to_proto(GateWithProperty(1.0)(q))

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_to_proto_no_getattr(self, q):
        """Test no op getter fails."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithProperty,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=float,
                                             op_getter='nope')
            ])
        with self.assertRaisesRegex(ValueError, expected_regex='does not have'):
            serializer.to_proto(GateWithProperty(1.0)(q))

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_to_proto_not_required_ok(self, q):
        """Test non require arg absense succeeds."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithProperty,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=float,
                                             op_getter='val'),
                op_serializer.SerializingArg(serialized_name='not_req',
                                             serialized_type=float,
                                             op_getter='not_req',
                                             required=False)
            ])
        expected = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                }
            },
            'qubits': [{
                'id': '1_2' if isinstance(q, cirq.GridQubit) else '4'
            }]
        })

        self.assertEqual(serializer.to_proto(GateWithProperty(0.125)(q)),
                         expected)

    @parameterized.parameters([{
        **x,
        **{
            'q': q
        }
    } for x in [{
        'val_type': float,
        'val': 's'
    }, {
        'val_type': str,
        'val': 1.0
    }, {
        'val_type': sympy.Symbol,
        'val': 1.0
    }, {
        'val_type': List[bool],
        'val': [1.0]
    }, {
        'val_type': List[bool],
        'val': 'a'
    }, {
        'val_type': List[bool],
        'val': (1.0,)
    }] for q in [cirq.GridQubit(1, 2), cirq.LineQubit(4)]])
    def test_to_proto_type_mismatch(self, val_type, val, q):
        """Test type mismatch fails."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithProperty,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=val_type,
                                             op_getter='val')
            ])
        with self.assertRaisesRegex(ValueError, expected_regex=str(type(val))):
            serializer.to_proto(GateWithProperty(val)(q))

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_can_serialize_operation_subclass(self, q):
        """Test can serialize subclass."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithAttribute,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=float,
                                             op_getter='val')
            ],
            can_serialize_predicate=lambda x: x.gate.val == 1)
        self.assertTrue(serializer.can_serialize_operation(SubclassGate(1)(q)))
        self.assertFalse(serializer.can_serialize_operation(SubclassGate(0)(q)))

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_defaults_not_serialized(self, q):
        """Test defaults not serialized."""
        serializer = op_serializer.GateOpSerializer(
            gate_type=GateWithAttribute,
            serialized_gate_id='my_gate',
            args=[
                op_serializer.SerializingArg(serialized_name='my_val',
                                             serialized_type=float,
                                             default=1.0,
                                             op_getter='val')
            ])
        no_default = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                }
            },
            'qubits': [{
                'id': '1_2' if isinstance(q, cirq.GridQubit) else '4'
            }]
        })
        self.assertEqual(no_default,
                         serializer.to_proto(GateWithAttribute(0.125)(q)))
        with_default = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'qubits': [{
                'id': '1_2' if isinstance(q, cirq.GridQubit) else '4'
            }]
        })
        self.assertEqual(with_default,
                         serializer.to_proto(GateWithAttribute(1.0)(q)))


if __name__ == "__main__":
    tf.test.main()
