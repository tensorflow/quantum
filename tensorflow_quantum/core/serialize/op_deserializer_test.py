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
"""Test op deserialization correctness."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

from typing import List
import tensorflow as tf

import cirq
import sympy
from absl.testing import parameterized
from google.protobuf import json_format
from tensorflow_quantum.core.proto import program_pb2
from tensorflow_quantum.core.serialize import op_deserializer


def op_proto(json_dict):
    """Json to proto."""
    op = program_pb2.Operation()
    json_format.ParseDict(json_dict, op)
    return op


@cirq.value_equality
class GateWithAttribute(cirq.testing.SingleQubitGate):
    """GateAttribute helper class."""

    def __init__(self, val, not_req=None):
        self.val = val
        self.not_req = not_req

    def _value_equality_values_(self):
        return (self.val,)


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
    (sympy.Symbol, sympy.Symbol('x'), {
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


class OpDeserializerTest(tf.test.TestCase, parameterized.TestCase):
    """Test OpDeserializer functionality."""

    @parameterized.parameters([
        CASE + (x,)
        for CASE in TEST_CASES
        for x in [cirq.GridQubit(1, 2), cirq.LineQubit(4)]
    ])
    def test_from_proto(self, val_type, val, arg_value, q):
        """Test from proto under many cases."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='my_val',
                    constructor_arg_name='val',
                )
            ])
        serialized = op_proto({
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
        result = deserializer.from_proto(serialized,
                                         arg_function_language='linear')
        self.assertEqual(result, GateWithAttribute(val)(q))

    def test_from_proto_required_missing(self):
        """Test error raised when required is missing."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='my_val',
                    constructor_arg_name='val',
                )
            ])
        serialized = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'not_my_val': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                }
            },
            'qubits': [{
                'id': '1_2'
            }]
        })
        with self.assertRaisesRegex(Exception, expected_regex='my_val'):
            deserializer.from_proto(serialized)

    def test_from_proto_unknown_function(self):
        """Unknown function throws error when deserializing."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='my_val',
                    constructor_arg_name='val',
                )
            ])
        serialized = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': {
                    'func': {
                        'type':
                            'UNKNOWN_OPERATION',
                        'args': [
                            {
                                'symbol': 'x'
                            },
                            {
                                'arg_value': {
                                    'float_value': -1.0
                                }
                            },
                        ]
                    }
                }
            },
            'qubits': [{
                'id': '1_2'
            }]
        })
        with self.assertRaisesRegex(
                ValueError, expected_regex='Unrecognized function type'):
            _ = deserializer.from_proto(serialized)

    def test_from_proto_value_type_not_recognized(self):
        """Ensure unrecognized value type errors."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='my_val',
                    constructor_arg_name='val',
                )
            ])
        serialized = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': {
                    'arg_value': {},
                }
            },
            'qubits': [{
                'id': '1_2'
            }]
        })
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='Unrecognized value type'):
            _ = deserializer.from_proto(serialized)

    def test_from_proto_function_argument_not_set(self):
        """Ensure unset function arguments error when deserializing."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='my_val',
                    constructor_arg_name='val',
                )
            ])
        serialized = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': {
                    'func': {
                        'type': 'mul',
                        'args': [
                            {
                                'symbol': 'x'
                            },
                            {},
                        ]
                    }
                }
            },
            'qubits': [{
                'id': '1_2'
            }]
        })
        with self.assertRaisesRegex(
                ValueError,
                expected_regex='A multiplication argument is missing'):
            _ = deserializer.from_proto(serialized,
                                        arg_function_language='linear')

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_from_proto_value_func(self, q):
        """Test value func deserialization in simple case."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(serialized_name='my_val',
                                                 constructor_arg_name='val',
                                                 value_func=lambda x: x + 1)
            ])
        serialized = op_proto({
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
        result = deserializer.from_proto(serialized)
        self.assertEqual(result, GateWithAttribute(1.125)(q))

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_from_proto_not_required_ok(self, q):
        """Deserialization succeeds for missing not required fields."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='my_val',
                    constructor_arg_name='val',
                ),
                op_deserializer.DeserializingArg(serialized_name='not_req',
                                                 constructor_arg_name='not_req',
                                                 required=False)
            ])
        serialized = op_proto({
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
        result = deserializer.from_proto(serialized)
        self.assertEqual(result, GateWithAttribute(0.125)(q))

    def test_from_proto_missing_required_arg(self):
        """Error raised when required field is missing."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='my_val',
                    constructor_arg_name='val',
                ),
                op_deserializer.DeserializingArg(serialized_name='not_req',
                                                 constructor_arg_name='not_req',
                                                 required=False)
            ])
        serialized = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'not_req': {
                    'arg_value': {
                        'float_value': 0.125
                    }
                }
            },
            'qubits': [{
                'id': '1_2'
            }]
        })
        with self.assertRaises(ValueError):
            deserializer.from_proto(serialized)

    def test_from_proto_required_arg_not_assigned(self):
        """Error if required arg isn't assigned."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(
                    serialized_name='my_val',
                    constructor_arg_name='val',
                ),
                op_deserializer.DeserializingArg(serialized_name='not_req',
                                                 constructor_arg_name='not_req',
                                                 required=False)
            ])
        serialized = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {
                'my_val': {}
            },
            'qubits': [{
                'id': '1_2'
            }]
        })
        with self.assertRaises(ValueError):
            deserializer.from_proto(serialized)

    @parameterized.parameters([cirq.GridQubit(1, 2), cirq.LineQubit(4)])
    def test_defaults(self, q):
        """Ensure default values still deserialize."""
        deserializer = op_deserializer.GateOpDeserializer(
            serialized_gate_id='my_gate',
            gate_constructor=GateWithAttribute,
            args=[
                op_deserializer.DeserializingArg(serialized_name='my_val',
                                                 constructor_arg_name='val',
                                                 default=1.0),
                op_deserializer.DeserializingArg(serialized_name='not_req',
                                                 constructor_arg_name='not_req',
                                                 default='hello',
                                                 required=False)
            ])
        serialized = op_proto({
            'gate': {
                'id': 'my_gate'
            },
            'args': {},
            'qubits': [{
                'id': '1_2' if isinstance(q, cirq.GridQubit) else '4'
            }]
        })
        g = GateWithAttribute(1.0)
        g.not_req = 'hello'
        self.assertEqual(deserializer.from_proto(serialized), g(q))


if __name__ == "__main__":
    tf.test.main()
