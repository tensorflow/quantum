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
"""op_serializer.py adapated from Cirq release 0.9.0"""

from typing import List
import cirq
import sympy
import numpy as np
from tensorflow_quantum.core.proto import program_pb2

SUPPORTED_SYMPY_OPS = (sympy.Symbol, sympy.Add, sympy.Mul, sympy.Pow)

SUPPORTED_FUNCTIONS_FOR_LANGUAGE = {
    '': frozenset(),
    'linear': frozenset({'add', 'mul'}),
    'exp': frozenset({'add', 'mul', 'pow'}),
    # None means any. Is used when inferring the language during serialization.
    None: frozenset({'add', 'mul', 'pow'}),
}


def qubit_to_proto(qubit):
    """Return proto representation of a GridQubit."""
    if isinstance(qubit, cirq.GridQubit):
        return '{}_{}'.format(qubit.row, qubit.col)
    if isinstance(qubit, cirq.LineQubit):
        return '{}'.format(qubit.x)
    raise ValueError('Unsupported qubit type:' + str(type(qubit)))


def _arg_to_proto(value, *, arg_function_language, out=None):
    """Writes an argument value into an Arg proto.
    Args:
        value: The value to encode.
        arg_function_language: The language to use when encoding functions. If
            this is set to None, it will be set to the minimal language
            necessary to support the features that were actually used.
        out: The proto to write the result into. Defaults to a new instance.
    Returns:
        The proto that was written into as well as the `arg_function_language`
        that was used.
    """

    if arg_function_language not in SUPPORTED_FUNCTIONS_FOR_LANGUAGE:
        raise ValueError(f'Unrecognized arg_function_language: '
                         f'{arg_function_language!r}')
    supported = SUPPORTED_FUNCTIONS_FOR_LANGUAGE[arg_function_language]

    msg = program_pb2.Arg() if out is None else out

    def check_support(func_type: str) -> str:
        if func_type not in supported:
            lang = (repr(arg_function_language)
                    if arg_function_language is not None else '[any]')
            raise ValueError(f'Function type {func_type!r} not supported by '
                             f'arg_function_language {lang}')
        return func_type

    if isinstance(value, (float, int, sympy.Integer, sympy.Float,
                          sympy.Rational, sympy.NumberSymbol)):
        msg.arg_value.float_value = float(value)
    elif isinstance(value, str):
        msg.arg_value.string_value = value
    elif (isinstance(value, (list, tuple, np.ndarray)) and
          all(isinstance(x, (bool, np.bool_)) for x in value)):
        # Some protobuf / numpy combinations do not support np.bool_, so cast.
        msg.arg_value.bool_values.values.extend([bool(x) for x in value])
    elif isinstance(value, sympy.Symbol):
        msg.symbol = str(value.free_symbols.pop())
    elif isinstance(value, sympy.Add):
        msg.func.type = check_support('add')
        for arg in value.args:
            _arg_to_proto(arg,
                          arg_function_language=arg_function_language,
                          out=msg.func.args.add())
    elif isinstance(value, sympy.Mul):
        msg.func.type = check_support('mul')
        for arg in value.args:
            _arg_to_proto(arg,
                          arg_function_language=arg_function_language,
                          out=msg.func.args.add())
    elif isinstance(value, sympy.Pow):
        msg.func.type = check_support('pow')
        for arg in value.args:
            _arg_to_proto(arg,
                          arg_function_language=arg_function_language,
                          out=msg.func.args.add())
    else:
        raise ValueError(f'Unrecognized arg type: {type(value)}')

    return msg


class SerializingArg:
    """Specification of the arguments for a Gate and its serialization.

    Args:
        serialized_name: The name of the argument when it is serialized.
        serialized_type: The type of the argument when it is serialized.
        op_getter: The name of the property or attribute for getting the
            value of this argument from a gate, or a function that takes a
            operation and returns this value. The later can be used to supply
            a value of the serialized arg by supplying a lambda that
            returns this value (i.e. `lambda x: default_value`)
        required: Whether this argument is a required argument for the
            serialized form.
        default: default value.  avoid serializing if this is the value.
            Note that the DeserializingArg must also have this as default.
    """

    def __init__(self,
                 serialized_name,
                 serialized_type,
                 op_getter,
                 required=True,
                 default=None):
        self.serialized_name = serialized_name
        self.serialized_type = serialized_type
        self.op_getter = op_getter
        self.required = required
        self.default = default


class GateOpSerializer:
    """Describes how to serialize a GateOperation for a given Gate type.

    Attributes:
        gate_type: The type of the gate that can be serialized.
        serialized_gate_id: The id used when serializing the gate.
    """

    def __init__(self,
                 *,
                 gate_type,
                 serialized_gate_id,
                 args,
                 can_serialize_predicate=lambda x: True):
        """Construct the serializer.

        Args:
            gate_type: The type of the gate that is being serialized.
            serialized_gate_id: The string id of the gate when serialized.
            can_serialize_predicate: Sometimes an Operation can only be
                serialized for particular parameters. This predicate will be
                checked before attempting to serialize the Operation. If the
                predicate is False, serialization will result in a None value.
                Default value is a lambda that always returns True.
            args: A list of specification of the arguments to the gate when
                serializing, including how to get this information from the
                gate of the given gate type.
        """
        self.gate_type = gate_type
        self.serialized_gate_id = serialized_gate_id
        self.args = args
        self.can_serialize_predicate = can_serialize_predicate

    def can_serialize_operation(self, op):
        """Whether the given operation can be serialized by this serializer.

        This checks that the gate is a subclass of the gate type for this
        serializer, and that the gate returns true for
        `can_serializer_predicate` called on the gate.
        """
        supported_gate_type = self.gate_type in type(op.gate).mro()
        return supported_gate_type and self.can_serialize_predicate(op)

    def to_proto(
            self,
            op,
            msg=None,
            *,
            arg_function_language='',
    ):
        """Returns the cirq_google.api.v2.Operation message as a proto dict."""

        gate = op.gate
        if not isinstance(gate, self.gate_type):
            raise ValueError(
                'Gate of type {} but serializer expected type {}'.format(
                    type(gate), self.gate_type))

        if not self.can_serialize_predicate(op):
            return None

        if msg is None:
            msg = program_pb2.Operation()

        msg.gate.id = self.serialized_gate_id
        for qubit in op.qubits:
            msg.qubits.add().id = qubit_to_proto(qubit)
        for arg in self.args:
            value = self._value_from_gate(op, arg)
            if value is not None and (not arg.default or value != arg.default):
                _arg_to_proto(value,
                              out=msg.args[arg.serialized_name],
                              arg_function_language=arg_function_language)
        return msg

    def _value_from_gate(self, op, arg):
        value = None
        op_getter = arg.op_getter
        if isinstance(op_getter, str):
            gate = op.gate
            value = getattr(gate, op_getter, None)
            if value is None and arg.required:
                raise ValueError(
                    'Gate {!r} does not have attribute or property {}'.format(
                        gate, op_getter))
        elif callable(op_getter):
            value = op_getter(op)

        if arg.required and value is None:
            raise ValueError(
                'Argument {} is required, but could not get from op {!r}'.
                format(arg.serialized_name, op))

        if isinstance(value, SUPPORTED_SYMPY_OPS):
            return value

        if value is not None:
            self._check_type(value, arg)

        return value

    def _check_type(self, value, arg):
        if arg.serialized_type == float:
            if not isinstance(value, (float, int)):
                raise ValueError(
                    'Expected type convertible to float but was {}'.format(
                        type(value)))
        elif arg.serialized_type == List[bool]:
            if (not isinstance(value, (list, tuple, np.ndarray)) or
                    not all(isinstance(x, (bool, np.bool_)) for x in value)):
                raise ValueError('Expected type List[bool] but was {}'.format(
                    type(value)))
        elif value is not None and not isinstance(value, arg.serialized_type):
            raise ValueError(
                'Argument {} had type {} but gate returned type {}'.format(
                    arg.serialized_name, arg.serialized_type, type(value)))
