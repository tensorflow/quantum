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
"""Adaption of op_deserializer.py from Cirq 0.9.0."""

import re
import math
import sympy
import cirq

GRID_QUBIT_ID_PATTERN = r'^q?(-?\d+)_(-?\d+)$'
LINE_QUBIT_ID_PATTERN = r'^q?(-?\d+)$'
SUPPORTED_FUNCTIONS_FOR_LANGUAGE = {
    '': frozenset(),
    'linear': frozenset({'add', 'mul'}),
    'exp': frozenset({'add', 'mul', 'pow'}),
    # None means any. Is used when inferring the language during serialization.
    None: frozenset({'add', 'mul', 'pow'}),
}


def qubit_from_proto(proto_id):
    """Parse a proto id to a `cirq.GridQubit`.
    Proto ids for grid qubits are of the form `{row}_{col}` where `{row}` is
    the integer row of the grid qubit, and `{col}` is the integer column of
    the qubit.
    Args:
        proto_id: The id to convert.
    Returns:
        A `cirq.GridQubit` corresponding to the proto id.
    Raises:
        ValueError: If the string not of the correct format.
    """

    match = re.match(GRID_QUBIT_ID_PATTERN, proto_id)
    if match is not None:
        row, col = match.groups()
        return cirq.GridQubit(row=int(row), col=int(col))

    match = re.match(LINE_QUBIT_ID_PATTERN, proto_id)
    if match is not None:
        x, = match.groups()
        return cirq.LineQubit(int(x))

    raise ValueError('Expected GridQubit proto w/ form [q]<int>_<int>,'
                     f' or LineQubit w/ form [q]<int> got {proto_id}')


def _arg_from_proto(
        arg_proto,
        *,
        arg_function_language,
        required_arg_name=None,
):
    """Extracts a python value from an argument value proto.
    Args:
        arg_proto: The proto containing a serialized value.
        arg_function_language: The `arg_function_language` field from
            `Program.Language`.
        required_arg_name: If set to `None`, the method will return `None` when
            given an unset proto value. If set to a string, the method will
            instead raise an error complaining that the value is missing in that
            situation.
    Returns:
        The deserialized value, or else None if there was no set value and
        `required_arg_name` was set to `None`.
    """
    supported = SUPPORTED_FUNCTIONS_FOR_LANGUAGE.get(arg_function_language)
    if supported is None:
        raise ValueError(f'Unrecognized arg_function_language: '
                         f'{arg_function_language!r}')

    which = arg_proto.WhichOneof('arg')
    if which == 'arg_value':
        arg_value = arg_proto.arg_value
        which_val = arg_value.WhichOneof('arg_value')
        if which_val == 'float_value' or which_val == 'double_value':
            if which_val == 'double_value':
                result = float(arg_value.double_value)
            else:
                result = float(arg_value.float_value)
            if math.ceil(result) == math.floor(result):
                result = int(result)
            return result
        if which_val == 'bool_values':
            return list(arg_value.bool_values.values)
        if which_val == 'string_value':
            return str(arg_value.string_value)
        raise ValueError(f'Unrecognized value type: {which_val!r}')

    if which == 'symbol':
        return sympy.Symbol(arg_proto.symbol)

    if which == 'func':
        func = arg_proto.func

        if func.type not in supported:
            raise ValueError(
                f'Unrecognized function type {func.type!r} '
                f'for arg_function_language={arg_function_language!r}')

        if func.type == 'add':
            return sympy.Add(*[
                _arg_from_proto(a,
                                arg_function_language=arg_function_language,
                                required_arg_name='An addition argument')
                for a in func.args
            ])

        if func.type == 'mul':
            return sympy.Mul(*[
                _arg_from_proto(a,
                                arg_function_language=arg_function_language,
                                required_arg_name='A multiplication argument')
                for a in func.args
            ])

        if func.type == 'pow':
            return sympy.Pow(*[
                _arg_from_proto(a,
                                arg_function_language=arg_function_language,
                                required_arg_name='A power argument')
                for a in func.args
            ])

    if required_arg_name is not None:
        raise ValueError(
            f'{required_arg_name} is missing or has an unrecognized '
            f'argument type (WhichOneof("arg")={which!r}).')

    return None


class DeserializingArg:
    """Specification of the arguments to deserialize an argument to a gate.

    Args:
        serialized_name: The serialized name of the gate that is being
            deserialized.
        constructor_arg_name: The name of the argument in the constructor of
            the gate corresponding to this serialized argument.
        value_func: Sometimes a value from the serialized proto needs to
            converted to an appropriate type or form. This function takes the
            serialized value and returns the appropriate type. Defaults to
            None.
        required: Whether a value must be specified when constructing the
            deserialized gate. Defaults to True.
        default: default value to set if the value is not present in the
            arg.  If set, required is ignored.
    """

    def __init__(self,
                 serialized_name,
                 constructor_arg_name,
                 value_func=None,
                 required=True,
                 default=None):
        self.serialized_name = serialized_name
        self.constructor_arg_name = constructor_arg_name
        self.value_func = value_func
        self.required = required
        self.default = default


class GateOpDeserializer:
    """Describes how to deserialize a proto to a given Gate type.

    Attributes:
        serialized_gate_id: The id used when serializing the gate.
    """

    def __init__(self,
                 serialized_gate_id,
                 gate_constructor,
                 args,
                 num_qubits_param=None,
                 op_wrapper=lambda x, y: x):
        """Constructs a deserializer.

        Args:
            serialized_gate_id: The serialized id of the gate that is being
                deserialized.
            gate_constructor: A function that produces the deserialized gate
                given arguments from args.
            args: A list of the arguments to be read from the serialized
                gate and the information required to use this to construct
                the gate using the gate_constructor above.
            num_qubits_param: Some gate constructors require that the number
                of qubits be passed to their constructor. This is the name
                of the parameter in the constructor for this value. If None,
                no number of qubits is passed to the constructor.
            op_wrapper: An optional Callable to modify the resulting
                GateOperation, for instance, to add tags
        """
        self.serialized_gate_id = serialized_gate_id
        self.gate_constructor = gate_constructor
        self.args = args
        self.num_qubits_param = num_qubits_param
        self.op_wrapper = op_wrapper

    def from_proto(self, proto, *, arg_function_language=''):
        """Turns a cirq_google.api.v2.Operation proto into a GateOperation."""
        qubits = [qubit_from_proto(q.id) for q in proto.qubits]
        args = self._args_from_proto(
            proto, arg_function_language=arg_function_language)
        if self.num_qubits_param is not None:
            args[self.num_qubits_param] = len(qubits)
        gate = self.gate_constructor(**args)
        return self.op_wrapper(gate.on(*qubits), proto)

    def _args_from_proto(self, proto, *, arg_function_language):
        return_args = {}
        for arg in self.args:
            if arg.serialized_name not in proto.args:
                if arg.default:
                    return_args[arg.constructor_arg_name] = arg.default
                    continue
                elif arg.required:
                    raise ValueError(
                        f'Argument {arg.serialized_name} '
                        'not in deserializing args, but is required.')

            value = _arg_from_proto(proto.args[arg.serialized_name],
                                    arg_function_language=arg_function_language,
                                    required_arg_name=None if not arg.required
                                    else arg.serialized_name)

            if arg.value_func is not None:
                value = arg.value_func(value)

            if value is not None:
                return_args[arg.constructor_arg_name] = value
        return return_args
