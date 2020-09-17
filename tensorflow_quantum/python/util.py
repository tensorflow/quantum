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
"""A collection of helper functions that are useful several places in tfq."""
import random
import itertools

import numpy as np
import sympy
import tensorflow as tf
import cirq

from tensorflow_quantum.core.proto import pauli_sum_pb2
from tensorflow_quantum.core.serialize import serializer


def get_supported_gates():
    """A helper to get the gates supported by tfq."""
    supported_gates = serializer.SERIALIZER.supported_gate_types()
    gate_arity_mapping_dict = dict()
    for gate in supported_gates:
        if gate is cirq.IdentityGate:
            g_num_qubits = 1
            g = gate(num_qubits=1)
        elif gate is cirq.FSimGate:
            g_num_qubits = 2
            g = gate(theta=0.123, phi=0.456)
        elif gate in serializer.PHASED_EIGEN_GATES_DICT:
            g = gate(phase_exponent=0.123)
            g_num_qubits = g.num_qubits()
        else:
            g = gate()
            g_num_qubits = gate().num_qubits()
        gate_arity_mapping_dict[g] = g_num_qubits
    return gate_arity_mapping_dict


def random_symbol_circuit(qubits,
                          symbols,
                          n_moments=15,
                          p=0.9,
                          include_scalars=True):
    """Generate a random circuit including some parameterized gates.

    Symbols are randomly included in the gates of the first `n_moments` moments
    of the resulting circuit.  Then, parameterized H gates are added as
    subsequent moments for any remaining unused symbols.
    """
    supported_gates = get_supported_gates()
    circuit = cirq.testing.random_circuit(qubits, n_moments, p, supported_gates)
    random_symbols = list(symbols)
    random.shuffle(random_symbols)
    location = 0

    for i in range(len(circuit)):
        if np.random.random() < p:
            op = random.choice(list(supported_gates.keys()))
            n_qubits = supported_gates[op]
            locs = tuple(random.sample(qubits, n_qubits))
            if isinstance(op, cirq.IdentityGate):
                circuit[:i] += op.on(*locs)
            else:
                circuit[:i] += (op**(
                    (np.random.random() if include_scalars else 1.0) *
                    sympy.Symbol(random_symbols[location % len(random_symbols)])
                )).on(*locs)
                location += 1

    # Use the rest of the symbols
    while location < len(random_symbols):
        circuit += cirq.Circuit(
            cirq.H(qubits[0])**sympy.Symbol(random_symbols[location]))
        location += 1

    return circuit


def random_circuit_resolver_batch(qubits, batch_size, n_moments=15, p=0.9):
    """Generate a batch of random circuits and symbolless resolvers."""
    return_circuits = []
    return_resolvers = []
    for _ in range(batch_size):
        return_circuits.append(
            cirq.testing.random_circuit(qubits, n_moments, p,
                                        get_supported_gates()))
        return_resolvers.append(cirq.ParamResolver({}))

    return return_circuits, return_resolvers


def random_symbol_circuit_resolver_batch(qubits,
                                         symbols,
                                         batch_size,
                                         n_moments=15,
                                         p=0.9,
                                         include_scalars=True):
    """Generate a batch of random circuits and resolvers."""
    return_circuits = []
    return_resolvers = []
    for _ in range(batch_size):
        return_circuits.append(
            random_symbol_circuit(qubits, symbols, n_moments, p,
                                  include_scalars))

        return_resolvers.append(
            cirq.ParamResolver(
                {symbol: np.random.random() for symbol in symbols}))

    return return_circuits, return_resolvers


def random_pauli_sums(qubits, max_sum_length, n_sums):
    """Generate a list of random cirq pauli sums of length |n_sums|."""
    sums = []
    paulis = [cirq.I, cirq.X, cirq.Y, cirq.Z]
    for _ in range(n_sums):
        this_sum_length = np.random.randint(1, max_sum_length + 1)
        terms = []
        for _ in range(this_sum_length):
            term_length = np.random.randint(1, len(qubits) + 1)
            this_term_qubits = random.sample(qubits, term_length)
            this_term_paulis = \
                [random.sample(paulis,1)[0] for _ in range(term_length)]
            terms.append(
                cirq.PauliString(dict(zip(this_term_qubits, this_term_paulis))))
        sums.append(cirq.PauliSum.from_pauli_strings(terms))
    return sums


# There are no native convertible ops inside of this function.
@tf.autograph.experimental.do_not_convert
def convert_to_tensor(items_to_convert):
    """Convert lists of tfq supported primitives to tensor representations.

    Recursively convert a nested lists of `cirq.PauliSum` or `cirq.Circuit`
    objects to a `tf.Tensor` representation. Note that cirq serialization only
    supports `cirq.GridQubit`s so we also require that input circuits and
    pauli sums are defined only on `cirq.GridQubit`s.


    >>> my_qubits = cirq.GridQubit.rect(1, 2)
    >>> my_circuits = [cirq.Circuit(cirq.X(my_qubits[0])),
    ...                cirq.Circuit(cirq.Z(my_qubits[0]))
    ... ]
    >>> tensor_input = tfq.convert_to_tensor(my_circuits)
    >>> # Now tensor_input can be used as model input etc.
    >>> same_circuits = tfq.from_tensor(tensor_input)
    >>> # same_circuits now holds cirq.Circuit objects once more.
    >>> same_circuits
    [cirq.Circuit([
        cirq.Moment(operations=[
            cirq.X.on(cirq.GridQubit(0, 0)),
        ]),
    ])
     cirq.Circuit([
        cirq.Moment(operations=[
            cirq.Z.on(cirq.GridQubit(0, 0)),
        ]),
    ])]

    Args:
        items_to_convert: Python `list` or nested `list` of `cirq.Circuit`
            or `cirq.Paulisum` objects. Should be rectangular, or this function
            will error.

    Returns:
        `tf.Tensor` that represents the input items.
    """

    # We use recursion here because np.ndenumerate tries to loop over
    # `cirq.Circuit`s and `cirq.PauliSum`s (they are iterable).
    # This code is safe for nested lists of depth less than the recursion limit,
    # which is deeper than any practical use the author can think of.
    def recur(items_to_convert, curr_type=None):
        tensored_items = []
        for item in items_to_convert:
            if isinstance(item, (list, np.ndarray, tuple)):
                tensored_items.append(recur(item, curr_type))
            elif isinstance(item, (cirq.PauliSum, cirq.PauliString)) and\
                    not curr_type == cirq.Circuit:
                curr_type = cirq.PauliSum
                tensored_items.append(
                    serializer.serialize_paulisum(item).SerializeToString())
            elif isinstance(item, cirq.Circuit) and\
                    not curr_type == cirq.PauliSum:
                curr_type = cirq.Circuit
                tensored_items.append(
                    serializer.serialize_circuit(item).SerializeToString())
            else:
                raise TypeError("Incompatible item passed into "
                                " convert_to_tensor. Tensor detected type: {}."
                                " got: {}".format(curr_type, type(item)))
        return tensored_items

    # This will catch impossible dimensions
    return tf.convert_to_tensor(recur(items_to_convert))


def _parse_single(item):
    try:
        if b'tfq_gate_set' in item:
            # Return a circuit parsing
            obj = cirq.google.api.v2.program_pb2.Program()
            obj.ParseFromString(item)
            out = serializer.deserialize_circuit(obj)
            return out

        # Return a PauliSum parsing.
        obj = pauli_sum_pb2.PauliSum()
        obj.ParseFromString(item)
        out = serializer.deserialize_paulisum(obj)
        return out
    except Exception:
        raise TypeError('Error decoding item: ' + str(item))


def from_tensor(tensor_to_convert):
    """Convert a tensor of tfq primitives back to Python objects.

    Convert a tensor representing `cirq.PauliSum` or `cirq.Circuit`
    objects back to Python objects.


    >>> my_qubits = cirq.GridQubit.rect(1, 2)
    >>> my_circuits = [cirq.Circuit(cirq.X(my_qubits[0])),
    ...                cirq.Circuit(cirq.Z(my_qubits[0]))
    ... ]
    >>> tensor_input = tfq.convert_to_tensor(my_circuits)
    >>> # Now tensor_input can be used as model input etc.
    >>> same_circuits = tfq.from_tensor(tensor_input)
    >>> # same_circuits now holds cirq.Circuit objects once more.
    >>> same_circuits
    [cirq.Circuit([
        cirq.Moment(operations=[
            cirq.X.on(cirq.GridQubit(0, 0)),
        ]),
    ])
     cirq.Circuit([
        cirq.Moment(operations=[
            cirq.Z.on(cirq.GridQubit(0, 0)),
        ]),
    ])]

    Args:
        tensor_to_convert: `tf.Tensor` or `np.ndarray` representation to
            convert back into python objects.

    Returns:
        Python `list` of items converted to their python representation stored
            in a (potentially nested) `list`.
    """
    if isinstance(tensor_to_convert, tf.Tensor):
        tensor_to_convert = tensor_to_convert.numpy()
    if not isinstance(tensor_to_convert, (np.ndarray, list, tuple)):
        raise TypeError("tensor_to_convert received bad "
                        "type {}".format(type(tensor_to_convert)))
    tensor_to_convert = np.array(tensor_to_convert)
    python_items = np.empty(tensor_to_convert.shape, dtype=object)
    curr_type = None
    for index, item in np.ndenumerate(tensor_to_convert):
        found_item = _parse_single(item)
        got_type = type(found_item)
        if (curr_type is not None) and (not got_type == curr_type):
            raise TypeError("from_tensor expected to find a tensor containing"
                            " elements of a single type.")
        curr_type = got_type
        python_items[index] = found_item
    return python_items


def kwargs_cartesian_product(**kwargs):
    """Compute the cartesian product of inputs yielding Python `dict`s.

    Note that all kwargs must provide `iterable` values. Useful for testing
    purposes.

    ```python
    a = {'one': [1,2,3], 'two': [4,5]}
    result = list(kwargs_cartesian_product(**a))

    # Result now contains:
    # [{'one': 1, 'two': 4},
    #  {'one': 1, 'two': 5},
    #  {'one': 2, 'two': 4},
    #  {'one': 2, 'two': 5},
    #  {'one': 3, 'two': 4},
    #  {'one': 3, 'two': 5}]
    ```

    Returns:
        Python `generator` of the cartesian product of the inputs `kwargs`.
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for (k, v) in zip(keys, vals):
        # Only reliable way to check for __iter__ and __getitem__
        try:
            _ = iter(v)
        except TypeError:
            raise ValueError(f'Value for argument {k} is not iterable.'
                             f' Got {v}.')

    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def _symbols_in_op(op):
    """Returns the set of symbols in a parameterized gate."""
    if isinstance(op, cirq.EigenGate):
        return op.exponent.free_symbols

    if isinstance(op, cirq.FSimGate):
        ret = set()
        if isinstance(op.theta, sympy.Basic):
            ret |= op.theta.free_symbols
        if isinstance(op.phi, sympy.Basic):
            ret |= op.phi.free_symbols
        return ret

    if isinstance(op, cirq.PhasedXPowGate):
        ret = set()
        if isinstance(op.exponent, sympy.Basic):
            ret |= op.exponent.free_symbols
        if isinstance(op.phase_exponent, sympy.Basic):
            ret |= op.phase_exponent.free_symbols
        return ret

    raise ValueError("Attempted to scan for symbols in circuit with unsupported"
                     " ops inside. Expected op found in tfq.get_supported_gates"
                     " but found: ".format(str(op)))


def get_circuit_symbols(circuit):
    """Returns a list of the sympy.Symbols that are present in `circuit`.

    Args:
        circuit: A `cirq.Circuit` object.

    Returns:
        Python `list` containing the symbols found in the circuit.
    """
    all_symbols = set()
    for moment in circuit:
        for op in moment:
            if cirq.is_parameterized(op):
                all_symbols |= _symbols_in_op(op.gate)
    return [str(x) for x in all_symbols]


def _many_clifford_to_many_z(pauli_sum):
    """Convert many clifford to many Z.
    Returns the gate set required for transforming an arbitrary tensor product
    of paulis into a product of all pauli -Z's.
    Args:
        pauli_sum: `cirq.PauliSum` object to be converted to all z's.
    Returns:
        gate_list: List of required gates to complete the transformation.
        conjugate_list: List of gates, but reversed and complex conjugate
            applied to each rotation gate
    """
    # TODO(jaeyoo): investigate how to apply cirq.PauliString.to_z_basis_ops
    gate_list = []
    # Hermitian conjugate
    conjugate_list = []
    for qubit, pauli in pauli_sum.items():
        if isinstance(pauli, cirq.ZPowGate):
            continue
        elif isinstance(pauli, cirq.XPowGate):
            gate_list.append(cirq.H(qubit))
            conjugate_list.append(cirq.H(qubit))
        elif isinstance(pauli, cirq.YPowGate):
            # It is identical to the conjugate of Phase and Hadamard gate up to
            # global phase. This global phase difference is gone with
            # multiplication of hermition conjugate later.
            gate_list.append(cirq.rx(np.pi / 2)(qubit))
            conjugate_list.append(cirq.rx(-np.pi / 2)(qubit))
    return gate_list, conjugate_list[::-1]


def _many_z_to_single_z(focal_qubit, pauli_sum):
    """Convert many Z's to single Z.
    Returns the gate set required for transforming an arbitrary tensor product
    of pauli-z's into a product of all identites and a single pauli-Z.
    Args:
        focal_qubit: central qubit among CNOT gates.
        pauli_sum: `cirq.PauliSum` object to be converted to CNOT's and Z.
    Returns:
        gate_list: List of the required CNOT gates for this conversion.
        gate_list_reversed: List of the same CNOT gates, but in reverse.
    """
    gate_list = []
    for q in pauli_sum.qubits:
        if q != focal_qubit:
            gate_list.append(cirq.CNOT(q, focal_qubit))
    return gate_list, gate_list[::-1]


def check_commutability(pauli_sum):
    """Return False if at least one pair of terms in pauli_sum is not
    commutable.

    Args:
        pauli_sum: `cirq.PauliSum` object to be checked if all of terms inside
            are commutable each other.
    """
    for term1 in pauli_sum:
        for term2 in pauli_sum:
            if not cirq.commutes(term1, term2):
                raise ValueError("Given an operator has non-commutable "
                                 "terms, whose exponentiation is not "
                                 "supported yet: {} and {}".format(
                                     term1, term2))


def exp_identity(param, c, zeroth_qubit):
    """Return a circuit for exponentiating an identity gate."""
    # TODO(jaeyoo): Reduce the number of gates for this decomposition.
    phase_shift = cirq.ZPowGate(exponent=-param * c / np.pi).on(zeroth_qubit)
    exp_circuit = cirq.Circuit(
        [cirq.X(zeroth_qubit), phase_shift,
         cirq.X(zeroth_qubit), phase_shift])
    return exp_circuit


def exponential(operators, coefficients=None):
    """Return a Cirq circuit with exponential forms of operators.

    Construct an exponential form of given `operators` and `coefficients`.
    Operators to be exponentiated are specified in `operators` as
    `cirq.PauliSum` or `cirq.PauliString`. Parameters are given by
    `coefficients`.

    Note that only operators whose standard representations consist of terms
    which all commute can be exponentiated.  This allows use of the identity
    exp(A+B+...) = exp(A)exp(B)... else there would need to be automatic
    handling of Trotterization and convergence, which is not supported yet.

    Args:
        operators: Python `list` or `tuple` of `cirq.PauliSum` or
            `cirq.PauliString` objects to be exponentiated.
            Here are simple examples.
            Let q = cirq.GridQubit(0, 0)
            E.g. operator = 0.5 * X(q) -> exp(-i * 0.5 * X(q))
                 operator = 0.5 * cirq.PauliString({q: cirq.I})
                           -> exp(-i * 0.5)*np.eye(2)
            Be careful of the negation and the PauliString of the identity gate.
        coefficients: (Optional) Python `list` of Python `str`, `float` or
            `sympy.Symbol` object of parameters. Defaults to None, then all
            coefficients of `operators` are set to 1.0.
    Returns:
        A `cirq.Circuit` containing exponential form of given `operators`
            and `coefficients`.
    """
    # Ingest operators.
    if not isinstance(operators, (list, tuple)):
        raise TypeError("operators is not a list of operators.")

    if not all(
            isinstance(x, (cirq.PauliSum, cirq.PauliString))
            for x in operators):
        raise TypeError("Each element in operators must be a "
                        "cirq.PauliSum or cirq.PauliString object.")

    # Ingest coefficients.
    if coefficients is None:
        coefficients = [1.0 for _ in operators]

    if not isinstance(coefficients, (list, tuple, np.ndarray)):
        raise TypeError("coefficients is not a list of coefficients.")

    if not all(isinstance(x, (str, sympy.Symbol, float)) for x in coefficients):
        raise TypeError("Each element in coefficients"
                        " must be a float or a string or sympy.Symbol.")

    if len(coefficients) != len(operators):
        raise ValueError("the number of operators should be the same as that "
                         "of coefficients. Got {} operators and {} coefficients"
                         "".format(len(operators), len(coefficients)))

    coefficients = [
        sympy.Symbol(s) if isinstance(s, str) else s
        for i, s in enumerate(coefficients)
    ]

    circuit = cirq.Circuit()

    operators = [
        cirq.PauliSum.from_pauli_strings(ps) if isinstance(
            ps, cirq.PauliString) else ps for ps in operators
    ]

    qubit_set = {q for psum in operators for q in psum.qubits}
    identity_ref_qubit = cirq.GridQubit(0, 0)
    if len(qubit_set) > 0:
        identity_ref_qubit = sorted(list(qubit_set))[0]

    for param, pauli_sum in zip(coefficients, operators):
        if isinstance(pauli_sum, cirq.PauliSum):
            check_commutability(pauli_sum)
        for op in pauli_sum:
            if abs(op.coefficient.imag) > 1e-9:
                raise TypeError('exponential only supports real '
                                'coefficients: got '
                                '{}'.format(op.coefficient))
            # Create a circuit with exponentiating `op` with param
            c = op.coefficient.real
            if len(op.gate.pauli_mask) == 0:
                # If given gate_op is identity.
                circuit += exp_identity(param, c, identity_ref_qubit)
                continue

            # Where to perform the Rz gate based on difficulty of CNOT's
            # TODO(jaeyoo): will write a super duper optimization on this.
            #  currently going on HIGHEST-indexed qubit.
            k = op.qubits[-1]
            # Set of gates to convert all X's and Y's -> Z's.
            u, u_dagger = _many_clifford_to_many_z(op)

            # Set of gates to convert many Z's into a single Z using CNOTs.
            w, w_dagger = _many_z_to_single_z(k, op)

            # cirq.rz(2*theta) = exp(-i*0.5*(2*theta)*Z) == exp(-i*theta*Z)
            # focal point of the CNOT ladder.
            exp_circuit = u + w + [cirq.rz(2 * param * c)(k)
                                  ] + w_dagger + u_dagger
            circuit += cirq.Circuit(exp_circuit)
    return circuit
