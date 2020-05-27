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
"""Layers for constructing quantum integer circuits on qubit backends."""
import numbers
import cirq

from tensorflow_quantum.python import util


def projector_on_one(qubit):
    """Returns the projector on 1 for the given qubit.

    Given a qubit k, the projector onto one can be represented by
        |1><1|_k = 0.5(I_k - Z_k).

    Args:
        qubit: A `cirq.GridQubit` on which the projector is supported.

    Returns:
        `cirq.PauliSum` representing the projector.
    """
    if not isinstance(qubit, cirq.GridQubit):
        raise TypeError("A projector must live on a cirq.GridQubit.")
    return 0.5 * cirq.I(qubit) - 0.5 * cirq.Z(qubit)


def integer_operator(qubits):
    """Returns operator representing position in binary on a qubit register.

    Unsigned integers on computers can be represented as a bitstring.  For an
    integer represented by N bits, the k-th bit represents the presence
    of the number 2**(N - k - 1) in the sum representing the integer.
    Similarly, we can define a binary operator J as
        J = \sum_P{k=0}^{N-1} 2^{N-k-1}|1><1|_k,
    where
        |1><1|_k = 0.5(I_k - Z_k).
    J can be represented by a `cirq.PauliSum`.

    Args:
        qubits: Python `list` of `GridQubit`s on which the operator is
            supported.

    Returns:
        int_op: A `cirq.PauliSum` representing the integer operator.
    """
    if not isinstance(qubits, list):
        raise TypeError("Argument qubits must be a list of cirq.GridQubits.")
    int_op = cirq.PauliSum()
    width = len(qubits)
    for loc, q in enumerate(qubits):
        int_op += 2**(width - 1 - loc) * projector_on_one(q)
    return int_op


def registers_from_precisions(precisions):
    """Returns list of cirq.GridQubit registers for the given precisions.

    Args:
        precisions: a Python `list` of `int`s.  Entry `precisions[i]` sets
            the number of qubits on which quantum integer `i` is supported.

    Returns:
        register_list: lists of `cirq.GridQubit`s, such that
            len(register_list[i]) == precisions[i] and all entries are unique.
    """
    if not isinstance(precisions, list):
        raise TypeError("Argument qubits must be a list of cirq.GridQubits.")
    register_list = []
    for r, width in enumerate(precisions):
        this_register = []
        for col in range(width):
            this_register.append(cirq.GridQubit(r, col))
        register_list.append(this_register)
    return register_list


def build_cliques_psum(precisions, cliques):
    """Returns the cirq.PauliSum corresponding to the given cliques.

    For example, the precisions list [3, 3] and the cliques dict
    {(0,): 2, (1,): 3, (0, 1): 6} corresponds to a ferromagnetic interaction
    Hamiltonian between two three-qubit quantum integer registers Ja and Jb,
    H = 2*Ja + 3*Jb + 6*Ja*Jb

    Args:
        precisions: a Python `list` of `int`s.  Entry precisions[i] sets
            the number of qubits on which quantum integer `i` is supported.
        cliques: a Python `dict` mapping tuples of quantum integer register
            labels to the weight of their product.

    Returns:
        cliques_psum: `cirq.PauliSum` representation of the Hamiltonian
            corresponding to the given precisions and clique weights.
    """
    if not isinstance(precisions, list):
        raise TypeError("`precisions` must be a Python list.")
    for p in precisions:
        if not isinstance(p, numbers.Integral) or p < 1:
            raise TypeError("Each entry in precisions must be an integer"
                            " greater than zero, got {}.".format(p))
    if not isinstance(cliques, dict):
        raise TypeError("`cliques` must be a Python dict.")
    for key, value in cliques.items():
        if not isinstance(key, tuple):
            raise TypeError("Each key in cliques must be a (possibly empty)"
                            " tuple of register labels, got {}."
                            " Check that no key has the form `(n)`."
                            " Instead use `(n,)` else the key will be"
                            " parsed as an integer.".format(key))
        for entry in key:
            if not isinstance(entry, numbers.Integral) or entry < 0:
                raise TypeError("Each entry of each key in cliques must be a"
                                " non-negative integer, got {}.".format(entry))
            if entry > (len(precisions) - 1):
                raise ValueError("Cannot access requested register {0} as there"
                                 " are only {1} registers available.".format(
                                     entry, len(precisions)))
        if not isinstance(value, numbers.Real):
            raise TypeError("Each value in cliques must be a real number,"
                            " got {}.".format(value))
    register_list = registers_from_precisions(precisions)
    op_list = [integer_operator(register) for register in register_list]
    cliques_psum = cirq.PauliSum()
    for clique in cliques:
        if clique:
            this_psum = cirq.PauliString(cirq.I(register_list[clique[0]][0]))
            for i in clique:
                this_psum *= op_list[i]
            this_psum *= cliques[clique]
        else:
            # Empty label tuple is a constant offset.
            this_psum = cliques[clique] * cirq.PauliString(
                cirq.I(register_list[0][0]))
        cliques_psum += this_psum
    return cliques_psum


def build_cliques_exp(precisions, cliques, exp_coeff=None):
    """Builds circuit representing the exponential of quantum integer cliques.

    Suppose we wish to specify the exponential exp(-i*s*(J0^2 + J0*J1 + J1^2)),
    where Ji is the integer operator on register i.  This exponential is
    specified by a precisions argument containing the number of qubits to use
    for each register, a cliques argument specifying the operator sum, and the
    exp_coeff argument specifying the prefactor of the operator sum:


    >>> precisions = [3, 3]
    >>> cliques = {(0, 0): 1, (0, 1): 1, (1, 1): 1}
    >>> symbol = sympy.Symbol("theta")
    >>> exp_circuit = build_cliques_exp(precisions, cliques, symbol)


    Args:
        precisions: a Python `list` of `int`s.  Entry precisions[i] sets
            the number of qubits on which quantum integer `i` is supported.
        cliques: a Python `dict` mapping tuples of quantum integer register
            labels to the weight of their product.
        exp_coeff: A Python `str`, `sympy.Symbol`, or float, which will be
            the coefficient of the cost Hamiltonian in the exponential.
            If None, the value 1.0 will be used.

    Returns:
        `cirq.Circuit` which is the exponential of the cliques on registers
            of the specified sizes.
    """
    cliques_psum = build_cliques_psum(precisions, cliques)
    if exp_coeff is None:
        exp_coeff = 1.0
    return util.exponential([cliques_psum], coefficients=[exp_coeff])


def build_momenta_exp(precisions, cliques, exp_coeff=None):
    """Builds circuit representing the exponential of quantum integer momenta.

    Suppose we wish to specify the exponential exp(-i*s*(K0^2 + K0*K1 + K1^2)),
    where Ki is the generator of the shift operator on the quantum integer on
    register i.  This exponential is specified by a precisions argument
    containing the number of qubits to use for each register, a cliques argument
    specifying the operator sum, and the exp_coeff argument specifying the
    prefactor of the operator sum:


    >>> precisions = [3, 3]
    >>> cliques = {(0, 0): 1, (0, 1): 1, (1, 1): 1}
    >>> symbol = sympy.Symbol("theta")
    >>> exp_layer = build_momenta_exp(precisions, cliques, symbol)


    Args:
        precisions: a Python `list` of `int`s.  Entry precisions[i] sets
            the number of qubits on which quantum integer `i` is supported.
        cliques: a Python `dict` mapping tuples of quantum integer register
            labels to the weight of their product.
        exp_coeff: A Python `str`, `sympy.Symbol`, or float, which will be
            the coefficient of the momentum Hamiltonian in the exponential.
            If None, the value 1.0 will be used.
    """
    cliques_psum = build_cliques_psum(precisions, cliques)
    if exp_coeff is None:
        exp_coeff = 1.0
    exp_circuit = util.exponential([cliques_psum], coefficients=[exp_coeff])
    transform = cirq.Circuit()
    convert = cirq.ConvertToCzAndSingleGates(allow_partial_czs=True)
    for r in registers_from_precisions(precisions):
        this_transform = cirq.Circuit(cirq.QFT(*r))
        convert(this_transform)
        transform += this_transform
    return transform + exp_circuit + transform**(-1)
