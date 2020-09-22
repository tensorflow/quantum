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
"""Module to register python op gradient."""
import os
import tensorflow as tf
from tensorflow_quantum.core.ops.load_module import load_module

MATH_OP_MODULE = load_module(os.path.join("math_ops", "_tfq_math_ops.so"))


def inner_product(programs, symbol_names, symbol_values, other_programs):
    """Calculate the inner product between circuits.

    Compute (potentially many) inner products between the given circuits and
    the symbol free comparison circuits.

    Calculates out[i][j] = \langle \psi_{\text{programs[i]}} \\
        (\text{symbol_values[i]}) | \psi_{\text{other_programs[j]}} \rangle


    >>> symbols = sympy.symbols('alpha beta')
    >>> qubits = cirq.GridQubit.rect(1, 2)
    >>> reference_circuits = [
    ...     cirq.Circuit((cirq.H**symbols[0]).on_each(qubits)),
    ...     cirq.Circuit(
    ...         cirq.X(qubits[0]) ** symbols[0],
    ...         cirq.Y(qubits[1]) ** symbols[1])
    ... ]
    >>> other_circuits = [
    ...     cirq.Circuit(cirq.X.on_each(qubits)),
    ...     cirq.Circuit((cirq.Y**0.125).on_each(qubits)),
    ...     cirq.Circuit((cirq.X**0.5).on_each(qubits))
    ... ]
    >>> reference_tensor = tfq.convert_to_tensor(reference_circuits)
    >>> symbol_tensor = tf.convert_to_tensor(list(symbols))
    >>> values_tensor = tf.convert_to_tensor(np.arange(4).reshape(2, 2))
    >>> other_tensor = tfq.convert_to_tensor([other_circuits, other_circuits])
    >>> ip = tfq.math.inner_product(reference_tensor)
    >>> ip
    tf.Tensor(
        [[ 0+0.j, 8.8871640e-01+0.3681184j,
           0+0.5j],
         [ 0+0.j, 7.3223300e-02-0.17677669j,
           0-0.5j]],shape=(2, 3), dtype=complex64)



    Note: `other_programs` must not contain any free symbols. These can
        be resolved beforehand with `tfq.resolve_parameters`.

    Note: Currently this op is not differentiable.

    Args:
        programs: `tf.Tensor` of strings with shape [batch_size] containing
            the string representations of the circuits
        symbol_names: `tf.Tensor` of strings with shape [n_params], which
            is used to specify the order in which the values in
            `symbol_values` should be placed inside of the circuits in
            `programs`.
        symbol_values: `tf.Tensor` of real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits specificed by programs, following the ordering
            dictated by `symbol_names`.
        other_programs: `tf.Tensor` of strings with shape [batch_size, n_others]
            containing the string representations of the circuits with which to
            compute the overlap on `programs` with. Must not contain any free
            symbols.
    Returns:
        `tf.Tensor` with shape [batch_size, n_others] where `out[i][j]` is equal
            to the inner product of `programs[i]` with `symbol_values[i]`
            resolved in and `other_programs[i][j]`.

    """
    return MATH_OP_MODULE.tfq_inner_product(programs, symbol_names,
                                            tf.cast(symbol_values, tf.float32),
                                            other_programs)
