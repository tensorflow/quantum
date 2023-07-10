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


def _inner_product_grad(programs, symbol_names, symbol_values, other_programs,
                        prev_grad):
    """Calculate the adjoint gradients of the inner product between circuits.

    Compute the gradients of the (potentially many) inner products between
    the given circuits and the symbol free comparison circuits.

    Calculates out[i][j][k] = $ \frac{\langle \psi_{\text{programs[i]}} \\
        (\text{symbol_values[i]})}{\partial \text{symbol_names[k]}} | \\
        \psi_{\text{other_programs[j]}} \rangle $


    Note: `other_programs` must not contain any free symbols. These can
        be resolved beforehand with `tfq.resolve_parameters`.

    Note: len(symbol_names) (=n_params) should be a positive integer.

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
        prev_grad: `tf.Tensor` of real numbers with shape [batch_size, n_ops]
            backprop of values from downstream in the compute graph.

    Returns:
        tf.Tensor` with shape [batch_size, n_symbols] where `out[i][j]` is equal
        to the gradient of the inner product between programs[i] and all
        other_programs[i] w.r.t. `symbol_names[j]` and `programs[i]` is resolved
        with `symbol_values[i]`.
    """
    # Due to TF gradient scheme, we return complex conjugate derivative.
    return tf.math.conj(
        MATH_OP_MODULE.tfq_inner_product_grad(
            programs, symbol_names, tf.cast(symbol_values, tf.float32),
            other_programs, tf.cast(prev_grad, tf.float32)))


@tf.custom_gradient
def inner_product(programs, symbol_names, symbol_values, other_programs):
    """Calculate the inner product between circuits.

    Compute (potentially many) inner products between the given circuits and
    the symbol free comparison circuits.

    Calculates out[i][j] = $ \langle \psi_{\text{programs[i]}} \\
     (\text{symbol\_values[i]}) | \psi_{\text{other\_programs[j]}} \rangle $


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
    >>> symbol_tensor = tf.convert_to_tensor([s.name for s in symbols])
    >>> values_tensor = tf.convert_to_tensor(np.arange(4).reshape(2, 2))
    >>> other_tensor = tfq.convert_to_tensor([other_circuits, other_circuits])
    >>> ip = tfq.math.inner_product(reference_tensor, symbol_tensor,
    ...                             values_tensor, other_tensor)
    >>> ip
    tf.Tensor(
        [[ 0+0.j, 8.8871640e-01+0.3681184j,
           0+0.5j],
         [ 0+0.j, 7.3223300e-02-0.17677669j,
           0-0.5j]],shape=(2, 3), dtype=complex64)



    Note: `other_programs` must not contain any free symbols. These can
        be resolved beforehand with `tfq.resolve_parameters`.

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

    def grad(dy):

        def _true_grad():
            return _inner_product_grad(programs, symbol_names, symbol_values,
                                       other_programs, dy)

        ret_zero = tf.equal(tf.size(symbol_names), 0)
        inner_prod_grad = tf.cond(
            ret_zero, lambda: tf.zeros_like(symbol_values, dtype=tf.complex64),
            _true_grad)
        return [None, None, inner_prod_grad, None]

    return MATH_OP_MODULE.tfq_inner_product(programs, symbol_names,
                                            tf.cast(symbol_values, tf.float32),
                                            other_programs), grad
