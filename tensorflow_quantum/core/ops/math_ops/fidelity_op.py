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
"""Module for tfq.math.fidelity op."""
import tensorflow as tf
from tensorflow_quantum.core.ops.math_ops import inner_product_op


@tf.function
@tf.custom_gradient
def fidelity(programs, symbol_names, symbol_values, other_programs):
    """Calculate the fidelity between circuits.

    Compute (potentially many) fidelities between the given circuits and
    the symbol free comparison circuits.

    Calculates out[i][j] = $ | \langle \psi_{\text{programs[i]}} \\
     (\text{symbol\_values[i]}) | \psi_{\text{other\_programs[j]}} \rangle \\
     |^2 $


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
    >>> fid = tfq.math.fidelity(reference_tensor, symbol_tensor,
    ...                             values_tensor, other_tensor)
    >>> fid
    tf.Tensor(
        [[ 0., 0.925, 0.25],
         [ 0., 0.036, 0.25]],shape=(2, 3), dtype=float32)



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
            to the fidelity of `programs[i]` with `symbol_values[i]`
            resolved in and `other_programs[i][j]`.
    """
    f32_vals = tf.cast(symbol_values, tf.float32)
    ip = inner_product_op.inner_product(programs, symbol_names, f32_vals,
                                        other_programs)

    def grad(dy):
        ret_zero = tf.equal(tf.size(symbol_names), 0)
        inner_prod_grad = tf.cond(
            ret_zero, lambda: tf.zeros_like(symbol_values, dtype=tf.float32),
            lambda: tf.math.real(2. * ip * inner_product_op._inner_product_grad(
                programs, symbol_names, symbol_values, other_programs, dy)))
        return [None, None, inner_prod_grad, None]

    return tf.math.abs(ip)**2, grad
