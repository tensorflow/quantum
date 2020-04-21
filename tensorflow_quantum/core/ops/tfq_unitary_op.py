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
import tensorflow as tf
from tensorflow_quantum.core.ops import tfq_utility_ops
from tensorflow_quantum.core.ops.load_module import load_module

OP_MODULE = load_module("_tfq_calculate_unitary_op.so")


def calculate_unitary(programs, symbol_names, symbol_values):
    """Calculate the unitary matrix for the given circuits.


    >>> qubit = cirq.GridQubit(0, 0)
    >>> symbol = sympy.Symbol('alpha')
    >>> my_circuit = cirq.Circuit(cirq.H(qubit) ** symbol)
    >>> tensor_circuit = tfq.convert_to_tensor([my_circuit])
    >>> tfq.calculate_unitary(tensor_circuit, ['alpha'], [[0.2]])
    <tf.RaggedTensor [
        [[(0.9720+0.0860j), (0.0675-0.2078j)],
         [(0.0675-0.2078j), (0.8369+0.5017j)]]]>


    Args:
        programs: `tf.Tensor` of strings with shape [batch_size] containing
            the string representations of the circuits to be executed.
        symbol_names: `tf.Tensor` of strings with shape [n_params], which
            is used to specify the order in which the values in
            `symbol_values` should be placed inside of the circuits in
            `programs`.
        symbol_values: `tf.Tensor` of real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits specified by programs, following the ordering
            dictated by `symbol_names`.

    Returns:
        `tf.RaggedTensor` with shape
            [batch_size, 2**n_qubits[i], 2**n_qubits[i]] where n_qubits[i] is
            the number of qubits for program `i` in `programs`. Each entry
            corresponds to the unitary matrix that circuit enacts.
    """
    return tfq_utility_ops.padded_to_ragged2d(
        OP_MODULE.tfq_calculate_unitary(programs, symbol_names,
                                        tf.cast(symbol_values, tf.float32)))
