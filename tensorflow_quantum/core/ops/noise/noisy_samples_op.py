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
"""Module for high performance noisy circuit sampling ops"""
import os
import tensorflow as tf
from tensorflow_quantum.core.ops import tfq_utility_ops
from tensorflow_quantum.core.ops.load_module import load_module

NOISY_OP_MODULE = load_module(os.path.join("noise", "_tfq_noise_ops.so"))


def samples(programs, symbol_names, symbol_values, num_samples):
    """Generate samples using the C++ noisy trajectory simulator.

    Simulate the final state of `programs` given `symbol_values` are placed
    inside of the symbols with the name in `symbol_names` in each circuit.
    Channels in this simulation will be "tossed" to a certain realization
    during simulation. After each simulation is a run a single bitstring
    will be drawn. These simulations are repeated `num_samples` times.


    >>> # Sample a noisy circuit with C++.
    >>> qubit = cirq.GridQubit(0, 0)
    >>> my_symbol = sympy.Symbol('alpha')
    >>> my_circuit_tensor = tfq.convert_to_tensor([
    ...     cirq.Circuit(
    ...         cirq.X(qubit) ** my_symbol,
    ...         cirq.depolarize(0.01)(qubit)
    ...     )
    ... ])
    >>> my_values = np.array([[0.123]])
    >>> my_num_samples = np.array([100])
    >>> # This op can now be run with:
    >>> output = tfq.noise.samples(
    ...     my_circuit_tensor, ['alpha'], my_values, my_num_samples)
    >>> output
    <tf.RaggedTensor [[[0], [0], [1], [0], [0], [0], [0], [1], [0], [0]]]>


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
        num_samples: `tf.Tensor` with one element indicating the number of
            samples to draw for all circuits in the batch.
    Returns:
        A `tf.Tensor` containing the samples taken from each circuit in
        `programs`.
    """
    padded_samples = NOISY_OP_MODULE.tfq_noisy_samples(
        programs, symbol_names, tf.cast(symbol_values, tf.float32), num_samples)
    return tfq_utility_ops.padded_to_ragged(padded_samples)