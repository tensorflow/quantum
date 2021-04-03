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
"""Module for high performance noisy circuit sampled epxectation ops."""
import os
import tensorflow as tf
from tensorflow_quantum.core.ops.load_module import load_module

NOISY_OP_MODULE = load_module(os.path.join("noise", "_tfq_noise_ops.so"))


def sampled_expectation(programs, symbol_names, symbol_values, pauli_sums,
                        num_samples):
    """Estimates (via sampling) expectation values using monte-carlo simulation.

    Simulate the final state of `programs` given `symbol_values` are placed
    inside of the symbols with the name in `symbol_names` in each circuit.
    Channels in this simulation will be "tossed" to a certain realization
    during simulation. This simulation is repeated `num_samples` times and
    bitstring based expectation calculations with the given `pauli_sums` are
    calculated after each run. Once all the runs are finished, these quantities
    are averaged together.


    >>> # Prepare some inputs.
    >>> qubit = cirq.GridQubit(0, 0)
    >>> my_symbol = sympy.Symbol('alpha')
    >>> my_circuit_tensor = tfq.convert_to_tensor([
    ...     cirq.Circuit(
    ...         cirq.H(qubit) ** my_symbol,
    ...         cirq.depolarize(0.01)(qubit)
    ...     )
    ... ])
    >>> my_values = np.array([[0.123]])
    >>> my_paulis = tfq.convert_to_tensor([[
    ...     3.5 * cirq.X(qubit) - 2.2 * cirq.Y(qubit)
    ... ]])
    >>> my_num_samples = np.array([[100]])
    >>> # This op can now be run with:
    >>> output = tfq.noise.sampled_expectation(
    ...     my_circuit_tensor, ['alpha'], my_values, my_paulis, my_num_samples)
    >>> output
    tf.Tensor([[0.71530885]], shape=(1, 1), dtype=float32)


    In order to make the op differentiable, a `tfq.differentiator` object is
    needed. see `tfq.differentiators` for more details. Below is a simple
    example of how to make the from the above code block differentiable:


    >>> diff = tfq.differentiators.ForwardDifference()
    >>> my_differentiable_op = diff.generate_differentiable_op(
    ...     sampled_op=tfq.noise.sampled_expectation
    ... )


    Args:
        programs: `tf.Tensor` of strings with shape [batch_size] containing
            the string representations of the circuits to be executed.
        symbol_names: `tf.Tensor` of strings with shape [n_params], which
            is used to specify the order in which the values in
            `symbol_values` should be placed inside of the circuits in
            `programs`.
        symbol_values: `tf.Tensor` of real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits specificed by programs, following the ordering
            dictated by `symbol_names`.
        pauli_sums: `tf.Tensor` of strings with shape [batch_size, n_ops]
            containing the string representation of the operators that will
            be used on all of the circuits in the expectation calculations.
        num_samples: `tf.Tensor` with `num_samples[i][j]` is equal to the
            number of times `programs[i]` will be simulated to estimate
            `pauli_sums[i][j]`. Therefore, `num_samples` must have the same
            shape as `pauli_sums`. Note: internally this quantity can get
            rounded up to the nearest multiple of the number of available
            threads to TensorFlow. For best performance ensure that the
            quantities in `num_samples` are a multiple of the number of
            available threads.
    Returns:
        `tf.Tensor` with shape [batch_size, n_ops] that holds the
            expectation value for each circuit with each op applied to it
            (after resolving the corresponding parameters in).
    """
    return NOISY_OP_MODULE.tfq_noisy_sampled_expectation(
        programs, symbol_names, tf.cast(symbol_values, tf.float32), pauli_sums,
        tf.cast(num_samples, dtype=tf.int32))
