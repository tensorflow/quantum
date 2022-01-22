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
"""Module to register MPS simulation ops."""
import os
import tensorflow as tf
from tensorflow_quantum.core.ops.load_module import load_module
from tensorflow_quantum.core.ops import tfq_utility_ops

MATH_OP_MODULE = load_module(os.path.join("math_ops", "_tfq_math_ops.so"))


def mps_1d_expectation(programs,
                       symbol_names,
                       symbol_values,
                       pauli_sums,
                       bond_dim=4):
    """Calculate the expectation value of circuits wrt some operator(s)

    Simulate the final state of `programs` given `symbol_values` are placed
    inside of the symbols with the name in `symbol_names` in each circuit.
    From there we will then compute the expectation values of `pauli_sums`
    on the final states. Note that this op requires 1D non periodic circuits.

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
        bond_dim: Integer value used for the bond dimension during simulation.

    Returns:
        `tf.Tensor` with shape [batch_size, n_ops] that holds the
            expectation value for each circuit with each op applied to it
            (after resolving the corresponding parameters in).
    """
    return MATH_OP_MODULE.tfq_simulate_mps1d_expectation(programs,
                                                         symbol_names,
                                                         tf.cast(
                                                             symbol_values,
                                                             tf.float32),
                                                         pauli_sums,
                                                         bond_dim=bond_dim)


def mps_1d_sample(programs,
                  symbol_names,
                  symbol_values,
                  num_samples,
                  bond_dim=4):
    """Generate samples using the C++ MPS simulator.

    Simulate the final state of `programs` given `symbol_values` are placed
    inside of the symbols with the name in `symbol_names` in each circuit.
    From there we will then sample from the final state. Note that this op
    requires 1D non periodic circuits.

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
            samples to draw.
        bond_dim: Integer value used for the bond dimension during simulation.

    Returns:
        A `tf.RaggedTensor` containing the samples taken from each circuit in
        `programs`.
    """
    padded_samples = MATH_OP_MODULE.tfq_simulate_mps1d_samples(
        programs,
        symbol_names,
        tf.cast(symbol_values, tf.float32),
        num_samples,
        bond_dim=bond_dim)

    return tfq_utility_ops.padded_to_ragged(padded_samples)


def mps_1d_sampled_expectation(programs,
                               symbol_names,
                               symbol_values,
                               pauli_sums,
                               num_samples,
                               bond_dim=4):
    """Calculate the expectation value of circuits using samples.

    Simulate the final state of `programs` given `symbol_values` are placed
    inside of the symbols with the name in `symbol_names` in each circuit.
    Them, sample the resulting state `num_samples` times and use these samples
    to compute expectation values of the given `pauli_sums`. Note that this op
    requires 1D non periodic circuits.

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
            number of samples to draw in each term of `pauli_sums[i][j]`
            when estimating the expectation. Therefore, `num_samples` must
            have the same shape as `pauli_sums`.
        bond_dim: Integer value used for the bond dimension during simulation.

    Returns:
        `tf.Tensor` with shape [batch_size, n_ops] that holds the
            expectation value for each circuit with each op applied to it
            (after resolving the corresponding parameters in).
    """
    return MATH_OP_MODULE.tfq_simulate_mps1d_sampled_expectation(
        programs,
        symbol_names,
        tf.cast(symbol_values, tf.float32),
        pauli_sums,
        tf.cast(num_samples, dtype=tf.int32),
        bond_dim=bond_dim)
