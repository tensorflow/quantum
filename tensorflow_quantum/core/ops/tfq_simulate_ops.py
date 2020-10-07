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
from tensorflow_quantum.core.ops.load_module import load_module

SIM_OP_MODULE = load_module("_tfq_simulate_ops.so")


def tfq_simulate_expectation(programs, symbol_names, symbol_values, pauli_sums):
    """Calculate the expectation value of circuits wrt some operator(s)

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
    Returns:
        `tf.Tensor` with shape [batch_size, n_ops] that holds the
            expectation value for each circuit with each op applied to it
            (after resolving the corresponding parameters in).
    """
    return SIM_OP_MODULE.tfq_simulate_expectation(
        programs, symbol_names, tf.cast(symbol_values, tf.float32), pauli_sums)


def tfq_simulate_state(programs, symbol_names, symbol_values):
    """Returns the state of the programs using the C++ state vector simulator.

    Simulate the final state of `programs` given `symbol_values` are placed
    inside of the symbols with the name in `symbol_names` in each circuit.

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
    Returns:
        A `tf.Tensor` containing the final state of each circuit in `programs`.
    """
    return SIM_OP_MODULE.tfq_simulate_state(programs, symbol_names,
                                            tf.cast(symbol_values, tf.float32))


def tfq_simulate_samples(programs, symbol_names, symbol_values, num_samples):
    """Generate samples using the C++ state vector simulator.

    Simulate the final state of `programs` given `symbol_values` are placed
    inside of the symbols with the name in `symbol_names` in each circuit.
    From there we will then sample from the final state using native tensorflow
    operations.

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
    Returns:
        A `tf.Tensor` containing the samples taken from each circuit in
        `programs`.
    """
    return SIM_OP_MODULE.tfq_simulate_samples(
        programs, symbol_names, tf.cast(symbol_values, tf.float32), num_samples)


def tfq_simulate_sampled_expectation(programs, symbol_names, symbol_values,
                                     pauli_sums, num_samples):
    """Calculate the expectation value of circuits using samples.

    Simulate the final state of `programs` given `symbol_values` are placed
    inside of the symbols with the name in `symbol_names` in each circuit.
    Them, sample the resulting state `num_samples` times and use these samples
    to compute expectation values of the given `pauli_sums`.

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
    Returns:
        `tf.Tensor` with shape [batch_size, n_ops] that holds the
            expectation value for each circuit with each op applied to it
            (after resolving the corresponding parameters in).
    """
    return SIM_OP_MODULE.tfq_simulate_sampled_expectation(
        programs, symbol_names, tf.cast(symbol_values, tf.float32), pauli_sums,
        tf.cast(num_samples, dtype=tf.int32))
