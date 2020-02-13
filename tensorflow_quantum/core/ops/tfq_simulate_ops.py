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
    """Returns the state of the programs using the C++ wavefunction simulator.

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


@tf.function
def tfq_simulate_samples(programs, symbol_names, symbol_values, num_samples):
    """Generate samples using the C++ wavefunction simulator.

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
    # get the state from the simulator
    state = tfq_simulate_state(programs, symbol_names,
                               tf.cast(symbol_values, tf.float32))

    # sample from the state
    real_state = tf.math.real(state)
    state_mask = tf.cast(tf.math.greater(real_state, -1.5), dtype=state.dtype)
    state_zeroed = tf.multiply(state, state_mask)
    log_probs = tf.math.log(
        tf.cast(tf.square(tf.abs(state_zeroed)), tf.float64) -
        tf.constant(10**-9, dtype=tf.float64))
    samples = tf.random.categorical(log_probs,
                                    tf.gather(
                                        tf.cast(num_samples, dtype=tf.int32),
                                        0),
                                    dtype=tf.int64)

    # determine how many qubits make up each state
    individual_sizes = tf.cast(
        tf.reduce_sum(tf.cast(state_mask, tf.int32), axis=1), tf.float64)
    n_qubits = tf.cast(
        tf.math.round(
            tf.math.log((individual_sizes)) /
            tf.math.log(tf.constant(2.0, dtype=tf.float64))), tf.int32)
    max_n_qubits = tf.reduce_max(n_qubits)

    # convert samples to binary
    def gen_binary_mask(x):
        return tf.bitwise.left_shift(tf.constant(1, dtype=x.dtype), x)

    binary_conversion_mask = tf.reverse(
        tf.vectorized_map(gen_binary_mask, tf.range(0, max_n_qubits)), [0])

    def num_to_bin(x):
        return tf.cast(tf.cast(
            tf.bitwise.bitwise_and(x, tf.cast(binary_conversion_mask, x.dtype)),
            tf.bool),
                       dtype=tf.int8)

    def row_to_num(y):
        return tf.vectorized_map(num_to_bin, y)

    binary_samples = tf.vectorized_map(row_to_num, samples)

    #create the padded output tensor
    vertical_dim = tf.gather(tf.shape(binary_samples), tf.constant(1))

    def create_pad_mask(x):
        right = tf.zeros([vertical_dim, x], dtype=tf.int8)
        left = tf.ones([vertical_dim, max_n_qubits - x], dtype=tf.int8)* \
               tf.constant(2, dtype=tf.int8)
        return tf.concat([left, right], axis=1)

    padding_mask = tf.map_fn(create_pad_mask, n_qubits, dtype=tf.int8)
    return binary_samples - padding_mask
