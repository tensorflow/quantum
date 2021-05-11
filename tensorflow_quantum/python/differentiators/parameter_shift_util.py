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
"""Util functions for general parameter-shift rule. """
import numpy as np
import tensorflow as tf

from tensorflow_quantum.core.ops import tfq_ps_util_ops

PARAMETER_IMPURITY_NAME = '_impurity_for_param_shift'


@tf.function
def parse_programs(programs, symbol_names, symbol_values, n_symbols,
                   n_shifts=2):
    """Helper function to get parameter-shifted programs after parsing programs.


    It follows:
    1. Decomposes given programs with `tfq_ps_decompose` c++ op.
    2. Construct new_programs with parameter-shifted copies of decomposed
        programs by `tfq_ps_symbol_replace` c++ op.
    3. Weights and shifts are also obtained by `tfq_ps_weights_from_symbols`
    3. Transpose the results to fed them into TensorFlow Quantum simulator.

    Args:
        programs: `tf.Tensor` of strings with shape [n_programs] containing
            the string representations of the circuits to be executed.
        symbol_names: `tf.Tensor` of strings with shape [n_symbols], which
            is used to specify the order in which the values in
            `symbol_values` should be placed inside of the circuits in
            `programs`.
        symbol_values: `tf.Tensor` of real numbers with shape
            [n_programs, n_symbols] specifying parameter values to resolve
            into the circuits specified by programs, following the ordering
            dictated by `symbol_names`.
        n_symbols: `tf.Tensor` of a positive integer representing the number of
            symbols.
        n_shifts: `tf.Tensor` of a positive integer representing the number of
            parameter-shift terms. Defaults to 2.

    Returns:
        new_programs: the new programs whose program has only one gate with
            impurity parameter-shift symbol name.
            [n_symbols, n_programs, n_param_gates, n_shifts]
        weights: parameter-shift coefficients of estimated observables.
            [n_symbols, n_programs, n_param_gates, n_shifts]
        shifts: parameter-shifted values (= matrix of symbol_values +/-shift)
            [n_symbols, n_programs, n_param_gates, n_shifts]
        n_param_gates: bypass of input n_param_gates to export it outside
    """
    decomposed_programs = tfq_ps_util_ops.tfq_ps_decompose(programs)
    delta_eig = 2.0

    # Collecting doped programs with impurity sympy.Symbol from all programs
    # with parameterized gates.
    impurity = tf.tile(tf.convert_to_tensor([PARAMETER_IMPURITY_NAME]),
                       [n_symbols])
    symbols = tf.convert_to_tensor(symbol_names)

    # Doping impurity sympy.Symbol into programs per gate per symbol.
    new_programs = tf.tile(
        tf.expand_dims(tf.transpose(
            tfq_ps_util_ops.tfq_ps_symbol_replace(decomposed_programs, symbols,
                                                  impurity), [1, 0, 2]),
                       axis=-1), [1, 1, 1, n_shifts])
    n_param_gates = tf.cast(tf.gather(tf.shape(new_programs), 2),
                            dtype=tf.int32)

    # This is a tensor of the `exponent_scalar`s of the shifted gates.
    coeff = tf.expand_dims(tf.transpose(
        tfq_ps_util_ops.tfq_ps_weights_from_symbols(decomposed_programs,
                                                    symbols), [1, 0, 2]),
                           axis=-1)

    weights_plus = coeff * np.pi * 0.5 * 0.5 * delta_eig
    weights = tf.concat([weights_plus, -weights_plus], axis=-1)
    shifts_plus = tf.math.divide_no_nan(tf.math.divide(1.0, delta_eig), coeff)

    val = tf.tile(
        tf.expand_dims(tf.expand_dims(tf.transpose(symbol_values, [1, 0]),
                                      axis=-1),
                       axis=-1), [1, 1, n_param_gates, n_shifts])

    shifts = val + tf.concat([shifts_plus, -shifts_plus], axis=-1)

    return new_programs, weights, shifts, n_param_gates
