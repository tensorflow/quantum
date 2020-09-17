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

MATH_OP_MODULE = load_module("_tfq_math_ops.so", "math")


def inner_product(programs, symbol_names, symbol_values, other_programs):
    """Calculate the inner product between circuits.

    Calculates out[i][j] = \langle \psi_{\text{programs[i]}} \\
        (\text{symvol_values[i]}) | \psi_{\text{other_programs[j]}} \rangle

    Note: `other_programs` must not contain any free symbols. These can resolved
        beforehand with `tfq.resolve_parameters`.

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
