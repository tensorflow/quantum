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
"""Layers for constructing qudit circuits on qubit backends."""
import numpy as np
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops import tfq_utility_ops
from tensorflow_quantum.python import util


def layer_input_check(inputs, precision, cliques):
    """Function for error checking and expanding qudit layer inputs.

    Args:
        inputs: a single `cirq.Circuit`, a Python `list` of
            `cirq.Circuit`s or a pre-converted `tf.Tensor` of
            `cirq.Circuit`s.
        precision: a Python `list` of `int`s.  Entry `precision[i]` sets
            the number of qubits on which qudit `i` is supported.
        cliques: a Python `dict` mapping sets of qudit labels which are to be
            combined via tensor product to the coefficient of that product.

    Returns:
        inputs: `tf.Tensor` of dtype `string` with shape [batch_size]
            containing the serialized circuits to which further operations
            will be appended.
        precision: Argument passed unchanged after error checking.
        cliques: Argument passed unchanged after error checking.
    """
    if isinstance(inputs, cirq.Circuit):
        inputs = util.convert_to_tensor([inputs])
    if isinstance(inputs, (tuple, list, np.ndarray)):
        inputs = util.convert_to_tensor(inputs)
    if not tf.is_tensor(inputs):
        raise TypeError("Circuits cannot be parsed with given input:"
                        " ".format(inputs))

    if precision is None:
        raise RuntimeError("`precisions` must be a list of integers.")

    if cliques is None:
        raise RuntimeError("`cliques` must be a dict mapping sets to floats.")

    return inputs, precision, cliques


class AppendCostExp(tf.keras.layers.Layer):
    """Layer appending exponential of a qudit cost to the input circuit tensor.


    Note: When specifying a new layer for a *compiled* `tf.keras.Model` using
    something like
    `tfq.layers.AppendCostExp()(cirq.Circuit(...), ...)`
    please be sure to instead use
    `tfq.layers.AppendCostExp()(circuit_input, ...)`
    where `circuit_input` is a `tf.keras.Input` that is filled with
    `tfq.conver_to_tensor([cirq.Circuit(..)] * batch_size)` at runtime. This
    is because compiled Keras models require non keyword layer `call` inputs to
    be traceable back to a `tf.keras.Input`.

    """

    def __init__(self, **kwargs):
        """Instantiate this layer."""
        super().__init__(**kwargs)

    def call(self, inputs, *, precision=None, cost=None):
        """Keras call method.

        Input options:
            `inputs`, `symbol_names`, `symbol_values`:
                see `layer_input_checks`

        Output shape:
            `tf.Tensor` of shape [batch_size] containing the exponential of the
                qudit cost appended to the input circuits.

        """

        inputs, precision, cost = layer_input_check(inputs, precision, cost)


        batch_dim = tf.gather(tf.shape(inputs), 0)
        if isinstance(append, cirq.Circuit):
            append = tf.tile(util.convert_to_tensor([append]), [batch_dim])
        else:
            append = util.convert_to_tensor(append)

        return tfq_utility_ops.tfq_append_circuit(inputs, append)
