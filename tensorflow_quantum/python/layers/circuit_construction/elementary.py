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
"""Elementary layers, such as the AddCircuit layer."""
import numpy as np
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops import tfq_utility_ops
from tensorflow_quantum.python import util


class AddCircuit(tf.keras.layers.Layer):
    """A layer that pre/appends a sequence of gates to the input circuit tensor.

    This layer allows for an arbitrary `cirq.Circuit` (or list of circuits of
    equal length to the input) to be appended or prepended to the list of input
    circuits.


    >>> qubits = cirq.GridQubit.rect(1, 4)
    >>> add = tfq.layers.AddCircuit()
    >>> output = add(
    ...     [cirq.Circuit(cirq.Y(qubits[0])), cirq.Circuit(cirq.Z(qubits[0]))]
    ...     append = cirq.Circuit(cirq.Y(qubits[0]))
    ... )
    >>> # Now we have a layer that would append a single Y gate to any inputs.
    >>> tfq.from_tensor(output)
    [cirq.Circuit([
        cirq.Moment(operations=[
            cirq.Y.on(cirq.GridQubit(0, 0)),
        ]),
        cirq.Moment(operations=[
            cirq.Y.on(cirq.GridQubit(0, 0)),
        ]),
    ])
     cirq.Circuit([
        cirq.Moment(operations=[
            cirq.Z.on(cirq.GridQubit(0, 0)),
        ]),
        cirq.Moment(operations=[
            cirq.Y.on(cirq.GridQubit(0, 0)),
        ]),
    ])]


    Note: When specifying a new layer for a *compiled* `tf.keras.Model` using
    something like
    `tfq.layers.AddCircuit()(cirq.Circuit(...), append/prepend=cirq.Circuit())`
    please be sure to instead use
    `tfq.layers.AddCircuit()(circuit_input, append/prepend=cirq.Circuit())`
    where `circuit_input` is a `tf.keras.Input` that is filled with
    `tfq.conver_to_tensor([cirq.Circuit(..)] * batch_size)` at runtime
    (`append/prepend` can still remain a `cirq.Circuit` object). This
    is because compiled Keras models require non keyword layer `call` inputs to
    be traceable back to a `tf.keras.Input`.

    """

    def __init__(self, **kwargs):
        """Instantiate this layer."""
        super().__init__(**kwargs)

    def call(self, inputs, *, append=None, prepend=None):
        """Keras call method.

        Input options:

            1. `inputs` can be a single `cirq.Circuit`, a Python `list` of
                `cirq.Circuit`s or a pre-converted `tf.Tensor` of
                `cirq.Circuit`s.

            2. `append` can be a Python `list` of `cirq.Circuit`s or a
                pre-converted `tf.Tensor` of type `str` (containing circuits).

            3. `prepend` can be a Python `list` of `cirq.Circuit`s or a
                pre-converted `tf.Tensor` of type `str` (containing circuits).

        Output shape:
            `tf.Tensor` of shape [input size] containing circuits with append
            circuits appended or prepend circuits prepended.

        """
        # inputs is circuit.

        if append is None and prepend is None:
            raise ValueError("Values must be provided for append or prepend.")

        if append is not None and prepend is not None:
            raise ValueError(
                "Values cannot be given for both append and prepend.")

        # Ingest input circuit(s).
        if isinstance(inputs, cirq.Circuit):
            inputs = util.convert_to_tensor([inputs])

        if isinstance(inputs, (tuple, list, np.ndarray)):
            inputs = util.convert_to_tensor(inputs)

        if not tf.is_tensor(inputs):
            raise TypeError("Circuits cannot be parsed with given input:"
                            " ".format(inputs))

        batch_dim = tf.gather(tf.shape(inputs), 0)

        # Ingest append circuit(s):
        if append is not None:
            if isinstance(append, cirq.Circuit):
                append = tf.tile(util.convert_to_tensor([append]), [batch_dim])
            if isinstance(append, (tuple, list, np.ndarray)):
                append = util.convert_to_tensor(append)
            if not tf.is_tensor(append):
                raise TypeError(
                    "Append circuits cannot be parsed with given input:"
                    " ".format(append))

            return tfq_utility_ops.append_circuit(inputs, append)

        # Otherwise ingest prepend circuits.
        if isinstance(prepend, cirq.Circuit):
            prepend = tf.tile(util.convert_to_tensor([prepend]), [batch_dim])
        if isinstance(prepend, (tuple, list, np.ndarray)):
            prepend = util.convert_to_tensor(prepend)
        if not tf.is_tensor(prepend):
            raise TypeError(
                "Prepend circuits cannot be parsed with given input:"
                " ".format(prepend))

        return tfq_utility_ops.append_circuit(prepend, inputs)
