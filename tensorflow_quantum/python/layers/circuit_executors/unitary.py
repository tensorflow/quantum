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
"""A tf.keras.layer that ingests programs and outputs a unitary."""
import tensorflow as tf

from tensorflow_quantum.core.ops import tfq_unitary_op
from tensorflow_quantum.python.layers.circuit_executors import input_checks


class Unitary(tf.keras.layers.Layer):
    """A Layer that calculates unitary matrices of circuits.

    The Unitary layer can function in several different ways. The first is:
    Given an input circuit and set of parameter values, Calculate the unitary
    matrices for each parameter setting and output it to the Tensorflow graph.


    >>> a_symbol = sympy.Symbol('alpha')
    >>> qubit = cirq.GridQubit(0, 0)
    >>> my_circuit = cirq.Circuit(cirq.H(qubit) ** a_symbol)
    >>> some_values = np.array([[0.5], [3.2]])
    >>> unitary = tfq.layers.Unitary()
    >>> unitary(my_circuit, symbol_names=[a_symbol], symbol_values=some_values)
    <tf.RaggedTensor [
        [[(0.85355+0.14645j), (0.35355-0.35355j)],
         [(0.35355-0.35355j), (0.14644+0.85355j)]],
        [[(0.73507-0.08607j), (0.63958+0.20781j)],
         [(0.63958+0.20781j), (-0.54409-0.50171j)]]
    ]>


    The second use case doesn't leverage batch computation or input tensors, but
    is very useful for testing and quick debugging:


    >>> quick_verify = cirq.Circuit(cirq.X(cirq.GridQubit(0, 0)))
    >>> tfq.layers.Unitary()(quick_verify)
    <tf.RaggedTensor [[[0, 1], [1, 0]]]>


    The last and most complex supported use case is one that handles batches of
    circuits in addition to batches of parameter values. The only constraint is
    that values be supplied for all symbols in all circuits:


    >>> a_smybol = sympy.Symbol('beta')
    >>> q = cirq.GridQubit(0, 0)
    >>> first_circuit = cirq.Circuit(cirq.X(q) ** a_symbol)
    >>> second_circuit = cirq.Circuit(cirq.Y(q) ** a_symbol)
    >>> some_values = np.array([[1.0], [0.5]])
    >>> unitary = tfq.layers.Unitary()
    >>> # Calculates the unitary for X^1 and Y**0.5
    >>> unitary([first_circuit, second_circuit],
    ...     symbol_names=[a_symbol], symbol_values=some_values)
    <tf.RaggedTensor [
        [[0, 1], [1, 0]],
        [[(0.5+0.5j), (-0.5-0.5j)], [(0.5+0.5j), (0.5+0.5j)]]
    ]>


    Note: When specifying a new layer for a *compiled* `tf.keras.Model` using
    something like `tfq.layers.Unitary()(cirq.Circuit(...), ...)` please
    be sure to instead use `tfq.layers.Unitary()(circuit_input, ...)`
    where `circuit_input` is a `tf.keras.Input` that is filled with
    `tfq.conver_to_tensor([cirq.Circuit(..)] * batch_size)` at runtime. This
    is because compiled keras models require non keyword layer `call` inputs to
    be traceable back to a `tf.keras.Input`.

    """

    def __init__(self, **kwargs):
        """Instantiate a Unitary Layer.

        Create a layer that will calculate circuit unitary matrices and output
        them into the TensorFlow graph given a correct set of inputs.
        """
        super().__init__(**kwargs)
        self.unitary_op = tfq_unitary_op.get_unitary_op()

    def call(self, inputs, *, symbol_names=None, symbol_values=None):
        """Keras call function.

        Input options:
            `inputs`, `symbol_names`, `symbol_values`:
                see `input_checks.expand_circuits`

        Output shape:
            `tf.RaggedTensor` with shape:
                [batch size of symbol_values, <size of state>, <size of state>]
                    or
                [number of circuits, <size of state>, <size of state>]
        """
        inputs, symbol_names, symbol_values = input_checks.expand_circuits(
            inputs, symbol_names, symbol_values)
        return self.unitary_op(inputs, symbol_names, symbol_values)
