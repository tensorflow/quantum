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
"""A tf.keras.layer that ingests programs and outputs bitstring samples."""
import numbers

import numpy as np
import sympy
import tensorflow as tf

import cirq

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python import util


class Sample(tf.keras.layers.Layer):
    """A Layer that samples from a quantum circuit.

    Given an input circuit and set of parameter values, output samples
    taken from the end of the circuit.

    First lets define a simple circuit to sample from:

    >>> def get_circuit():
    ...     q0 = cirq.GridQubit(0, 0)
    ...     q1 = cirq.GridQubit(1, 0)
    ...     circuit = cirq.Circuit(
    ...         cirq.X(q0),
    ...         cirq.CNOT(q1)
    ...     )
    ...
    ...     return circuit

    When printed:

    >>> get_circuit()
    (0, 0): ───X───@───
                   │
    (1, 0): ───────X───

    Using `tfq.layers.Sample`, it's possible to sample outputs from a given
    circuit. The circuit above will put both qubits in the |1> state.

    To retrieve samples of the output state:

    >>> sample_layer = tfq.layers.Sample()
    >>> output = sample_layer(get_circuit(), repetitions=4)
    >>> output
    <tf.RaggedTensor [[[1, 1], [1, 1], [1, 1], [1, 1]]]>

    Notice above that there were no parameters passed as input into the
    layer, because the circuit wasn't parameterized. If instead the circuit
    had parameters, e.g.

    >>> def get_parameterized_circuit(symbols):
    ...     q0 = cirq.GridQubit(0, 0)
    ...     q1 = cirq.GridQubit(1, 0)
    ...     circuit = cirq.Circuit(
    ...         cirq.X(q0) ** symbols[0],
    ...         cirq.CNOT(q1)
    ...     )
    ...
    ...     return circuit

    Then it becomes necessary to provide a value for the symbol using
    `symbol_names` and `symbol_values`.

    >>> symbols = sympy.symbols(['x'])
    >>> sample_layer = tfq.layers.Sample()
    >>> output = sample_layer(get_parameterized_circuit(),
    ...     symbol_names=symbols, symbol_values=[[0.5]], repetitions=4)
    >>> tf.shape(output.to_tensor())
    tf.Tensor([1 4 2], shape=(3,), dtype=int32)

    Note that using multiple sets of parameters returns multiple
    independent samples on the same circuit.

    >>> symbols = sympy.symbols(['x'])
    >>> sample_layer = tfq.layers.Sample()
    >>> params = tf.convert_to_tensor([[0.5], [0.4]],
    ...                               dtype=tf.dtypes.float32)
    >>> output = sample_layer(get_parameterized_circuit(),
    ...     symbol_names=symbols, symbol_values=params, repetitions=4)
    >>> tf.shape(output.to_tensor())
    tf.Tensor([2 4 2], shape=(3,), dtype=int32)

    The sample layer can also be used without explicitly passing in a
    circuit, but instead using the layer with a batch of circuits. This layer
    will then sample the circuits provided in the batch with multiple sets of
    parameters, at the same time. Note that the parameters will not be
    crossed along all circuits, the circuit at index i will be run with the
    parameters at index i.

    >>> symbols = sympy.symbols(['x'])
    >>> sample_layer = tfq.layers.Sample()

    With the sample layer defined, just define both the circuit and
    parameter inputs.

    >>> q0 = cirq.GridQubit(0, 0)
    >>> q1 = cirq.GridQubit(1, 0)
    >>> circuits = tfq.convert_to_tensor([
    ...     cirq.Circuit(
    ...         cirq.X(q0) ** s[0],
    ...         cirq.CNOT(q0, q1),
    ...     ),
    ...     cirq.Circuit(
    ...         cirq.Y(q0) ** s[0],
    ...         cirq.CNOT(q0, q1),
    ...     )
    ... ])
    >>> params = tf.convert_to_tensor([[0.5], [0.4]],
    ...                              dtype=tf.dtypes.float32)

    The layer can be used as usual:

    >>> output = sample_layer(circuits,
    ...     symbol_names=symbols, symbol_values = params, repetitions=4)
    >>> tf.shape(output.to_tensor())
        tf.Tensor([2 4 2], shape=(3,), dtype=int32)


    Note: When specifying a new layer for a *compiled* `tf.keras.Model` using
    something like `tfq.layers.Sample()(cirq.Circuit(...), ...)` please
    be sure to instead use `tfq.layers.Sample()(circuit_input, ...)`
    where `circuit_input` is a `tf.keras.Input` that is filled with
    `tfq.conver_to_tensor([cirq.Circuit(..)] * batch_size)` at runtime. This
    is because compiled Keras models require non keyword layer `call` inputs to
    be traceable back to a `tf.keras.Input`.

    """

    def __init__(self, backend=None, **kwargs):
        """Instantiate this Layer.

        Create a layer that will output bitstring samples taken from either a
        simulated quantum state or a real quantum computer

        Args:
            backend: Optional Backend to use to simulate this state. Defaults
                to the native Tensorflow simulator (None), however users may
                also specify a preconfigured cirq execution object to use
                instead, which must inherit `cirq.SimulatesSamples` or a
                `cirq.Sampler`.
        """
        super().__init__(**kwargs)
        self.sample_op = circuit_execution_ops.get_sampling_op(backend)

    def call(self,
             inputs,
             *,
             symbol_names=None,
             symbol_values=None,
             repetitions=None):
        """Keras call function.

        Reference of options that are shown in examples above.

        Input options:

            1. `inputs` can be a single `cirq.Circuit`, a Python `list` of
                `cirq.Circuit`s or a pre-converted `tf.Tensor` of
                `cirq.Circuit`s.

            2. `symbol_names` can be a Python `list` of `str` or `sympy.Symbols`
                or a pre-converted `tf.Tensor` of type `str`.

            3. `symbol_values` can be a Python `list` of floating point values
                or `np.ndarray` or pre-converted `tf.Tensor` of floats.

            4. `repetitions` can be a Python `int` or a pre-converted
                `tf.Tensor` containing a single `int` entry.

        Output shape:
            `tf.RaggedTensor` with shape:
                [batch size of symbol_values, repetitions, <ragged string size>]
                    or
                [number of circuits, repetitions, <ragged string size>]

        """
        # inputs is the circuit(s).
        symbols_empty = False
        if symbol_names is None:
            symbol_names = []
        if symbol_values is None:
            symbols_empty = True
            symbol_values = [[]]

        if repetitions is None:
            raise ValueError("Number of repetitions not specified.")

        # Ingest and promote repetitions.
        if isinstance(repetitions, numbers.Integral):
            if not repetitions > 0:
                raise ValueError("Repetitions must be greater than zero.")
            repetitions = tf.convert_to_tensor([repetitions], dtype=tf.int32)

        if not tf.is_tensor(repetitions):
            raise TypeError("repetitions cannot be parsed to int32 tensor"
                            " tensor given input: ".format(repetitions))

        # Ingest and promote symbol_names.
        if isinstance(symbol_names, (list, tuple, np.ndarray)):
            if symbol_names and not all(
                [isinstance(x, (str, sympy.Symbol)) for x in symbol_names]):
                raise TypeError("Each element in symbol_names"
                                " must be a string or sympy.Symbol.")
            symbol_names = [str(s) for s in symbol_names]
            if not len(symbol_names) == len(list(set(symbol_names))):
                raise ValueError("All elements of symbol_names must be unique.")
            symbol_names = tf.convert_to_tensor(symbol_names,
                                                dtype=tf.dtypes.string)
        if not tf.is_tensor(symbol_names):
            raise TypeError("symbol_names cannot be parsed to string"
                            " tensor given input: ".format(symbol_names))

        # Ingest and promote symbol_values.
        if isinstance(symbol_values, (list, tuple, np.ndarray)):
            symbol_values = tf.convert_to_tensor(symbol_values,
                                                 dtype=tf.dtypes.float32)
        if not tf.is_tensor(symbol_values):
            raise TypeError("symbol_values cannot be parsed to float32"
                            " tensor given input: ".format(symbol_values))

        symbol_batch_dim = tf.gather(tf.shape(symbol_values), 0)

        # Ingest and promote circuit.
        if isinstance(inputs, cirq.Circuit):
            # process single circuit.
            inputs = tf.tile(util.convert_to_tensor([inputs]),
                             [symbol_batch_dim])

        elif isinstance(inputs, (list, tuple, np.ndarray)):
            # process list of circuits.
            inputs = util.convert_to_tensor(inputs)

        if not tf.is_tensor(inputs):
            raise TypeError("circuits cannot be parsed with given input:"
                            " ".format(inputs))

        if symbols_empty:
            # No symbol_values were provided. so we must tile up the
            # symbol values so that symbol_values = [[]] * number of circuits
            # provided.
            circuit_batch_dim = tf.gather(tf.shape(inputs), 0)
            symbol_values = tf.tile(symbol_values,
                                    tf.stack([circuit_batch_dim, 1]))

        return self.sample_op(inputs, symbol_names, symbol_values, repetitions)
