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

import tensorflow as tf

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.core.ops.noise import noisy_samples_op
from tensorflow_quantum.python.layers.circuit_executors import input_checks


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
    ...         cirq.CNOT(q0, q1)
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

    def __init__(self, backend='noiseless', **kwargs):
        """Instantiate this Layer.

        Create a layer that will output bitstring samples taken from either a
        simulated quantum state or a real quantum computer

        Args:
            backend: Optional Backend to use to simulate this state. Defaults
                to the noiseless simulator. Options are {'noisy', 'noiseless'},
                however users may also specify a preconfigured cirq execution
                object to use instead, which must inherit `cirq.Sampler`.
        """
        super().__init__(**kwargs)
        used_op = None
        if backend == 'noiseless':
            used_op = circuit_execution_ops.get_sampling_op(None)
        elif backend == 'noisy':
            used_op = noisy_samples_op.samples
        else:
            used_op = circuit_execution_ops.get_sampling_op(backend)

        self.sample_op = used_op

    def call(self,
             inputs,
             *,
             symbol_names=None,
             symbol_values=None,
             repetitions=None):
        """Keras call function.

        Input options:
            `inputs`, `symbol_names`, `symbol_values`:
                see `input_checks.expand_circuits`
            `repetitions`: a Python `int` or a pre-converted
                `tf.Tensor` containing a single `int` entry.

        Output shape:
            `tf.RaggedTensor` with shape:
                [batch size of symbol_values, repetitions, <ragged string size>]
                    or
                [number of circuits, repetitions, <ragged string size>]
        """
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

        inputs, symbol_names, symbol_values = input_checks.expand_circuits(
            inputs, symbol_names, symbol_values)

        return self.sample_op(inputs, symbol_names, symbol_values, repetitions)
