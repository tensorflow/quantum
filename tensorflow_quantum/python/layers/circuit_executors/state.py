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
"""A tf.keras.layer that ingests programs and parameters and outputs a state."""
import numpy as np
import sympy
import tensorflow as tf

import cirq

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python import util


class State(tf.keras.layers.Layer):
    """A Layer that simulates a quantum state.

    Given an input circuit and set of parameter values, Simulate a quantum state
    and output it to the Tensorflow graph.


    A more common application is for determining the set of states produced
    by a parametrized circuit where the values of the parameters vary. Suppose
    we want to generate a family of states with varying degrees of entanglement
    ranging from separable to maximally entangled. We first define a
    parametrized circuit that can accomplish this

    >>> q0, q1 = cirq.GridQubit.rect(1, 2)
    >>> alpha = sympy.Symbol('alpha') # degree of entanglement between q0, q1
    >>> parametrized_bell_circuit = cirq.Circuit(
    ...    cirq.H(q0), cirq.CNOT(q0, q1) ** alpha)

    Now pass all of the alpha values desired to `tfq.layers.State` to compute
    a tensor of states corresponding to these preparation angles.

    >>> state_layer = tfq.layers.State()
    >>> alphas = tf.reshape(tf.range(0, 1.1, delta=0.5), (3, 1)) # FIXME: #805
    >>> state_layer(parametrized_bell_circuit,
    ...     symbol_names=[alpha], symbol_values=alphas)
    <tf.RaggedTensor [[0.707106, 0j, 0.707106, 0j],
    [(0.707106-1.2802768623032534e-08j), 0j,
        (0.353553+0.3535534143447876j), (0.353553-0.3535533547401428j)],
    [(0.707106-1.2802768623032534e-08j), 0j,
        (0.-3.0908619663705394e-08j), (0.707106+6.181723932741079e-08j)]]>


    This use case can be simplified to compute the wavefunction produced by a
    fixed circuit where the values of the parameters vary. For example, this
    layer produces a Bell state.

    >>> q0, q1 = cirq.GridQubit.rect(1, 2)
    >>> bell_circuit = cirq.Circuit(cirq.H(q0), cirq.CNOT(q0, q1))
    >>> state_layer = tfq.layers.State()
    >>> state_layer(bell_circuit)
    <tf.RaggedTensor [[(0.707106-1.2802768623032534e-08j),
                        0j,
                       (0.-3.0908619663705394e-08j),
                       (0.707106+6.181723932741079e-08j)]]>

    Not specifying `symbol_names` or `symbol_values` indicates that the
    circuit(s) does not contain any `sympy.Symbols` inside of it and tfq won't
    look for any symbols to resolve.


    `tfq.layers.State` also allows for a more complicated input signature
    wherein a different (possibly parametrized) circuit is used to prepare
    a state for each batch of input parameters. This might be useful when
    the State layer is being used to generate entirely different families
    of states. Suppose we want to generate a stream of states that are
    either computational basis states or 'diagonal' basis states (as in the
    BB84 QKD protocol). The circuits to prepare these states are:

    >>> q0 = cirq.GridQubit(0, 0)
    >>> bitval = sympy.Symbol('bitval')
    >>> computational_circuit = cirq.Circuit(cirq.X(q0) ** bitval)
    >>> diagonal_circuit = cirq.Circuit(cirq.X(q0) ** bitval, cirq.H(q0))

    Now a stream of random classical bit values can be encoded into one of
    these bases by preparing a state layer and passing in the bit values
    accompanied by their preparation circuits

    >>> qkd_layer = tfq.layers.State()
    >>> bits = [[1], [1], [0], [0]]
    >>> states_to_send = [computational_circuit,
    ...                   diagonal_circuit,
    ...                   diagonal_circuit,
    ...                   computational_circuit]
    >>> qkd_states = qkd_layer(
    ...     states_to_send, symbol_names=[bitval], symbol_values=bits)
    >>> # The third state was a '0' prepared in the diagonal basis:
    >>> qkd_states
    <tf.RaggedTensor [[-4.371138828673793e-08j, (1+4.371138828673793e-08j)],
    [(0.707106+3.0908619663705394e-08j), (-0.707106-1.364372508305678e-07j)],
    [(0.707106-1.2802768623032534e-08j), (0.707106+3.0908619663705394e-08j)],
    [(1+0j), 0j]]>


    Note: When specifying a new layer for a *compiled* `tf.keras.Model` using
    something like `tfq.layers.State()(cirq.Circuit(...), ...)` please
    be sure to instead use `tfq.layers.State()(circuit_input, ...)`
    where `circuit_input` is a `tf.keras.Input` that is filled with
    `tfq.conver_to_tensor([cirq.Circuit(..)] * batch_size)` at runtime. This
    is because compiled keras models require non keyword layer `call` inputs to
    be traceable back to a `tf.keras.Input`.

    """

    def __init__(self, backend=None, **kwargs):
        """Instantiate a State Layer.

        Create a layer that will simulate a quantum state and output it into
        the TensorFlow graph given a correct set of inputs.

        Args:
            backend: Optional Backend to use to simulate this state. Defaults
                to the native TensorFlow Quantum state vector simulator,
                however users may also specify a preconfigured cirq execution
                object to use instead, which must inherit
                `cirq.SimulatesFinalState`. Note that C++ Density Matrix
                simulation is not yet supported so to do Density Matrix
                simulation please use `cirq.DensityMatrixSimulator`.
        """
        super().__init__(**kwargs)
        self.state_op = circuit_execution_ops.get_state_op(backend)

    def call(self, inputs, *, symbol_names=None, symbol_values=None):
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

        Output shape:
            `tf.RaggedTensor` with shape:
                [batch size of symbol_values, <size of state>]
                    or
                [number of circuits, <size of state>]

        """
        # inputs is the circuit(s).
        symbols_empty = False
        if symbol_names is None:
            symbol_names = []
        if symbol_values is None:
            symbols_empty = True
            symbol_values = [[]]

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

        return self.state_op(inputs, symbol_names, symbol_values)
