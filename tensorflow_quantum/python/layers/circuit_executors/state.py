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
import tensorflow as tf

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python.layers.circuit_executors import input_checks


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


    This use case can be simplified to compute the state vector produced by a
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

        Input options:
            `inputs`, `symbol_names`, `symbol_values`:
                see `input_checks.expand_circuits`

        Output shape:
            `tf.RaggedTensor` with shape:
                [batch size of symbol_values, <size of state>]
                    or
                [number of circuits, <size of state>]
        """
        inputs, symbol_names, symbol_values = input_checks.expand_circuits(
            inputs, symbol_names, symbol_values)
        return self.state_op(inputs, symbol_names, symbol_values)
