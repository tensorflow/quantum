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
"""A tf.keras.layer that ingests programs and outputs sampled expectation values
."""
import numbers

import numpy as np
import sympy
import tensorflow as tf

import cirq
from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python.differentiators import differentiator as diff
from tensorflow_quantum.python.differentiators import parameter_shift
from tensorflow_quantum.python import util


class SampledExpectation(tf.keras.layers.Layer):
    """A layer that calculates a sampled expectation value.

    Given an input circuit and set of parameter values, output expectation
    values of observables computed using measurement results sampled from
    the input circuit.


    First define a simple helper function for generating a parametrized
    quantum circuit that we will use throughout:

    >>> def _gen_single_bit_rotation_problem(bit, symbols):
    ...     \"""Generate a toy problem on 1 qubit.\"""
    ...     starting_state = [0.123, 0.456, 0.789]
    ...     circuit = cirq.Circuit(
    ...         cirq.Rx(starting_state[0])(bit),
    ...         cirq.Ry(starting_state[1])(bit),
    ...         cirq.Rz(starting_state[2])(bit),
    ...         cirq.Rz(symbols[2])(bit),
    ...         cirq.Ry(symbols[1])(bit),
    ...         cirq.Rx(symbols[0])(bit)
    ...     )
    ...     return circuit


    In quantum machine learning there are two very common use cases that
    align with keras layer constructs. The first is where the circuits
    represent the input data points:


    >>> bit = cirq.GridQubit(0, 0)
    >>> symbols = sympy.symbols('x y z')
    >>> ops = [-1.0 * cirq.Z(bit), cirq.X(bit) + 2.0 * cirq.Z(bit)]
    >>> num_samples = [100, 200]
    >>> circuit_list = [
    ...     _gen_single_bit_rotation_problem(bit, symbols),
    ...     cirq.Circuit(
    ...         cirq.Z(bit) ** symbols[0],
    ...         cirq.X(bit) ** symbols[1],
    ...         cirq.Z(bit) ** symbols[2]
    ...     ),
    ...     cirq.Circuit(
    ...         cirq.X(bit) ** symbols[0],
    ...         cirq.Z(bit) ** symbols[1],
    ...         cirq.X(bit) ** symbols[2]
    ...     )
    ... ]
    >>> sampled_expectation_layer = tfq.layers.SampledExpectation()
    >>> output = sampled_expectation_layer(
    ...     circuit_list,
    ...     symbol_names=symbols,
    ...     operators=ops,
    ...     repetitions=num_samples)
    >>> # Here output[i][j] corresponds to the sampled expectation
    >>> # of all the ops in ops w.r.t circuits[i] where Keras managed
    >>> # variables are placed in the symbols 'x', 'y', 'z'.
    >>> tf.shape(output)
    tf.Tensor([3 2], shape=(2,), dtype=int32)


    Here, different `cirq.Circuit` instances sharing the common symbols 'x',
    'y' and 'z' are used as input. Keras uses the `symbol_names`
    argument to map Keras managed variables to these circuits constructed
    with `sympy.Symbol`s. The shape of `num_samples` is equal to that of `ops`.


    The second most common use case is where there is a fixed circuit and
    the expectation operators vary:


    >>> bit = cirq.GridQubit(0, 0)
    >>> symbols = sympy.symbols('x, y, z')
    >>> ops = [-1.0 * cirq.Z(bit), cirq.X(bit) + 2.0 * cirq.Z(bit)]
    >>> fixed_circuit = _gen_single_bit_rotation_problem(bit, symbols)
    >>> expectation_layer = tfq.layers.SampledExpectation()
    >>> output = expectation_layer(
    ...     fixed_circuit,
    ...     symbol_names=symbols,
    ...     operators=ops,
    ...     repetitions=5000,
    ...     initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi))
    >>> # Here output[i][j] corresponds to
    >>> # the sampled expectation of operators[i][j] using 5000 samples w.r.t
    >>> # the circuit where variable values are managed by keras and store
    >>> # numbers in the symbols 'x', 'y', 'z'.
    >>> tf.shape(output)
    tf.Tensor([1 2], shape=(2,), dtype=int32)


    Here different `cirq.PauliSum` or `cirq.PauliString` instances can be
    used as input to calculate the expectation on the fixed circuit that
    the layer was initially constructed with.


    There are also some more complex use cases that provide greater flexibility.
    Notably these configurations all make use of the `symbol_values` parameter
    that causes the `SampledExpectation` layer to stop managing the
    `sympy.Symbol`s in the quantum circuits and instead requires the user to
    supply inputs themselves. Lets look at the case where there
    is a single fixed circuit, some fixed operators and symbols that must be
    common to all circuits:


    >>> bit = cirq.GridQubit(0, 0)
    >>> symbols = sympy.symbols('x y z')
    >>> ops = [cirq.Z(bit), cirq.X(bit)]
    >>> num_samples = [100, 200]
    >>> circuit = _gen_single_bit_rotation_problem(bit, symbols)
    >>> values = [[1,1,1], [2,2,2], [3,3,3]]
    >>> sampled_expectation_layer = tfq.layers.SampledExpectation()
    >>> output = sampled_expectation_layer(
    ...     circuit,
    ...     symbol_names=symbols,
    ...     symbol_values=values,
    ...     operators=ops,
    ...     repetitions=num_samples)
    >>> # output[i][j] = The sampled expectation of ops[j] with
    >>> # values_tensor[i] placed into the symbols of the circuit
    >>> # with the order specified by feed_in_params.
    >>> # so output[1][2] = The sampled expectation of a circuit with parameter
    >>> # values [2,2,2] w.r.t Pauli X, estimated using 200 samples per term.
    >>> output  # Non-deterministic result. It can vary every time.
    tf.Tensor(
    [[0.52, 0.72],
     [0.34, 1.  ],
     [0.78, 0.48]], shape=(3, 2), dtype=float32)


    Tip: you can compare the above result with that of `Expectation`:
    tf.Tensor(
    [[0.63005245 0.76338404]
     [0.25707167 0.9632684 ]
     [0.79086655 0.5441111 ]], shape=(3, 2), dtype=float32)


    Here is a simple model that uses this particular input signature of
    `tfq.layers.SampledExpectation`, that learns to undo the random rotation
    of the qubit:


    >>> bit = cirq.GridQubit(0, 0)
    >>> symbols = sympy.symbols('x, y, z')
    >>> circuit = _gen_single_bit_rotation_problem(bit, symbols)
    >>> control_input = tf.keras.Input(shape=(1,))
    >>> circuit_inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
    >>> d1 = tf.keras.layers.Dense(10)(control_input)
    >>> d2 = tf.keras.layers.Dense(3)(d1)
    >>> expectation = tfq.layers.SampledExpectation()(
    ...     circuit_inputs, # See note below!
    ...     symbol_names=symbols,
    ...     symbol_values=d2,
    ...     operators=cirq.Z(bit),
    ...     repetitions=5000)
    >>> data_in = np.array([[1], [0]], dtype=np.float32)
    >>> data_out = np.array([[1], [-1]], dtype=np.float32)
    >>> model = tf.keras.Model(
    ...     inputs=[circuit_inputs, control_input], outputs=expectation)
    >>> model.compile(
    ...     optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    ...     loss=tf.keras.losses.mean_squared_error)
    >>> history = model.fit(
    ...     x=[tfq.convert_to_tensor([circuit] * 2), data_in],
    ...     y=data_out,
    ...     epochs=100)


    For an example featuring this layer, please check out `Taking gradients`
    in our dev website http://www.tensorflow.org/quantum/tutorials.

    Lastly `symbol_values`, `operators` and circuit `inputs` can all be fed
    Python `list` objects. In addition to this they can also be fed `tf.Tensor`
    inputs, meaning that you can input all of these things from other Tensor
    objects (like `tf.keras.Dense` layer outputs or `tf.keras.Input`s etc).


    Note: When specifying a new layer for a *compiled* `tf.keras.Model` using
    something like `tfq.layers.SampledExpectation()(cirq.Circuit(...), ...)`
    please be sure to instead use
    `tfq.layers.SampledExpectation()(circuit_input, ...)` where
    `circuit_input` is a `tf.keras.Input` that is filled with
    `tfq.conver_to_tensor([cirq.Circuit(..)] * batch_size)` at runtime. This
    is because compiled Keras models require non keyword layer `call` inputs to
    be traceable back to a `tf.keras.Input`.

    """

    def __init__(self, backend=None, differentiator=None, **kwargs):
        """Instantiate this Layer.

        Create a layer that will output expectation values gained from
        simulating a quantum circuit.

        Args:
            backend: Optional Backend to use to simulate states. Defaults to
                the native TensorFlow simulator (None), however users may also
                specify a preconfigured cirq simulation object to use instead,
                which must inherit `cirq.SimulatesFinalState`.
            differentiator: Optional Differentiator to use to calculate analytic
                derivative values of given operators_to_measure and circuit,
                which must inherit `tfq.differentiators.Differentiator`.
                Defaults to None, which uses `parameter_shift.ParameterShift()`.

        """
        super().__init__(**kwargs)

        # Ingest backend.
        if not isinstance(backend, cirq.Sampler) and \
                isinstance(backend, cirq.SimulatesFinalState):
            raise TypeError("Backend implements cirq.SimulatesFinalState but "
                            "not cirq.Sampler. Please use Expectation instead.")

        # Ingest differentiator.
        if differentiator is None:
            differentiator = parameter_shift.ParameterShift()

        if not isinstance(differentiator, diff.Differentiator):
            raise TypeError("Differentiator must inherit from "
                            "tfq.differentiators.Differentiator")

        self._expectation_op = differentiator.generate_differentiable_op(
            sampled_op=circuit_execution_ops.get_sampled_expectation_op(
                backend=backend))

        self._w = None

    def call(self,
             inputs,
             *,
             symbol_names=None,
             symbol_values=None,
             operators=None,
             repetitions=None,
             initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi)):
        """Keras call function."""

        # inputs is the circuit(s).
        values_empty = False
        if symbol_names is None:
            symbol_names = []
        if symbol_values is None:
            values_empty = True
            symbol_values = [[]]

        # Ingest and promote symbol_names.
        if isinstance(symbol_names, (list, tuple, np.ndarray)):
            if not all(
                    isinstance(x, (str, sympy.Symbol)) for x in symbol_names):
                raise TypeError("Each element in symbol_names"
                                " must be a string or sympy.Symbol.")
            symbol_names = [str(s) for s in symbol_names]
            if not len(symbol_names) == len(list(set(symbol_names))):
                raise ValueError("All elements of symbol_names must be unique.")
            symbol_names = tf.identity(
                tf.convert_to_tensor(symbol_names, dtype=tf.dtypes.string))

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

        # Ingest and promote circuits.
        # Would be nice to support python circuits *fully* in this layer.
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

        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)

        # Ingest and promote operators.
        if operators is None:
            raise RuntimeError("Value for operators not provided. operators "
                               "must be one of cirq.PauliSum, cirq.PauliString"
                               ", or a list/tensor/tuple containing "
                               "cirq.PauliSum or cirq.PauliString.")

        op_needs_tile = False
        if isinstance(operators, (cirq.PauliSum, cirq.PauliString)):
            # If we are given a single operator promote it to a list and tile
            # it up to size.
            operators = [[operators]]
            op_needs_tile = True

        if isinstance(operators, (list, tuple, np.ndarray)):
            if not isinstance(operators[0], (list, tuple, np.ndarray)):
                # If we are given a flat list of operators. tile them up
                # to match the batch size of circuits.
                operators = [operators]
                op_needs_tile = True
            operators = util.convert_to_tensor(operators)

        if op_needs_tile:
            # Don't tile up if the user gave a python list that was precisely
            # the correct size to match circuits outer batch dim.
            operators = tf.tile(operators, [circuit_batch_dim, 1])

        if not tf.is_tensor(operators):
            raise TypeError("operators cannot be parsed to string tensor"
                            " given input: ".format(operators))

        # Ingest and promote repetitions.
        if repetitions is None:
            raise RuntimeError("Value for repetitions not provided.")

        reps_need_tile = False
        if isinstance(repetitions, numbers.Integral):
            # Must tile it up to size to match operators if many operators
            # were provided but only one number was provided.
            repetitions = tf.ones(tf.shape(operators),
                                  dtype=tf.dtypes.int32) * repetitions

        if isinstance(repetitions, (list, tuple, np.ndarray)):
            if not isinstance(repetitions[0], (list, tuple, np.ndarray)):
                repetitions = [repetitions]
                reps_need_tile = True

            repetitions = tf.convert_to_tensor(repetitions,
                                               dtype=tf.dtypes.int32)

        if reps_need_tile:
            # Don't tile up if the user gave a python list that was precisely
            # the correct size to match circuits outer batch dim.
            repetitions = tf.tile(repetitions, [circuit_batch_dim, 1])

        if not tf.is_tensor(repetitions):
            raise TypeError("repetitions cannot be parsed to int32 tensor"
                            " given input: ".format(repetitions))

        if values_empty:
            # No symbol_values were provided. So we assume the user wants us
            # to create and manage variables for them. We will do so by
            # creating a weights variable and tiling it up to appropriate
            # size of [batch, num_symbols].

            if self._w is None:
                # don't re-add variable.
                self._w = self.add_weight(name='circuit_learnable_parameters',
                                          shape=symbol_names.shape,
                                          initializer=initializer)

            symbol_values = tf.tile(tf.expand_dims(self._w, axis=0),
                                    tf.stack([circuit_batch_dim, 1]))

        num_samples = repetitions

        return self._expectation_op(inputs, symbol_names, symbol_values,
                                    operators, num_samples)
