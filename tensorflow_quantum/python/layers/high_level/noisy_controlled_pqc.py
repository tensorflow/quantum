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
"""Module for tfq.python.layers.high_level.noisy_controlled_pqc layer."""
import numbers
import numpy as np
import tensorflow as tf

import cirq
import sympy
from tensorflow_quantum.core.ops.noise import noisy_expectation_op
from tensorflow_quantum.core.ops.noise import noisy_sampled_expectation_op
from tensorflow_quantum.python.differentiators import parameter_shift
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python import util


class NoisyControlledPQC(tf.keras.layers.Layer):
    """Noisy Controlled Parametrized Quantum Circuit (PQC) Layer.

    The `NoisyControlledPQC` layer is the noisy variant of the `ControlledPQC`
    layer. This layer uses monte carlo trajectory simulation to support noisy
    simulation functionality for the `ControlledPQC` layer. Here is a simple
    example you can use to get started:


    >>> bit = cirq.GridQubit(0, 0)
    >>> model = cirq.Circuit(
    ...     cirq.X(bit) ** sympy.Symbol('alpha'),
    ...     cirq.Z(bit) ** sympy.Symbol('beta'),
    ...     cirq.depolarize(0.01)(bit)
    ... )
    >>> outputs = tfq.layers.NoisyControlledPQC(
    ...     model,
    ...     cirq.Z(bit),
    ...     repetitions=1000,
    ...     sample_based=False
    .. )
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(bit))
    ... ])
    >>> model_params = tf.convert_to_tensor([[0.5, 0.5], [0.25, 0.75]])
    >>> res = outputs([quantum_data, model_params])
    >>> res
    tf.Tensor(
    [[-1.4901161e-08]
     [-7.0710683e-01]], shape=(2, 1), dtype=float32)


    The above example estimates the noisy expectation values using 1000
    monte-carlo trajectory simulations with analytical calculations done on each
    trajectory. Just like with the `PQC` it is *very important* that the quantum
    datapoint circuits do not contain any `sympy.Symbols` themselves (This can
    be supported with advanced usage of the `tfq.layers.Expectation` layer with
    backend='noisy'). Just like `ControlledPQC` it is possible to specify
    multiple readout operations and switch to sample based expectation
    calculation based on measured bitstrings instead of analytic calculation:


    >>> bit = cirq.GridQubit(0, 0)
    >>> model = cirq.Circuit(
    ...     cirq.X(bit) ** sympy.Symbol('alpha'),
    ...     cirq.Z(bit) ** sympy.Symbol('beta'),
    ...     cirq.depolarize(0.01)(bit)
    ... )
    >>> outputs = tfq.layers.NoisyControlledPQC(
    ...     model,
    ...     [cirq.Z(bit), cirq.X(bit), cirq.Y(bit)],
    ...     repetitions=1000,
    ...     sample_based=True
    ... )
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(bit))
    ... ])
    >>> model_params = tf.convert_to_tensor([[0.5, 0.5], [0.25, 0.75]])
    >>> res = outputs([quantum_data, model_params])
    >>> res
    tf.Tensor(
    [[-0.0028  1.     -0.0028]
     [-0.6956 -0.498  -0.498 ]], shape=(2, 3), dtype=float32)


    Unlike `ControlledPQC` a value for `backend` can not be supplied in the
    layer constructor. If you want to use a custom backend please use
    `tfq.layers.PQC` instead. A value for `differentiator` can also be supplied
    in the constructor to indicate the differentiation scheme this
    `NoisyControlledPQC` layer should use. Here's how you would take the
    gradients of the above example:


    >>> bit = cirq.GridQubit(0, 0)
    >>> model = cirq.Circuit(
    ...     cirq.X(bit) ** sympy.Symbol('alpha'),
    ...     cirq.Z(bit) ** sympy.Symbol('beta'),
    ...     cirq.depolarize(0.01)(bit)
    ... )
    >>> outputs = tfq.layers.NoisyControlledPQC(
    ...     model,
    ...     [cirq.Z(bit), cirq.X(bit), cirq.Y(bit)],
    ...     repetitions=5000,
    ...     sample_based=True,
    ...     differentiator=tfq.differentiators.ParameterShift())
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(bit))
    ... ])
    >>> model_params = tf.convert_to_tensor([[0.5, 0.5], [0.25, 0.75]])
    >>> with tf.GradientTape() as g:
    ...     g.watch(model_params)
    ...     res = outputs([quantum_data, model_params])
    >>> grads = g.gradient(res, model_params)
    >>> grads
    tf.Tensor(
    [[-3.1415927   3.1415927 ]
     [-0.9211149   0.02764606]], shape=(2, 2), dtype=float32)]


    Lastly, like all layers in TensorFlow the `NoisyControlledPQC` layer can be
    called on any `tf.Tensor` as long as it is the right shape. This means
    you could replace `model_params` in the above example with the outputs
    from a `tf.keras.Dense` layer or replace `quantum_data` with values fed
    in from a `tf.keras.Input`.
    """

    def __init__(self,
                 model_circuit,
                 operators,
                 *,
                 repetitions=None,
                 sample_based=None,
                 differentiator=None,
                 **kwargs):
        """Instantiate this layer.

        Create a layer that will output noisy expectation values of the given
        operators when fed quantum data to it's input layer. This layer will
        take two input tensors, one representing a quantum data source (these
        circuits must not contain any symbols) and the other representing
        control parameters for the model circuit that gets appended to the
        datapoints.

        model_circuit: `cirq.Circuit` containing `sympy.Symbols` that will be
            used as the model which will be fed quantum data inputs.
        operators: `cirq.PauliSum` or Python `list` of `cirq.PauliSum` objects
            used as observables at the end of the model circuit.
        repetitions: Python `int` indicating how many trajectories to use
            when estimating expectation values.
        sample_based: Python `bool` indicating whether to use sampling to
            estimate expectations or analytic calculations with each
            trajectory.
        differentiator: Optional `tfq.differentiator` object to specify how
            gradients of `model_circuit` should be calculated.
        """
        super().__init__(**kwargs)
        # Ingest model_circuit.
        if not isinstance(model_circuit, cirq.Circuit):
            raise TypeError("model_circuit must be a cirq.Circuit object."
                            " Given: ".format(model_circuit))

        self._symbols_list = list(
            sorted(util.get_circuit_symbols(model_circuit)))
        self._symbols = tf.constant([str(x) for x in self._symbols_list])

        self._circuit = util.convert_to_tensor([model_circuit])

        if len(self._symbols_list) == 0:
            raise ValueError("model_circuit has no sympy.Symbols. Please "
                             "provide a circuit that contains symbols so "
                             "that their values can be trained.")

        # Ingest operators.
        if isinstance(operators, (cirq.PauliString, cirq.PauliSum)):
            operators = [operators]

        if not isinstance(operators, (list, np.ndarray, tuple)):
            raise TypeError("operators must be a cirq.PauliSum or "
                            "cirq.PauliString, or a list, tuple, "
                            "or np.array containing them. "
                            "Got {}.".format(type(operators)))
        if not all([
                isinstance(op, (cirq.PauliString, cirq.PauliSum))
                for op in operators
        ]):
            raise TypeError("Each element in operators to measure "
                            "must be a cirq.PauliString"
                            " or cirq.PauliSum")

        self._operators = util.convert_to_tensor([operators])

        # Ingest and promote repetitions.
        if repetitions is None:
            raise ValueError("Value for repetitions must be provided when "
                             "using noisy simulation.")
        if not isinstance(repetitions, numbers.Integral):
            raise TypeError("repetitions must be a positive integer value."
                            " Given: ".format(repetitions))
        if repetitions <= 0:
            raise ValueError("Repetitions must be greater than zero.")

        self._repetitions = tf.constant(
            [[repetitions for _ in range(len(operators))]],
            dtype=tf.dtypes.int32)

        # Ingest differentiator.
        if differentiator is None:
            differentiator = parameter_shift.ParameterShift()

        # Ingest and promote sample based.
        if sample_based is None:
            raise ValueError("Please specify sample_based=False for analytic "
                             "calculations based on monte-carlo trajectories,"
                             " or sampled_based=True for measurement based "
                             "noisy estimates.")
        if not isinstance(sample_based, bool):
            raise TypeError("sample_based must be either True or False."
                            " received: {}".format(type(sample_based)))

        if not sample_based:
            self._executor = differentiator.generate_differentiable_op(
                sampled_op=noisy_expectation_op.expectation)
        else:
            self._executor = differentiator.generate_differentiable_op(
                sampled_op=noisy_sampled_expectation_op.sampled_expectation)

        self._append_layer = elementary.AddCircuit()

    @property
    def symbols(self):
        """The symbols that are managed by this layer (in-order).

        Note: `symbols[i]` indicates what symbol name the managed variables in
            this layer map to.
        """
        return [sympy.Symbol(x) for x in self._symbols_list]

    def call(self, inputs):
        """Keras call function."""
        circuit_batch_dim = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_model = tf.tile(self._circuit, [circuit_batch_dim])
        model_appended = self._append_layer(inputs[0], append=tiled_up_model)
        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])

        tiled_up_repetitions = tf.tile(self._repetitions,
                                       [circuit_batch_dim, 1])
        return self._executor(model_appended, self._symbols, inputs[1],
                              tiled_up_operators, tiled_up_repetitions)
