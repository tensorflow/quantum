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
"""Module for tfq.python.layers.high_level.controlled_pqc layer."""
import numbers
import numpy as np
import tensorflow as tf
import cirq
import sympy

from tensorflow_quantum.python.layers.circuit_executors import \
    expectation, sampled_expectation
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python import util


class ControlledPQC(tf.keras.layers.Layer):
    """Controlled Parametrized Quantum Circuit (PQC) Layer.

    The `ControlledPQC` layer is very similar to the regular `PQC` layer, but
    with one major difference. The `ControlledPQC` layer requires the caller
    of the layer to provide the control parameter inputs for `model_circuit`.
    You can see how this works through a simple example:


    >>> bit = cirq.GridQubit(0, 0)
    >>> model = cirq.Circuit(
    ...     cirq.X(bit) ** sympy.Symbol('alpha'),
    ...     cirq.Z(bit) ** sympy.Symbol('beta')
    ... )
    >>> outputs = tfq.layers.ControlledPQC(model, cirq.Z(bit))
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


    Just like with the `PQC` it is *very important* that the quantum datapoint
    circuits do not contain any `sympy.Symbols` themselves (This can be
    supported with advanced usage of the `tfq.layers.Expectation` layer). Just
    like `PQC` it is possible to specify multiple readout operations and
    switch to sample based expectation calculation:


    >>> bit = cirq.GridQubit(0, 0)
    >>> model = cirq.Circuit(
    ...     cirq.X(bit) ** sympy.Symbol('alpha'),
    ...     cirq.Z(bit) ** sympy.Symbol('beta')
    ... )
    >>> outputs = tfq.layers.ControlledPQC(
    ...     model,
    ...     [cirq.Z(bit), cirq.X(bit), cirq.Y(bit)],
    ...     repetitions=5000)
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


    A value for `backend` can also be supplied in the layer constructor
    arguments to indicate which supported backend you would like to use.
    A value for `differentiator` can also be supplied in the constructor
    to indicate the differentiation scheme this `ControlledPQC` layer
    should use. Here's how you would take the gradients of the
    above example using a `cirq.Simulator` backend (which is slower
    than `backend='noiseless'` which uses C++):


    >>> bit = cirq.GridQubit(0, 0)
    >>> model = cirq.Circuit(
    ...     cirq.X(bit) ** sympy.Symbol('alpha'),
    ...     cirq.Z(bit) ** sympy.Symbol('beta')
    ... )
    >>> outputs = tfq.layers.ControlledPQC(
    ...     model,
    ...     [cirq.Z(bit), cirq.X(bit), cirq.Y(bit)],
    ...     repetitions=5000,
    ...     backend=cirq.Simulator(),
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


    Lastly, like all layers in TensorFlow the `ControlledPQC` layer can be
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
                 backend='noiseless',
                 differentiator=None,
                 **kwargs):
        """Instantiate this layer.

        Create a layer that will output expectation values of the given
        operators when fed quantum data to it's input layer. This layer will
        take two input tensors, one representing a quantum data source (these
        circuits must not contain any symbols) and the other representing
        control parameters for the model circuit that gets appended to the
        datapoints.

        model_circuit: `cirq.Circuit` containing `sympy.Symbols` that will be
            used as the model which will be fed quantum data inputs.
        operators: `cirq.PauliSum` or Python `list` of `cirq.PauliSum` objects
            used as observables at the end of the model circuit.
        repetitions: Optional Python `int` indicating how many samples to use
            when estimating expectation values. If `None` analytic expectation
            calculation is used.
        backend: Optional Backend to use to simulate states. Defaults to
            the noiseless TensorFlow simulator, however users may also
            specify a preconfigured cirq simulation object to use instead.
            If a cirq object is given it must inherit `cirq.Sampler` if
            `sampled_based` is True or it must inherit
            `cirq.sim.simulator.SimulatesExpectationValues` if `sample_based` is
            False.
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
        self._analytic = False
        if repetitions is None:
            self._analytic = True

        if not self._analytic and not isinstance(repetitions, numbers.Integral):
            raise TypeError("repetitions must be a positive integer value."
                            " Given: ".format(repetitions))

        if not self._analytic and repetitions <= 0:
            raise ValueError("Repetitions must be greater than zero.")

        if not self._analytic:
            self._repetitions = tf.constant(
                [[repetitions for _ in range(len(operators))]],
                dtype=tf.dtypes.int32)

        # Ingest backend and differentiator.
        if backend == 'noisy':
            raise ValueError("noisy backend value is not supported in "
                             "tfq.layers.ControlledPQC. Please use "
                             "tfq.layers.NoisyControlledPQC instead.")

        not_default = backend != 'noiseless'
        not_default &= backend is not None  # legacy backend=None support.
        if not isinstance(
                backend,
                cirq.Sampler) and repetitions is not None and not_default:
            raise TypeError("provided backend does not inherit cirq.Sampler "
                            "and repetitions!=None. Please provide a backend "
                            "that inherits cirq.Sampler or set "
                            "repetitions=None.")

        if not isinstance(backend, cirq.sim.simulator.SimulatesExpectationValues
                         ) and repetitions is None and not_default:
            raise TypeError("provided backend does not inherit "
                            "cirq.sim.simulator.SimulatesExpectationValues and "
                            "repetitions=None. Please provide a backend that "
                            "inherits "
                            "cirq.sim.simulator.SimulatesExpectationValues.")

        if self._analytic:
            self._layer = expectation.Expectation(backend=backend,
                                                  differentiator=differentiator)
        else:
            self._layer = sampled_expectation.SampledExpectation(
                backend=backend, differentiator=differentiator)

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

        # this is disabled to make autograph compilation easier.
        # pylint: disable=no-else-return
        if self._analytic:
            return self._layer(model_appended,
                               symbol_names=self._symbols,
                               symbol_values=inputs[1],
                               operators=tiled_up_operators)
        else:
            tiled_up_repetitions = tf.tile(self._repetitions,
                                           [circuit_batch_dim, 1])
            return self._layer(model_appended,
                               symbol_names=self._symbols,
                               symbol_values=inputs[1],
                               operators=tiled_up_operators,
                               repetitions=tiled_up_repetitions)

        # pylint: enable=no-else-return
