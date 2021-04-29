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
"""Module for tfq.python.layers.high_level.noisy_pqc layer."""
import numbers
import numpy as np
import tensorflow as tf

import cirq
import sympy
from tensorflow_quantum.core.ops.noise import \
    noisy_expectation_op, noisy_sampled_expectation_op
from tensorflow_quantum.python.differentiators import parameter_shift
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python import util


class NoisyPQC(tf.keras.layers.Layer):
    """Noisy Parametrized Quantum Circuit (PQC) Layer.

    This layer is for training **noisy** parameterized quantum models.
    Given a parameterized circuit, this layer initializes the parameters
    and manages them in a Keras native way.

    We start by defining a simple quantum circuit on one qubit.
    This circuit parameterizes an arbitrary rotation on the Bloch sphere in
    terms of the three angles a, b, and c, along with some noise:


    >>> q = cirq.GridQubit(0, 0)
    >>> (a, b, c) = sympy.symbols("a b c")
    >>> circuit = cirq.Circuit(
    ...     cirq.rz(a)(q),
    ...     cirq.rx(b)(q),
    ...     cirq.rz(c)(q),
    ...     cirq.rx(-b)(q),
    ...     cirq.rz(-a)(q),
    ...     cirq.depolarize(0.01)(q)
    ... )


    In order to extract information from our circuit, we must apply measurement
    operators.  For now we choose to make a Z measurement.  In order to observe
    an output, we must also feed our model quantum data (NOTE: quantum data
    means quantum circuits with no free parameters).  Though the output values
    will depend on the default random initialization of the angles in our model,
    one will be the negative of the other since `cirq.X(q)` causes a bit flip:


    >>> outputs = tfq.layers.NoisyPQC(
    ...     circuit,
    ...     cirq.Z(q),
    ...     repetitions=1000,
    ...     sample_based=False
    ... )
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(q))
    ... ])
    >>> res = outputs(quantum_data)
    >>> res
    <tf.Tensor: id=577, shape=(2, 1), dtype=float32, numpy=
    array([[ 0.8722095],
           [-0.8722095]], dtype=float32)>


    In the above example we estimate the value of the expectation using
    monte-carlo trajectory simulations and analytic expectation calculation.
    To emulate the process used when sampling from a truly noisy device, we
    set `sampled_based=True` to estimate the expectation value via noisy
    bitstring sampling.


    >>> measurement = [cirq.X(q), cirq.Y(q), cirq.Z(q)]
    >>> outputs = tfq.layers.NoisyPQC(
    ...     circuit,
    ...     measurement,
    ...     repetitions=5000,
    ...     sample_based=True
    ... )
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(q))
    ... ])
    >>> res = outputs(quantum_data)
    >>> res
    <tf.Tensor: id=808, shape=(2, 3), dtype=float32, numpy=
    array([[-0.38,  0.9 ,  0.14],
           [ 0.19, -0.95, -0.35]], dtype=float32)>


    Unlike `tfq.layers.PQC` no value for `backend` can be supplied in the
    layer constructor. If you want to use a custom backend please use
    `tfq.layers.PQC` instead. A value for `differentiator` can also be
    supplied in the constructor to indicate the differentiation scheme this
    `NoisyPQC` layer should use. Here's how you would take the gradients of
    the above example using a `tfq.layers.ParameterShift` differentiator.


    >>> measurement = [cirq.X(q), cirq.Y(q), cirq.Z(q)]
    >>> outputs = tfq.layers.NoisyPQC(
    ...     circuit,
    ...     measurement,
    ...     repetitions=5000,
    ...     sample_based=True,
    ...     differentiator=tfq.differentiators.ParameterShift())
    >>> quantum_data = tfq.convert_to_tensor([
    ...     cirq.Circuit(),
    ...     cirq.Circuit(cirq.X(q))
    ... ])
    >>> res = outputs(quantum_data)
    >>> res
    <tf.Tensor: id=891, shape=(2, 3), dtype=float32, numpy=
    array([[-0.5956, -0.2152,  0.7756],
           [ 0.5728,  0.1944, -0.7848]], dtype=float32)>


    Lastly, like all layers in TensorFlow the `NoisyPQC` layer can be called on
    any `tf.Tensor` as long as it is the right shape. This means you could
    replace replace `quantum_data` with values fed in from a `tf.keras.Input`.
    """

    def __init__(
            self,
            model_circuit,
            operators,
            *,
            repetitions=None,
            sample_based=None,
            differentiator=None,
            initializer=tf.keras.initializers.RandomUniform(0, 2 * np.pi),
            regularizer=None,
            constraint=None,
            **kwargs,
    ):
        """Instantiate this layer.

        Create a layer that will output noisy expectation values of the given
        operators when fed quantum data to it's input layer. This layer will
        accept one input tensor representing a quantum data source (these
        circuits must not contain any symbols) and append the model_circuit to
        them, execute them and then finally output the expectation values.


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
        initializer: Optional `tf.keras.initializer` object to specify how the
            symbols in `model_circuit` should be initialized when creating
            the managed variables.
        regularizer: Optional `tf.keras.regularizer` object applied to the
            managed variables parameterizing `model_circuit`.
        constraint: Optional `tf.keras.constraint` object applied to the
            managed variables parameterizing `model_circuit`.
        """
        super().__init__(**kwargs)

        # Ingest model_circuit.
        if not isinstance(model_circuit, cirq.Circuit):
            raise TypeError("model_circuit must be a cirq.Circuit object."
                            " Given: {}".format(model_circuit))

        self._symbols_list = list(
            sorted(util.get_circuit_symbols(model_circuit)))
        self._symbols = tf.constant([str(x) for x in self._symbols_list])

        self._model_circuit = util.convert_to_tensor([model_circuit])
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

        # Set additional parameter controls.
        self.initializer = tf.keras.initializers.get(initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)
        self.constraint = tf.keras.constraints.get(constraint)

        # Weight creation is not placed in a Build function because the number
        # of weights is independent of the input shape.
        self.parameters = self.add_weight('parameters',
                                          shape=self._symbols.shape,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer,
                                          constraint=self.constraint,
                                          dtype=tf.float32,
                                          trainable=True)

    @property
    def symbols(self):
        """The symbols that are managed by this layer (in-order).

        Note: `symbols[i]` indicates what symbol name the managed variables in
            this layer map to.
        """
        return [sympy.Symbol(x) for x in self._symbols_list]

    def symbol_values(self):
        """Returns a Python `dict` containing symbol name, value pairs.

        Returns:
            Python `dict` with `str` keys and `float` values representing
                the current symbol values.
        """
        return dict(zip(self.symbols, self.get_weights()[0]))

    def build(self, input_shape):
        """Keras build function."""
        super().build(input_shape)

    def call(self, inputs):
        """Keras call function."""
        circuit_batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_model = tf.tile(self._model_circuit, [circuit_batch_dim])
        model_appended = self._append_layer(inputs, append=tiled_up_model)
        tiled_up_parameters = tf.tile([self.parameters], [circuit_batch_dim, 1])
        tiled_up_operators = tf.tile(self._operators, [circuit_batch_dim, 1])

        tiled_up_repetitions = tf.tile(self._repetitions,
                                       [circuit_batch_dim, 1])
        return self._executor(model_appended, self._symbols,
                              tiled_up_parameters, tiled_up_operators,
                              tiled_up_repetitions)
