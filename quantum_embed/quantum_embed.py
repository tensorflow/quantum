# Copyright 2021 The TensorFlow Quantum Authors. All Rights Reserved.
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

import cirq
import numpy as np
import sympy
import tensorflow as tf

from tensorflow_quantum.python import util
from tensorflow_quantum.python.layers.circuit_construction import elementary
from tensorflow_quantum.python.layers.circuit_executors import expectation


class QuantumEmbed(tf.keras.layers.Layer):
    """Quantum Embdedding Layer.

    This layers emed classical data according the papers:

    The effect of data encoding on the expressive power of variational quantum machine learning models
    Maria Schuld, Ryan Sweke, Johannes Jakob Meyer
    https://arxiv.org/abs/2008.08605

    Quantum embeddings for machine learning
    Seth Lloyd, Maria Schuld, Aroosa Ijaz, Josh Izaac, Nathan Killoran
    https://arxiv.org/abs/2001.03622

    Supervised quantum machine learning models are kernel methods
    Maria Schuld
    https://arxiv.org/abs/2101.11020

    Also useful to get started:
    https://www.youtube.com/watch?v=mNR-7OmilIo
    """

    def __init__(self, qubits, num_repetitions_input, depth_input,
                 num_unitary_layers, num_repetitions, **kwargs) -> None:
        """Instantiate this layer."""
        super().__init__(**kwargs)

        assert len(qubits) == num_repetitions_input
        assert len(qubits[0]) == depth_input
        self._qubits = qubits
        self._num_repetitions_input = num_repetitions_input
        self._depth_input = depth_input
        self._num_unitary_layers = num_unitary_layers
        self._num_repetitions = num_repetitions

        self._theta = [[[[[
            sympy.symbols(f'theta_{i}_{j}_{k}_{l}_{r}')
            for r in range(3)
        ]
                          for l in range(self._num_repetitions + 1)]
                         for k in range(self._num_unitary_layers)]
                        for j in range(self._depth_input)]
                       for i in range(self._num_repetitions_input)]

        model_circuits = []
        self._model_circuits = []
        for l in range(num_repetitions + 1):
            circuit = cirq.Circuit(self._build_parametrized_unitary(l))
            model_circuits.append(circuit)
            self._model_circuits.append(util.convert_to_tensor([circuit]))

        self._symbols_list = list(
            sorted(
                set([
                    symbol for model_circuit in model_circuits
                    for symbol in util.get_circuit_symbols(model_circuit)
                ])))
        self._symbols = tf.constant([str(x) for x in self._symbols_list])

        # Set additional parameter controls.
        self.initializer = tf.keras.initializers.get(
            tf.keras.initializers.RandomUniform(0, 2 * np.pi))
        self.regularizer = tf.keras.regularizers.get(None)
        self.constraint = tf.keras.constraints.get(None)

        # Weight creation is not placed in a Build function because the number
        # of weights is independent of the input shape.
        self.parameters = self.add_weight('parameters',
                                          shape=self._symbols.shape,
                                          initializer=self.initializer,
                                          regularizer=self.regularizer,
                                          constraint=self.constraint,
                                          dtype=tf.float32,
                                          trainable=True)

        self._operators = util.convert_to_tensor([[cirq.Z(qubits[0][0])]])
        self._executor = expectation.Expectation(backend='noiseless',
                                                 differentiator=None)
        self._append_layer = elementary.AddCircuit()

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

    def call(self, data_circuits):
        """Keras call function."""
        num_examples = tf.gather(tf.shape(data_circuits), 0)

        model_appended = tf.tile(util.convert_to_tensor([cirq.Circuit()]),
                                 [num_examples])
        for l in range(self._num_repetitions):
            tiled_up_model = tf.tile(self._model_circuits[l], [num_examples])
            model_appended = self._append_layer(model_appended,
                                                append=tiled_up_model)
            model_appended = self._append_layer(model_appended,
                                                append=data_circuits)
        tiled_up_model = tf.tile(self._model_circuits[-1], [num_examples])
        model_appended = self._append_layer(model_appended,
                                            append=tiled_up_model)

        tiled_up_parameters = tf.tile([self.parameters], [num_examples, 1])
        tiled_up_operators = tf.tile(self._operators, [num_examples, 1])
        return self._executor(model_appended,
                              symbol_names=self._symbols,
                              symbol_values=tiled_up_parameters,
                              operators=tiled_up_operators)

    def build_param_rotator(self, x):
        assert x.shape == (self._depth_input,)
        assert len(self._qubits) == self._num_repetitions_input
        assert len(self._qubits[0]) == self._depth_input

        for i in range(self._num_repetitions_input):
            for j in range(self._depth_input):
                yield cirq.Rx(rads=x[j]).on(self._qubits[i][j])

    def _build_parametrized_unitary(self, l):
        # Circuit-centric quantum classifiers
        # Maria Schuld, Alex Bocharov, Krysta Svore, Nathan Wiebe
        # https://arxiv.org/abs/1804.00633

        # PennyLane StronglyEntanglingLayers:
        # https://sourcegraph.com/github.com/PennyLaneAI/pennylane/-/blob/pennylane/templates/layers/strongly_entangling.py
        assert len(self._qubits) == self._num_repetitions_input
        assert len(self._qubits[0]) == self._depth_input

        num_qubits = self._depth_input * self._num_repetitions_input

        if num_qubits > 1:
            ranges = [(k % (num_qubits - 1)) + 1
                      for k in range(self._num_unitary_layers)]

        for k in range(self._num_unitary_layers):
            for i in range(self._num_repetitions_input):
                for j in range(self._depth_input):
                    yield cirq.Rz(rads=self._theta[i][j][k][l][0]).on(
                        self._qubits[i][j])
                    yield cirq.Ry(rads=self._theta[i][j][k][l][1]).on(
                        self._qubits[i][j])
                    yield cirq.Rz(rads=self._theta[i][j][k][l][2]).on(
                        self._qubits[i][j])

            if num_qubits > 1:
                for ij1 in range(num_qubits):
                    ij2 = (ij1 + ranges[k]) % num_qubits
                    assert ij1 != ij2

                    i1 = ij1 % self._num_repetitions_input
                    j1 = ij1 // self._num_repetitions_input

                    i2 = ij2 % self._num_repetitions_input
                    j2 = ij2 // self._num_repetitions_input

                    yield cirq.CNOT(self._qubits[i1][j1], self._qubits[i2][j2])
