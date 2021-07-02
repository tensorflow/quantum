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
import math
import numpy as np

import tensorflow as tf
from tensorflow_quantum.python import util

import quantum_embed


class QuantumEmbedTest(tf.test.TestCase):
    """Tests for the QuantumEmbed layer."""

    def _train(self, num_repetitions_input, depth_input, num_unitary_layers,
               num_repetitions, num_examples, data_in, data_out, epochs):
        qubits = [[cirq.GridQubit(i, j)
                   for j in range(depth_input)]
                  for i in range(num_repetitions_input)]

        quantum_datum = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        qe = quantum_embed.QuantumEmbed(qubits, num_repetitions_input,
                                        depth_input, num_unitary_layers,
                                        num_repetitions)
        outputs = qe(quantum_datum)

        model = tf.keras.Model(inputs=quantum_datum, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
                      loss=tf.keras.losses.mean_squared_error)

        data_circuits = util.convert_to_tensor([
            cirq.Circuit(qe.build_param_rotator(data_in[m, :]))
            for m in range(num_examples)
        ])

        return model.fit(x=data_circuits, y=data_out, epochs=epochs)

    def test_dimensions(self):
        """Test that dimensions are handled properly by using prime numbers."""
        num_repetitions_input = 2
        depth_input = 5
        num_unitary_layers = 7
        num_repetitions = 11
        num_examples = 13

        data_in = np.random.normal(0.0, 0.1, (
            num_examples,
            depth_input,
        ))
        data_out = np.array(
            [[2.0 * (np.linalg.norm(data_in[m, :]) < 0.15) - 1.0]
             for m in range(num_examples)],
            dtype=np.float32)

        history = self._train(num_repetitions_input,
                              depth_input,
                              num_unitary_layers,
                              num_repetitions,
                              num_examples,
                              data_in,
                              data_out,
                              epochs=20)

        assert history.history['loss'][-1] < history.history['loss'][0] * 0.5

    def test_1_sine_wave(self):
        num_repetitions_input = 1
        depth_input = 1
        num_unitary_layers = 1
        num_repetitions = 1
        num_examples = 100

        data_in = np.repeat(np.expand_dims(np.linspace(0, 2.0 * np.pi,
                                                       num_examples),
                                           axis=1),
                            depth_input,
                            axis=1)
        data_out = np.cos(data_in - 0.20160913)

        history = self._train(num_repetitions_input,
                              depth_input,
                              num_unitary_layers,
                              num_repetitions,
                              num_examples,
                              data_in,
                              data_out,
                              epochs=100)
        assert history.history['loss'][-1] < 1e-3


if __name__ == "__main__":
    tf.test.main()
