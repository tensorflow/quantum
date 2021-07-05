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
import pytest

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
        qe = quantum_embed.QuantumEmbed(qubits, num_repetitions_input,
                                        depth_input, num_unitary_layers,
                                        num_repetitions)

        quantum_datum = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        outputs = qe(quantum_datum)

        model = tf.keras.Model(inputs=quantum_datum, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.3),
                      loss=tf.keras.losses.mean_squared_error)

        data_circuits = util.convert_to_tensor([
            cirq.Circuit(qe.build_param_rotator(data_in[m, :]))
            for m in range(num_examples)
        ])

        return model.fit(x=data_circuits, y=data_out, epochs=epochs)

    def test_dimensions(self):
        """Test that dimensions are handled properly."""

        # All the dimensions are prime numbers. We skip 3 because it's already in use (there are
        # three angles per parameter). In case there is a bug in terms of dimensions, the code
        # could still run because of ragged arrays. Using distinct prime numbers prevents such
        # silencing of failures.
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

        # We don't care about the convergence, just that the code does not crash.
        _ = self._train(num_repetitions_input,
                        depth_input,
                        num_unitary_layers,
                        num_repetitions,
                        num_examples,
                        data_in,
                        data_out,
                        epochs=1)

    def _run_one_convergence_test(self, coefficients, num_repetitions):
        num_repetitions_input = 1
        depth_input = 1
        num_unitary_layers = 1
        num_examples = 70

        data_in = np.linspace(-np.pi, np.pi, num_examples, endpoint=False)
        data_in = np.repeat(np.expand_dims(data_in, axis=1),
                            depth_input,
                            axis=1)
        data_out = sum(coefficient[0] *
                       np.cos((i + 1.0) * data_in + coefficient[1])
                       for i, coefficient in enumerate(coefficients))

        history = self._train(num_repetitions_input,
                              depth_input,
                              num_unitary_layers,
                              num_repetitions,
                              num_examples,
                              data_in,
                              data_out,
                              epochs=50)

        return history.history['loss'][-1] < 1e-4

    def test_convergence(self):
        # Single sine wave, one repetition -> Should converge.
        assert self._run_one_convergence_test([(0.456, -np.pi / 3.0)],
                                              num_repetitions=1)
        # Two sine waves, two repetitions -> Should converge.
        assert self._run_one_convergence_test([(0.1, -np.pi / 3.0),
                                               (0.2, np.pi / 7.0)],
                                              num_repetitions=2)
        assert self._run_one_convergence_test([(0.15, -np.pi / 5.0),
                                               (0.6, np.pi / 3.0)],
                                              num_repetitions=2)
        # But if there is only one repetition, it does not converge
        assert not self._run_one_convergence_test([(0.1, -np.pi / 3.0),
                                                   (0.2, np.pi / 7.0)],
                                                  num_repetitions=1)
        assert not self._run_one_convergence_test([(0.15, -np.pi / 5.0),
                                                   (0.6, np.pi / 3.0)],
                                                  num_repetitions=1)


if __name__ == "__main__":
    tf.test.main()
