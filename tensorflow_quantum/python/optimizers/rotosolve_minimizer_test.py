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
"""Test module for tfq.python.optimizers.rotosolve_minimizer optimizer."""
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import cirq
import sympy

from tensorflow_quantum.python.layers.high_level import pqc
from tensorflow_quantum.python import util
from tensorflow_quantum.python.optimizers import rotosolve_minimizer
from tensorflow_quantum.python.optimizers.utils import function_factory


class RotosolveMinimizerTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the rotosolve optimization algorithm."""

    def test_optimization(self):
        """Optimization test."""

        x = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=float)

        y = np.asarray([[-1], [1], [1], [-1]], dtype=float)

        def convert_to_circuit(input_data):
            """Encode into quantum datapoint."""
            values = np.ndarray.flatten(input_data)
            qubits = cirq.GridQubit.rect(1, 2)
            circuit = cirq.Circuit()
            for i, value in enumerate(values):
                if value:
                    circuit.append(cirq.X(qubits[i]))
            return circuit

        x_circ = util.convert_to_tensor([convert_to_circuit(x) for x in x])

        # Create two qubits
        q0, q1 = cirq.GridQubit.rect(1, 2)

        # Create an anzatz on these qubits.
        a, b = sympy.symbols('a b')  # parameters for the circuit
        circuit = cirq.Circuit(
            cirq.rx(a).on(q0),
            cirq.ry(b).on(q1), cirq.CNOT(control=q0, target=q1))

        # Build the Keras model.
        model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the
            # readout gate, range [-1,1].
            pqc.PQC(circuit, cirq.Z(q1)),
        ])

        def hinge_loss(y_true, y_pred):
            # Here we use hinge loss as the cost function
            return tf.reduce_mean(tf.cast(1 - y_true * y_pred, tf.float32))

        # Initial guess of the parameter from random number
        rotosolve_minimizer.minimize(
            function_factory(model, hinge_loss, x_circ, y),
            np.random.random(2) * 2 * np.pi)
