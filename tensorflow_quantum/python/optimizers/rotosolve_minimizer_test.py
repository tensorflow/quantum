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
from operator import mul
from functools import reduce
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import cirq
import sympy
from tensorflow_quantum.python.layers.high_level import pqc
from tensorflow_quantum.python import util
from tensorflow_quantum.python.optimizers import rotosolve_minimizer


def loss_function_with_model_parameters(model, loss, train_x, train_y):
    """Create a new function that assign the model parameter to the model
    and evaluate its value.

    Args:
        model : an instance of `tf.keras.Model` or its subclasses.
        loss : a function with signature loss_value = loss(pred_y, true_y).
        train_x : the input part of training data.
        train_y : the output part of training data.

    Returns:
        A function that has a signature of:
            loss_value = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    count = 0
    sizes = []

    # Record the shape of each parameter
    for shape in shapes:
        n = reduce(mul, shape)
        sizes.append(n)
        count += n

    # Function accept the parameter and evaluate model
    @tf.function
    def func(params):
        """A function that can be used by tfq.optimizer.rotosolve_minimize.

        Args:
           params [in]: a 1D tf.Tensor.

        Returns:
            Loss function value
        """

        # update the parameters of the model
        start = 0
        for i, size in enumerate(sizes):
            model.trainable_variables[i].assign(
                tf.reshape(params[start:start + size], shape))
            start += size

        # evaluate the loss
        loss_value = loss(model(train_x, training=True), train_y)
        return loss_value

    return func


class RotosolveMinimizerTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the rotosolve optimization algorithm."""

    def test_function_optimization(self):
        """Optimize a simple sinusoid function."""

        n = 10  # Number of parameters to be optimized
        coefficient = tf.random.uniform(shape=[n])
        min_value = -tf.math.reduce_sum(tf.abs(coefficient))

        func = lambda x: tf.math.reduce_sum(tf.sin(x) * coefficient)

        result = rotosolve_minimizer.minimize(func, np.random.random(n))

        self.assertAlmostEqual(func(result.position), min_value)
        self.assertAlmostEqual(result.objective_value, min_value)
        self.assertTrue(result.converged)
        self.assertLess(result.num_iterations,
                        50)  # 50 is the default max iteration

    def test_nonlinear_function_optimization(self):
        """Test to optimize a non-linear function.
        A non-linear function cannot be optimized by rotosolve,
        therefore the optimization must never converge.
        """
        func = lambda x: x[0]**2 + x[1]**2

        result = rotosolve_minimizer.minimize(func,
                                              tf.random.uniform(shape=[2]))

        self.assertFalse(result.converged)
        self.assertEqual(result.num_iterations,
                         50)  # 50 is the default max iteration

    def test_keras_model_optimization(self):
        """Optimizate a PQC based keras model."""

        x = np.asarray([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ], dtype=float)

        y = np.asarray([[-1], [1], [1], [-1]], dtype=np.float32)

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

        # Initial guess of the parameter from random number
        result = rotosolve_minimizer.minimize(
            loss_function_with_model_parameters(model, tf.keras.losses.Hinge(),
                                                x_circ, y),
            tf.random.uniform(shape=[2]) * 2 * np.pi)

        self.assertAlmostEqual(result.objective_value, 0)
        self.assertTrue(result.converged)


if __name__ == "__main__":
    tf.test.main()
