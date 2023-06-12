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
"""Test module for tfq.python.optimizers.spsa_minimizer optimizer."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

from operator import mul
from functools import reduce
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import cirq
import sympy
from tensorflow_quantum.python.layers.high_level import pqc
from tensorflow_quantum.python import util
from tensorflow_quantum.python.optimizers import spsa_minimizer


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
        """A function that can be used by tfq.optimizer.spsa_minimize.

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
        if loss_value.shape != ():
            loss_value = tf.cast(tf.math.reduce_mean(loss_value), tf.float32)
        return loss_value

    return func


class SPSAMinimizerTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the SPSA optimization algorithm."""

    def test_nonlinear_function_optimization(self):
        """Test to optimize a non-linear function.
        """
        func = lambda x: x[0]**2 + x[1]**2

        result = spsa_minimizer.minimize(func, tf.random.uniform(shape=[2]))
        self.assertAlmostEqual(func(result.position).numpy(), 0, delta=1e-4)
        self.assertTrue(result.converged)

    def test_quadratic_function_optimization(self):
        """Test to optimize a sum of quadratic function.
        """
        n = 2
        coefficient = tf.random.uniform(minval=0, maxval=1, shape=[n])
        func = lambda x: tf.math.reduce_sum(np.power(x, 2) * coefficient)

        result = spsa_minimizer.minimize(func, tf.random.uniform(shape=[n]))
        self.assertAlmostEqual(func(result.position).numpy(), 0, delta=2e-4)
        self.assertTrue(result.converged)

    def test_noisy_sin_function_optimization(self):
        """Test noisy ssinusoidal function
        """
        n = 10
        func = lambda x: tf.math.reduce_sum(
            tf.math.sin(x) + tf.random.uniform(
                minval=-0.1, maxval=0.1, shape=[n]))

        result = spsa_minimizer.minimize(func, tf.random.uniform(shape=[n]))
        self.assertLessEqual(func(result.position).numpy(), -n + 0.1 * n)

    def test_failure_optimization(self):
        """Test a function that is completely random and cannot be minimized
        """
        n = 100
        func = lambda x: np.random.uniform(-10, 10, 1)[0]
        it = 50

        result = spsa_minimizer.minimize(func,
                                         tf.random.uniform(shape=[n]),
                                         max_iterations=it)
        self.assertFalse(result.converged)
        self.assertEqual(result.num_iterations, it)

    def test_blocking(self):
        """Test the blocking functionality.
        """
        n = 10
        it = 50

        init = 1
        self.incr = 0

        def block_func1(params):
            self.incr += init
            return self.incr

        result = spsa_minimizer.minimize(block_func1,
                                         tf.random.uniform(shape=[n]),
                                         blocking=True,
                                         allowed_increase=0.5,
                                         max_iterations=it)
        self.assertFalse(result.converged)
        self.assertEqual(result.num_iterations, it)
        self.assertEqual(result.objective_value,
                         init * 4)  # function executd 3 (in step) +
        # 1 (initial evaluation) times

        init = 1 / 6 * 0.49
        self.incr = 0

        def block_func2(params):
            self.incr += init
            return self.incr

        result = spsa_minimizer.minimize(block_func2,
                                         tf.random.uniform(shape=[n]),
                                         blocking=True,
                                         allowed_increase=0.5,
                                         max_iterations=it)
        self.assertFalse(result.converged)
        self.assertEqual(result.num_iterations, it)
        self.assertEqual(result.objective_value, init * 3 * it + init)

    def test_3_qubit_circuit(self):
        """Test quantum circuit optimization, adapted from Qiskit SPSA testing
        https://github.com/Qiskit/qiskit-terra/blob/main/test/python/algorithms/optimizers/test_spsa.py#L37
        """
        qubits = [cirq.GridQubit(0, i) for i in range(3)]
        params = sympy.symbols("q0:9")
        circuit = cirq.Circuit()
        for i in qubits:
            circuit += cirq.ry(np.pi / 4).on(i)
        circuit += cirq.ry(params[0]).on(qubits[0])
        circuit += cirq.ry(params[1]).on(qubits[1])
        circuit += cirq.rz(params[2]).on(qubits[2])

        circuit += cirq.CZ(qubits[0], qubits[1])
        circuit += cirq.CZ(qubits[1], qubits[2])
        circuit += cirq.rz(params[3]).on(qubits[0])
        circuit += cirq.rz(params[4]).on(qubits[1])
        circuit += cirq.rx(params[5]).on(qubits[2])

        # Build the Keras model.
        model = tf.keras.Sequential([
            # The input is the data-circuit, encoded as a tf.string
            tf.keras.layers.Input(shape=(), dtype=tf.string),
            # The PQC layer returns the expected value of the
            # readout gate, range [-1,1].
            pqc.PQC(circuit,
                    cirq.Z(qubits[0]) * cirq.Z(qubits[1]),
                    repetitions=1024),
        ])

        initial_point = np.array([
            0.82311034, 0.02611798, 0.21077064, 0.61842177, 0.09828447,
            0.62013131
        ])

        result = spsa_minimizer.minimize(loss_function_with_model_parameters(
            model, lambda x, y: x[0][0],
            util.convert_to_tensor([cirq.Circuit()]), None),
                                         initial_point,
                                         max_iterations=100)

        self.assertTrue(result.converged)
        self.assertLess(result.objective_value.numpy(), -0.95)

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
            pqc.PQC(circuit, 2 * cirq.Z(q1)),
        ])

        # Initial guess of the parameter from random number
        result = spsa_minimizer.minimize(
            loss_function_with_model_parameters(model, tf.keras.losses.Hinge(),
                                                x_circ, y),
            tf.random.uniform(shape=[2]) * 2 * np.pi)

        self.assertAlmostEqual(result.objective_value.numpy(), 0)
        self.assertTrue(result.converged)


if __name__ == "__main__":
    tf.test.main()
