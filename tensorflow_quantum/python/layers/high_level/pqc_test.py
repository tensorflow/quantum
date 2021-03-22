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
"""Test module for tfq.python.layers.high_level.pqc layer."""
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import cirq
import sympy

from tensorflow_quantum.python.layers.high_level import pqc
from tensorflow_quantum.python import util


class PQCTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the PQC layer."""

    def test_pqc_instantiate(self):
        """Basic creation test."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)
        pqc.PQC(learnable_flip, cirq.Z(qubit))
        pqc.PQC(learnable_flip, cirq.Z(qubit), repetitions=500)

    def test_pqc_model_circuit_error(self):
        """Test that invalid circuits error properly."""
        qubit = cirq.GridQubit(0, 0)
        no_symbols = cirq.Circuit(cirq.X(qubit))

        with self.assertRaisesRegex(
                TypeError,
                expected_regex="model_circuit must be a cirq.Circuit"):
            pqc.PQC('junk', cirq.Z(qubit))

        with self.assertRaisesRegex(
                ValueError,
                expected_regex="model_circuit has no sympy.Symbols"):
            pqc.PQC(no_symbols, cirq.Z(qubit))

    def test_pqc_operators_error(self):
        """Test that invalid operators error properly."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)

        with self.assertRaisesRegex(
                TypeError, expected_regex="cirq.PauliSum or cirq.PauliString"):
            pqc.PQC(learnable_flip, 'junk')

        with self.assertRaisesRegex(TypeError, expected_regex="Each element"):
            pqc.PQC(learnable_flip, [[cirq.Z(qubit)]])

        with self.assertRaisesRegex(TypeError, expected_regex="Each element"):
            pqc.PQC(learnable_flip, [cirq.Z(qubit), 'bad'])

    def test_pqc_repetitions_error(self):
        """Test that invalid repetitions error properly."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="positive integer value"):
            pqc.PQC(learnable_flip, cirq.Z(qubit), repetitions='junk')

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="greater than zero."):
            pqc.PQC(learnable_flip, cirq.Z(qubit), repetitions=-100)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="greater than zero."):
            pqc.PQC(learnable_flip, cirq.Z(qubit), repetitions=0)

    def test_pqc_backend_error(self):
        """Test that invalid backends error properly."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)

        class MyExpectation(cirq.SimulatesExpectationValues):
            """My expectation values simulator."""

            def simulate_expectation_values_sweep(self):
                """do nothing."""
                return

        class MySample(cirq.Sampler):
            """My state simulator."""

            def run_sweep(self):
                """do nothing."""
                return

        with self.assertRaisesRegex(TypeError, expected_regex="cirq.Sampler"):
            pqc.PQC(learnable_flip,
                    cirq.Z(qubit),
                    backend=MyExpectation,
                    repetitions=500)

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cirq.SimulatesExpectation"):
            pqc.PQC(learnable_flip,
                    cirq.Z(qubit),
                    backend=MySample,
                    repetitions=None)

    def test_pqc_initializer(self):
        """Test action of initializer."""
        (a, b, c) = sympy.symbols("a b c")
        qubit = cirq.GridQubit(0, 0)
        three_parameters = cirq.Circuit(
            [cirq.X(qubit)**a,
             cirq.Y(qubit)**b,
             cirq.Z(qubit)**c])
        mpqc_zeros = pqc.PQC(three_parameters,
                             cirq.Z(qubit),
                             initializer='zeros')
        mpqc_ones = pqc.PQC(three_parameters, cirq.Z(qubit), initializer='ones')
        self.assertAllEqual([[0, 0, 0]], mpqc_zeros.get_weights())
        self.assertAllEqual([[1, 1, 1]], mpqc_ones.get_weights())

    def test_pqc_regularizer(self):
        """Test attachment of regularizer to layer."""
        (a, b, c) = sympy.symbols("a b c")
        qubit = cirq.GridQubit(0, 0)
        three_parameters = cirq.Circuit(
            [cirq.X(qubit)**a,
             cirq.Y(qubit)**b,
             cirq.Z(qubit)**c])
        mpqc = pqc.PQC(three_parameters, cirq.Z(qubit))
        mpqc_r = pqc.PQC(three_parameters, cirq.Z(qubit), regularizer='l2')
        self.assertEqual(0, len(mpqc.losses))
        self.assertEqual(1, len(mpqc_r.losses))

    def test_pqc_constraint(self):
        """Test attachment of constraint to layer."""
        my_constraint = tf.keras.constraints.NonNeg()
        (a, b, c) = sympy.symbols("a b c")
        qubit = cirq.GridQubit(0, 0)
        three_parameters = cirq.Circuit(
            [cirq.X(qubit)**a,
             cirq.Y(qubit)**b,
             cirq.Z(qubit)**c])
        mpqc = pqc.PQC(three_parameters,
                       cirq.Z(qubit),
                       constraint=my_constraint)
        self.assertEqual(my_constraint, mpqc.parameters.constraint)

    def test_pqc_symbols_property(self):
        """Test that the `symbols` property returns the symbols."""
        c, b, a, d = sympy.symbols('c b a d')
        bit = cirq.GridQubit(0, 0)
        test_circuit = cirq.Circuit(
            cirq.H(bit)**a,
            cirq.Z(bit)**b,
            cirq.X(bit)**d,
            cirq.Y(bit)**c)
        layer = pqc.PQC(test_circuit, cirq.Z(bit))
        self.assertEqual(layer.symbols, [a, b, c, d])

    def test_pqc_symbol_values(self):
        """Test that PQC symbol_values returns the correct key value pairs."""
        c, b, a, d = sympy.symbols('c b a d')
        bit = cirq.GridQubit(0, 0)
        test_circuit = cirq.Circuit(
            cirq.H(bit)**a,
            cirq.Z(bit)**b,
            cirq.X(bit)**d,
            cirq.Y(bit)**c)
        init_vals = [1, 2, 3, 4]
        layer = pqc.PQC(test_circuit,
                        cirq.Z(bit),
                        initializer=tf.constant_initializer(init_vals))
        expected_vals = dict(zip([a, b, c, d], init_vals))
        self.assertAllClose(layer.symbol_values(), expected_vals)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(backend=[None, cirq.Simulator()],
                                          repetitions=[None, 5000])))
    def test_pqc_simple_learn(self, backend, repetitions):
        """Test a simple learning scenario using analytic and sample expectation
        on many backends."""
        qubit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(cirq.X(qubit)**sympy.Symbol('bit'))

        quantum_datum = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        mpqc = pqc.PQC(circuit,
                       cirq.Z(qubit),
                       backend=backend,
                       repetitions=repetitions,
                       initializer=tf.keras.initializers.Constant(value=0.5))
        outputs = mpqc(quantum_datum)
        model = tf.keras.Model(inputs=quantum_datum, outputs=outputs)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
                      loss=tf.keras.losses.mean_squared_error)

        data_circuits = util.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubit)),
             cirq.Circuit()])
        print(data_circuits)
        data_out = np.array([[1], [-1]], dtype=np.float32)

        # Model should learn to flip the qubit
        self.assertNear(mpqc.get_weights()[0][0], 0.5, 1e-1)
        history = model.fit(x=data_circuits, y=data_out, epochs=40)
        self.assertAllClose(history.history['loss'][-1], 0, atol=1e-1)
        self.assertNear(mpqc.get_weights()[0][0], 1, 1e-1)


if __name__ == "__main__":
    tf.test.main()
