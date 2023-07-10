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
"""Test module for tfq.python.layers.high_level.noisy_pqc layer."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import cirq
import sympy

from tensorflow_quantum.python.layers.high_level import noisy_pqc
from tensorflow_quantum.python import util


class NoisyPQCTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the NoisyPQC layer."""

    def test_noisy_pqc_instantiate(self):
        """Basic creation test."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)
        noisy_pqc.NoisyPQC(learnable_flip,
                           cirq.Z(qubit),
                           repetitions=1000,
                           sample_based=False)

    def test_noisy_pqc_model_circuit_error(self):
        """Test that invalid circuits error properly."""
        qubit = cirq.GridQubit(0, 0)
        no_symbols = cirq.Circuit(cirq.X(qubit))

        with self.assertRaisesRegex(
                TypeError,
                expected_regex="model_circuit must be a cirq.Circuit"):
            noisy_pqc.NoisyPQC('junk',
                               cirq.Z(qubit),
                               repetitions=1000,
                               sample_based=False)

        with self.assertRaisesRegex(
                ValueError,
                expected_regex="model_circuit has no sympy.Symbols"):
            noisy_pqc.NoisyPQC(no_symbols,
                               cirq.Z(qubit),
                               repetitions=1000,
                               sample_based=False)

    def test_noisy_pqc_operators_error(self):
        """Test that invalid operators error properly."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)

        with self.assertRaisesRegex(
                TypeError, expected_regex="cirq.PauliSum or cirq.PauliString"):
            noisy_pqc.NoisyPQC(learnable_flip,
                               'junk',
                               repetitions=1000,
                               sample_based=False)

        with self.assertRaisesRegex(TypeError, expected_regex="Each element"):
            noisy_pqc.NoisyPQC(learnable_flip, [[cirq.Z(qubit)]],
                               repetitions=1000,
                               sample_based=False)

        with self.assertRaisesRegex(TypeError, expected_regex="Each element"):
            noisy_pqc.NoisyPQC(learnable_flip, [cirq.Z(qubit), 'bad'],
                               repetitions=1000,
                               sample_based=False)

    def test_noisy_pqc_repetitions_error(self):
        """Test that invalid repetitions error properly."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="positive integer value"):
            noisy_pqc.NoisyPQC(learnable_flip,
                               cirq.Z(qubit),
                               repetitions='junk',
                               sample_based=False)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="greater than zero."):
            noisy_pqc.NoisyPQC(learnable_flip,
                               cirq.Z(qubit),
                               repetitions=-100,
                               sample_based=False)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="greater than zero."):
            noisy_pqc.NoisyPQC(learnable_flip,
                               cirq.Z(qubit),
                               repetitions=0,
                               sample_based=False)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="must be provided"):
            noisy_pqc.NoisyPQC(learnable_flip,
                               cirq.Z(qubit),
                               sample_based=False)

    def test_noisy_pqc_sample_based_error(self):
        """Test that invalid sampled_based values error properly."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)

        with self.assertRaisesRegex(TypeError, expected_regex="True or False"):
            noisy_pqc.NoisyPQC(learnable_flip,
                               cirq.Z(qubit),
                               repetitions=10,
                               sample_based='junk')

        with self.assertRaisesRegex(
                ValueError, expected_regex="specify sample_based=False"):
            noisy_pqc.NoisyPQC(learnable_flip, cirq.Z(qubit), repetitions=10)

    def test_noisy_pqc_initializer(self):
        """Test action of initializer."""
        (a, b, c) = sympy.symbols("a b c")
        qubit = cirq.GridQubit(0, 0)
        three_parameters = cirq.Circuit(
            [cirq.X(qubit)**a,
             cirq.Y(qubit)**b,
             cirq.Z(qubit)**c])
        mpqc_zeros = noisy_pqc.NoisyPQC(three_parameters,
                                        cirq.Z(qubit),
                                        repetitions=100,
                                        sample_based=False,
                                        initializer='zeros')
        mpqc_ones = noisy_pqc.NoisyPQC(three_parameters,
                                       cirq.Z(qubit),
                                       initializer='ones',
                                       repetitions=100,
                                       sample_based=False)
        self.assertAllEqual([[0, 0, 0]], mpqc_zeros.get_weights())
        self.assertAllEqual([[1, 1, 1]], mpqc_ones.get_weights())

    def test_noisy_pqc_regularizer(self):
        """Test attachment of regularizer to layer."""
        (a, b, c) = sympy.symbols("a b c")
        qubit = cirq.GridQubit(0, 0)
        three_parameters = cirq.Circuit(
            [cirq.X(qubit)**a,
             cirq.Y(qubit)**b,
             cirq.Z(qubit)**c])
        mpqc = noisy_pqc.NoisyPQC(three_parameters,
                                  cirq.Z(qubit),
                                  repetitions=100,
                                  sample_based=False)
        mpqc_r = noisy_pqc.NoisyPQC(three_parameters,
                                    cirq.Z(qubit),
                                    regularizer='l2',
                                    repetitions=100,
                                    sample_based=False)
        self.assertEqual(0, len(mpqc.losses))
        self.assertEqual(1, len(mpqc_r.losses))

    def test_noisy_pqc_constraint(self):
        """Test attachment of constraint to layer."""
        my_constraint = tf.keras.constraints.NonNeg()
        (a, b, c) = sympy.symbols("a b c")
        qubit = cirq.GridQubit(0, 0)
        three_parameters = cirq.Circuit(
            [cirq.X(qubit)**a,
             cirq.Y(qubit)**b,
             cirq.Z(qubit)**c])
        mpqc = noisy_pqc.NoisyPQC(three_parameters,
                                  cirq.Z(qubit),
                                  repetitions=100,
                                  sample_based=False,
                                  constraint=my_constraint)
        self.assertEqual(my_constraint, mpqc.parameters.constraint)

    def test_noisy_pqc_symbols_property(self):
        """Test that the `symbols` property returns the symbols."""
        c, b, a, d = sympy.symbols('c b a d')
        bit = cirq.GridQubit(0, 0)
        test_circuit = cirq.Circuit(
            cirq.H(bit)**a,
            cirq.Z(bit)**b,
            cirq.X(bit)**d,
            cirq.Y(bit)**c)
        layer = noisy_pqc.NoisyPQC(test_circuit,
                                   cirq.Z(bit),
                                   repetitions=100,
                                   sample_based=False)
        self.assertEqual(layer.symbols, [a, b, c, d])

    def test_noisy_pqc_symbol_values(self):
        """Test that PQC symbol_values returns the correct key value pairs."""
        c, b, a, d = sympy.symbols('c b a d')
        bit = cirq.GridQubit(0, 0)
        test_circuit = cirq.Circuit(
            cirq.H(bit)**a,
            cirq.Z(bit)**b,
            cirq.X(bit)**d,
            cirq.Y(bit)**c)
        init_vals = [1, 2, 3, 4]
        layer = noisy_pqc.NoisyPQC(
            test_circuit,
            cirq.Z(bit),
            repetitions=1000,
            sample_based=False,
            initializer=tf.constant_initializer(init_vals))
        expected_vals = dict(zip([a, b, c, d], init_vals))
        self.assertAllClose(layer.symbol_values(), expected_vals)

    @parameterized.parameters([{'sample_based': True}, {'sample_based': False}])
    def test_noisy_pqc_simple_learn(self, sample_based):
        """Test a simple learning scenario using analytic and sample expectation
        on many backends."""
        qubit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(
            cirq.X(qubit)**sympy.Symbol('bit'),
            cirq.depolarize(0.01)(qubit))

        quantum_datum = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        mpqc = noisy_pqc.NoisyPQC(
            circuit,
            cirq.Z(qubit),
            repetitions=5000,
            sample_based=sample_based,
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
