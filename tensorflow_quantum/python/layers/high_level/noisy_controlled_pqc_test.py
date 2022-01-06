# Copyright 2019 The TensorFlow Quantum Authors. All Rights Reserved.
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
"""Test module for tfq.python.layers.high_level.controlled_pqc layer."""
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

from tensorflow_quantum.python.layers.high_level import noisy_controlled_pqc
from tensorflow_quantum.python import util


class NoisyControlledPQCTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the NoisyControlledPQC layer."""

    def test_controlled_pqc_instantiate(self):
        """Basic creation test."""
        symbol = sympy.Symbol('alpha')
        bit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(bit)**symbol)
        noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                cirq.Z(bit),
                                                repetitions=500,
                                                sample_based=False)

    def test_controlled_pqc_model_circuit_error(self):
        """Test that invalid circuits error properly."""
        bit = cirq.GridQubit(0, 0)
        no_symbols = cirq.Circuit(cirq.X(bit))

        with self.assertRaisesRegex(TypeError, expected_regex="cirq.Circuit"):
            noisy_controlled_pqc.NoisyControlledPQC('junk',
                                                    cirq.Z(bit),
                                                    repetitions=500,
                                                    sample_based=False)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="no sympy.Symbols"):
            noisy_controlled_pqc.NoisyControlledPQC(no_symbols,
                                                    cirq.Z(bit),
                                                    repetitions=500,
                                                    sample_based=False)

    def test_controlled_pqc_operators_error(self):
        """Test that invalid operators error properly."""
        symbol = sympy.Symbol('alpha')
        bit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(bit)**symbol)

        with self.assertRaisesRegex(
                TypeError, expected_regex="cirq.PauliSum or cirq.PauliString"):
            noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                    'junk',
                                                    repetitions=500,
                                                    sample_based=False)

        with self.assertRaisesRegex(TypeError, expected_regex="Each element"):
            noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                    [[cirq.Z(bit)]],
                                                    repetitions=500,
                                                    sample_based=False)

        with self.assertRaisesRegex(TypeError, expected_regex="Each element"):
            noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                    [cirq.Z(bit), 'bad'],
                                                    repetitions=500,
                                                    sample_based=False)

    def test_controlled_pqc_repetitions_error(self):
        """Test that invalid repetitions error properly."""
        symbol = sympy.Symbol('alpha')
        bit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(bit)**symbol)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="greater than zero."):
            noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                    cirq.Z(bit),
                                                    repetitions=-100,
                                                    sample_based=False)

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="positive integer value"):
            noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                    cirq.Z(bit),
                                                    repetitions='junk',
                                                    sample_based=False)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="must be provided"):
            noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                    cirq.Z(bit),
                                                    sample_based=False)

    def test_noisy_controlled_pqc_sample_based_error(self):
        """Test that invalid sampled_based values error properly."""
        symbol = sympy.Symbol('alpha')
        qubit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(qubit)**symbol)

        with self.assertRaisesRegex(TypeError, expected_regex="True or False"):
            noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                    cirq.Z(qubit),
                                                    repetitions=10,
                                                    sample_based='junk')

        with self.assertRaisesRegex(
                ValueError, expected_regex="specify sample_based=False"):
            noisy_controlled_pqc.NoisyControlledPQC(learnable_flip,
                                                    cirq.Z(qubit),
                                                    repetitions=10)

    def test_controlled_pqc_symbols_property(self):
        """Test that the `symbols` property returns the symbols."""
        c, b, a, d = sympy.symbols('c b a d')
        bit = cirq.GridQubit(0, 0)
        test_circuit = cirq.Circuit(
            cirq.H(bit)**a,
            cirq.Z(bit)**b,
            cirq.X(bit)**d,
            cirq.Y(bit)**c)
        layer = noisy_controlled_pqc.NoisyControlledPQC(test_circuit,
                                                        cirq.Z(bit),
                                                        repetitions=100,
                                                        sample_based=False)
        self.assertEqual(layer.symbols, [a, b, c, d])

    @parameterized.parameters([{'sample_based': True, 'sample_based': False}])
    def test_controlled_pqc_simple_learn(self, sample_based):
        """Test a simple learning scenario using analytic and sample expectation
        on many backends."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(
            cirq.rx(sympy.Symbol('theta'))(bit),
            cirq.depolarize(0.01)(bit))

        inputs = tf.keras.Input(shape=(1,), dtype=tf.dtypes.float32)
        quantum_datum = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        l1 = tf.keras.layers.Dense(10)(inputs)
        l2 = tf.keras.layers.Dense(1)(l1)
        outputs = noisy_controlled_pqc.NoisyControlledPQC(
            circuit, cirq.Z(bit), repetitions=5000,
            sample_based=sample_based)([quantum_datum, l2])
        model = tf.keras.Model(inputs=[quantum_datum, inputs], outputs=outputs)

        data_in = np.array([[1], [0]], dtype=np.float32)
        data_out = np.array([[1], [-1]], dtype=np.float32)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                      loss=tf.keras.losses.mean_squared_error)

        data_circuits = util.convert_to_tensor(
            [cirq.Circuit(cirq.X(bit)),
             cirq.Circuit()])

        history = model.fit(x=[data_circuits, data_in], y=data_out, epochs=30)
        self.assertAllClose(history.history['loss'][-1], 0, atol=1e-1)


if __name__ == "__main__":
    tf.test.main()
