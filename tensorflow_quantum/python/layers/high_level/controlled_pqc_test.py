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
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import cirq
import sympy

from tensorflow_quantum.python.layers.high_level import controlled_pqc
from tensorflow_quantum.python import util


class ControlledPQCTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the ControlledPQC layer."""

    def test_controlled_pqc_instantiate(self):
        """Basic creation test."""
        symbol = sympy.Symbol('alpha')
        bit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(bit)**symbol)
        controlled_pqc.ControlledPQC(learnable_flip, cirq.Z(bit))
        controlled_pqc.ControlledPQC(learnable_flip,
                                     cirq.Z(bit),
                                     repetitions=500)

    def test_controlled_pqc_backend_error(self):
        """Test that invalid backends error properly."""
        symbol = sympy.Symbol('alpha')
        bit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(bit)**symbol)

        class MyState(cirq.SimulatesFinalState):
            """My state simulator."""

            def simulate_sweep(self):
                """do nothing."""
                return

        class MySample(cirq.Sampler):
            """My state simulator."""

            def run_sweep(self):
                """do nothing."""
                return

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cirq.SimulatesFinalState"):
            controlled_pqc.ControlledPQC(learnable_flip,
                                         cirq.Z(bit),
                                         backend='junk')

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cirq.SimulatesFinalState"):
            controlled_pqc.ControlledPQC(learnable_flip,
                                         cirq.Z(bit),
                                         repetitions=None,
                                         backend=MySample)

        with self.assertRaisesRegex(TypeError, expected_regex="cirq.Sampler"):
            controlled_pqc.ControlledPQC(learnable_flip,
                                         cirq.Z(bit),
                                         repetitions=500,
                                         backend=MyState)

    def test_controlled_pqc_model_circuit_error(self):
        """Test that invalid circuits error properly."""
        bit = cirq.GridQubit(0, 0)
        no_symbols = cirq.Circuit(cirq.X(bit))

        with self.assertRaisesRegex(TypeError, expected_regex="cirq.Circuit"):
            controlled_pqc.ControlledPQC('junk', cirq.Z(bit))

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="no sympy.Symbols"):
            controlled_pqc.ControlledPQC(no_symbols, cirq.Z(bit))

    def test_controlled_pqc_operators_error(self):
        """Test that invalid operators error properly."""
        symbol = sympy.Symbol('alpha')
        bit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(bit)**symbol)

        with self.assertRaisesRegex(
                TypeError, expected_regex="cirq.PauliSum or cirq.PauliString"):
            controlled_pqc.ControlledPQC(learnable_flip, 'junk')

        with self.assertRaisesRegex(TypeError, expected_regex="Each element"):
            controlled_pqc.ControlledPQC(learnable_flip, [[cirq.Z(bit)]])

        with self.assertRaisesRegex(TypeError, expected_regex="Each element"):
            controlled_pqc.ControlledPQC(learnable_flip, [cirq.Z(bit), 'bad'])

    def test_controlled_pqc_repetitions_error(self):
        """Test that invalid repetitions error properly."""
        symbol = sympy.Symbol('alpha')
        bit = cirq.GridQubit(0, 0)
        learnable_flip = cirq.Circuit(cirq.X(bit)**symbol)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="greater than zero."):
            controlled_pqc.ControlledPQC(learnable_flip,
                                         cirq.Z(bit),
                                         repetitions=-100)

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="positive integer value"):
            controlled_pqc.ControlledPQC(learnable_flip,
                                         cirq.Z(bit),
                                         repetitions='junk')

    def test_controlled_pqc_symbols_property(self):
        """Test that the `symbols` property returns the symbols."""
        c, b, a, d = sympy.symbols('c b a d')
        bit = cirq.GridQubit(0, 0)
        test_circuit = cirq.Circuit(
            cirq.H(bit)**a,
            cirq.Z(bit)**b,
            cirq.X(bit)**d,
            cirq.Y(bit)**c)
        layer = controlled_pqc.ControlledPQC(test_circuit, cirq.Z(bit))
        self.assertEqual(layer.symbols, [a, b, c, d])

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(repetitions=[None, 5000],
                                          backend=[None, cirq.Simulator()])))
    def test_controlled_pqc_simple_learn(self, backend, repetitions):
        """Test a simple learning scenario using analytic and sample expectation
        on many backends."""
        bit = cirq.GridQubit(0, 0)
        circuit = \
            cirq.Circuit(cirq.Rx(sympy.Symbol('theta'))(bit))

        inputs = tf.keras.Input(shape=(1,), dtype=tf.dtypes.float32)
        quantum_datum = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        l1 = tf.keras.layers.Dense(10)(inputs)
        l2 = tf.keras.layers.Dense(1)(l1)
        outputs = controlled_pqc.ControlledPQC(circuit,
                                               cirq.Z(bit),
                                               repetitions=repetitions,
                                               backend=backend)(
                                                   [quantum_datum, l2])
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
