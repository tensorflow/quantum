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
"""Tests for tensorflow_quantum.layers.circuit_executors.expectation."""
import numpy as np
import sympy
import tensorflow as tf

import cirq
from tensorflow_quantum.python.layers.circuit_executors import expectation
from tensorflow_quantum.python.differentiators import linear_combination
from tensorflow_quantum.python import util


def _gen_single_bit_rotation_problem(bit, symbols):
    """Generate a toy problem on 1 qubit."""
    starting_state = np.random.uniform(0, 2 * np.pi, 3)
    circuit = cirq.Circuit(
        cirq.Rx(starting_state[0])(bit),
        cirq.Ry(starting_state[1])(bit),
        cirq.Rz(starting_state[2])(bit),
        cirq.Rz(symbols[2])(bit),
        cirq.Ry(symbols[1])(bit),
        cirq.Rx(symbols[0])(bit))

    return circuit


class ExpectationTest(tf.test.TestCase):
    """Basic tests for the expectation layer."""

    def test_expectation_instantiate(self):
        """Test that Expectation instantiates correctly."""
        expectation.Expectation()
        expectation.Expectation(backend=cirq.Simulator())
        expectation.Expectation(
            differentiator=linear_combination.ForwardDifference())

    def test_expectation_instantiate_error(self):
        """Test that Expectation errors with bad inputs."""

        class MySampler(cirq.Sampler):
            """Class to test sampler detection in Expectation."""

            def run_sweep(self):
                """do nothing."""
                return

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="SampledExpectation"):
            expectation.Expectation(backend=MySampler())

        with self.assertRaisesRegex(
                TypeError, expected_regex="SimulatesFinalState or None"):
            expectation.Expectation(backend='junk')

        with self.assertRaisesRegex(
                TypeError, expected_regex="tfq.differentiators.Differentiator"):
            expectation.Expectation(differentiator='junk')

    def test_expectation_type_inputs_error(self):
        """Test that expectation errors within Keras call."""

        bit = cirq.GridQubit(0, 0)
        symbol = sympy.Symbol('alpha')
        test_pstring = cirq.Z(bit)
        test_psum = cirq.PauliSum.from_pauli_strings([test_pstring])
        symb_circuit = cirq.Circuit(cirq.H(bit)**symbol)
        reg_circuit = cirq.Circuit(cirq.H(bit))

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="string or sympy.Symbol"):
            expectation.Expectation()(symb_circuit,
                                      symbol_names=[symbol, 5.0],
                                      operators=test_psum)

        with self.assertRaisesRegex(ValueError,
                                    expected_regex="must be unique."):
            expectation.Expectation()(symb_circuit,
                                      symbol_names=[symbol, symbol],
                                      operators=test_psum)

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cannot be parsed"):
            expectation.Expectation()(symb_circuit,
                                      symbol_names='junk',
                                      operators=test_psum)

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cannot be parsed"):
            expectation.Expectation()(symb_circuit,
                                      symbol_names=[symbol],
                                      symbol_values='junk',
                                      operators=test_psum)

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cannot be parsed"):
            expectation.Expectation()('junk',
                                      symbol_names=[symbol],
                                      symbol_values=[[0.5]],
                                      operators=test_psum)

        with self.assertRaisesRegex(RuntimeError,
                                    expected_regex="operators not provided"):
            expectation.Expectation()(symb_circuit,
                                      symbol_names=[symbol],
                                      symbol_values=[[0.5]])

        with self.assertRaisesRegex(Exception,
                                    expected_regex="Unknown initializer"):
            expectation.Expectation()(reg_circuit,
                                      operators=test_psum,
                                      initializer='junk')

    def test_expectation_op_error(self):
        """Test that expectation errors within underlying ops correctly."""

        bit = cirq.GridQubit(0, 0)
        symbol = sympy.Symbol('alpha')
        test_pstring = cirq.Z(bit)
        test_psum = cirq.PauliSum.from_pauli_strings([test_pstring])
        symb_circuit = cirq.Circuit(cirq.H(bit)**symbol)
        reg_circuit = cirq.Circuit(cirq.H(bit))

        with self.assertRaisesRegex(Exception,
                                    expected_regex="Could not find symbol"):
            # No symbol matchups.
            expectation.Expectation()([symb_circuit], operators=test_psum)

        with self.assertRaisesRegex(Exception,
                                    expected_regex="Unparseable proto"):
            # Proto is unparseable.
            expectation.Expectation()([reg_circuit],
                                      operators=tf.convert_to_tensor(
                                          [['bad_operator']]))

        with self.assertRaisesRegex(Exception, expected_regex="rank 2"):
            # Operators has wrong rank.
            expectation.Expectation()([reg_circuit],
                                      operators=util.convert_to_tensor(
                                          [test_psum]))

        with self.assertRaisesRegex(Exception, expected_regex="rank 2"):
            # symbol_values has wrong rank.
            expectation.Expectation()([symb_circuit],
                                      symbol_names=[symbol],
                                      symbol_values=[0.5],
                                      operators=test_psum)

        with self.assertRaisesRegex(Exception, expected_regex="do not match."):
            # Wrong batch size for pauli operators.
            expectation.Expectation()(symb_circuit,
                                      symbol_names=[symbol],
                                      operators=[[test_psum], [test_psum]])

    def test_static_cases(self):
        """Run inputs through in complex cases."""

        bit = cirq.GridQubit(0, 0)
        symbol = sympy.Symbol('alpha')
        test_pstring = cirq.Z(bit)
        test_psum = cirq.PauliSum.from_pauli_strings([test_pstring])
        symb_circuit = cirq.Circuit(cirq.H(bit)**symbol)
        reg_circuit = cirq.Circuit(cirq.H(bit))

        # Passing a 2d operators input requires a 1d circuit input.
        expectation.Expectation()([reg_circuit, reg_circuit],
                                  operators=[[test_psum, test_psum],
                                             [test_psum, test_psum]])

        # Passing 2d operators along with other inputs.
        expectation.Expectation()([symb_circuit, symb_circuit],
                                  symbol_names=[symbol],
                                  operators=[[test_psum, test_psum],
                                             [test_psum, test_psum]])
        expectation.Expectation()([symb_circuit, symb_circuit],
                                  symbol_names=[symbol],
                                  symbol_values=[[0.5], [0.8]],
                                  operators=[[test_psum, test_psum],
                                             [test_psum, test_psum]])

        # Ensure tiling up of circuits works as expected.
        expectation.Expectation()(reg_circuit, operators=test_psum)
        expectation.Expectation()(reg_circuit, operators=[test_psum, test_psum])

        # Ensure tiling up of symbol_values works as expected.
        expectation.Expectation()(symb_circuit,
                                  symbol_names=[symbol],
                                  symbol_values=[[0.5], [0.8]],
                                  operators=test_psum)
        expectation.Expectation()(symb_circuit,
                                  symbol_names=[symbol],
                                  symbol_values=[[0.5]],
                                  operators=test_psum)

    def test_expectation_simple_tf_train(self):
        """Train a layer using standard tf (not keras).
        This is a subtle test that will work since we don't use keras compile.
        """
        bit = cirq.GridQubit(0, 0)
        circuit = \
            cirq.Circuit(cirq.Rx(sympy.Symbol('theta'))(bit))
        op = cirq.Z(bit)
        layer = expectation.Expectation()
        optimizer = tf.optimizers.Adam(learning_rate=0.05)
        for _ in range(200):
            with tf.GradientTape() as tape:
                circuit_out = layer(circuit,
                                    symbol_names=['theta'],
                                    operators=op)
                mse = tf.square(tf.reduce_sum(tf.subtract(circuit_out, -1)))
            grads = tape.gradient(mse, layer.trainable_weights)
            optimizer.apply_gradients(zip(grads, layer.trainable_weights))
        self.assertAllClose(mse.numpy(), 0, atol=1e-3)
