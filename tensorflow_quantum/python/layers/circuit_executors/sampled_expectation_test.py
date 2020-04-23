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
"""Tests for tensorflow_quantum.layers.circuit_executors.sampled_expectation."""

import numpy as np
import sympy
import tensorflow as tf

import cirq
from tensorflow_quantum.python.layers.circuit_executors import \
    sampled_expectation
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


class SampledExpectationTest(tf.test.TestCase):
    """Basic tests for the SampledExpectation layer."""

    def test_sampled_expectation_symbol_input(self):
        """Test that SampledExpectation only accepts valid permutations of
        symbols."""
        sampled_expectation.SampledExpectation()
        sampled_expectation.SampledExpectation(backend=cirq.Simulator())
        sampled_expectation.SampledExpectation(
            differentiator=linear_combination.ForwardDifference())

    def test_sampled_expectation_instantiate_error(self):
        """Test that SampledExpectation errors with bad inputs."""

        class MySim(cirq.SimulatesFinalState):
            """Class to test sampler detection in Expectation."""

            def simulate_sweep(self):
                """Do nothing."""
                return

        with self.assertRaisesRegex(TypeError, expected_regex="Expectation"):
            sampled_expectation.SampledExpectation(backend=MySim())

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="Sampler or None"):
            sampled_expectation.SampledExpectation(backend='junk')

        with self.assertRaisesRegex(
                TypeError, expected_regex="tfq.differentiators.Differentiator"):
            sampled_expectation.SampledExpectation(differentiator='junk')

    def test_sampled_expectation_type_inputs_error(self):
        """Test that SampledExpectation errors within Keras call."""

        bit = cirq.GridQubit(0, 0)
        symbol = sympy.Symbol('alpha')
        test_pstring = cirq.Z(bit)
        test_psum = cirq.PauliSum.from_pauli_strings([test_pstring])
        symb_circuit = cirq.Circuit(cirq.H(bit)**symbol)
        reg_circuit = cirq.Circuit(cirq.H(bit))

        with self.assertRaisesRegex(RuntimeError,
                                    expected_regex="operators not provided"):
            sampled_expectation.SampledExpectation()(symb_circuit,
                                                     symbol_names=[symbol],
                                                     symbol_values=[[0.5]],
                                                     repetitions=1)

        with self.assertRaisesRegex(RuntimeError,
                                    expected_regex="repetitions not provided"):
            sampled_expectation.SampledExpectation()(symb_circuit,
                                                     symbol_names=[symbol],
                                                     symbol_values=[[0.5]],
                                                     operators=test_psum)

        with self.assertRaisesRegex(Exception,
                                    expected_regex="Unknown initializer"):
            sampled_expectation.SampledExpectation()(reg_circuit,
                                                     operators=test_psum,
                                                     initializer='junk',
                                                     repetitions=1)

        with self.assertRaisesRegex(Exception,
                                    expected_regex="cannot be parsed"):
            sampled_expectation.SampledExpectation()(reg_circuit,
                                                     operators=test_psum,
                                                     repetitions='junk')

    def test_sampled_expectation_op_error(self):
        """Test that expectation errors within underlying ops correctly."""
        # Note the expected_regex is left blank here since there is a
        # discrepancy between the error strings provided between backends.
        bit = cirq.GridQubit(0, 0)
        symbol = sympy.Symbol('alpha')
        test_pstring = cirq.Z(bit)
        test_psum = cirq.PauliSum.from_pauli_strings([test_pstring])
        symb_circuit = cirq.Circuit(cirq.H(bit)**symbol)
        reg_circuit = cirq.Circuit(cirq.H(bit))

        with self.assertRaisesRegex(Exception, expected_regex="bytes-like"):
            # Operators has wrong rank. Parse error.
            sampled_expectation.SampledExpectation()(
                [reg_circuit],
                operators=util.convert_to_tensor([test_psum]),
                repetitions=1)

        with self.assertRaisesRegex(
                Exception, expected_regex="must match second dimension"):
            # symbol_values has wrong rank.
            sampled_expectation.SampledExpectation()([symb_circuit],
                                                     symbol_names=[symbol],
                                                     symbol_values=[0.5],
                                                     operators=test_psum,
                                                     repetitions=1)

        with self.assertRaisesRegex(Exception, expected_regex="same batch"):
            # Wrong batch size for pauli operators.
            sampled_expectation.SampledExpectation()(symb_circuit,
                                                     symbol_names=[symbol],
                                                     operators=[[test_psum],
                                                                [test_psum]],
                                                     repetitions=1)

        with self.assertRaisesRegex(Exception, expected_regex="same batch"):
            # Wrong batch size for pauli operators.
            sampled_expectation.SampledExpectation()(reg_circuit,
                                                     operators=[[test_psum],
                                                                [test_psum]],
                                                     repetitions=1)

        with self.assertRaisesRegex(Exception, expected_regex="<= 0"):
            # Wrong repetitions.
            sampled_expectation.SampledExpectation()(reg_circuit,
                                                     operators=test_psum,
                                                     repetitions=-1)

        with self.assertRaisesRegex(Exception,
                                    expected_regex="same shape as pauli_sums"):
            # Wrong second dimension size for repetitions & pauli operators.
            sampled_expectation.SampledExpectation()(reg_circuit,
                                                     operators=test_psum,
                                                     repetitions=[5, 4, 3])

    def test_static_cases(self):
        """Run inputs through in complex cases."""

        bit = cirq.GridQubit(0, 0)
        symbol = sympy.Symbol('alpha')
        test_pstring = cirq.Z(bit)
        test_psum = cirq.PauliSum.from_pauli_strings([test_pstring])
        symb_circuit = cirq.Circuit(cirq.H(bit)**symbol)
        reg_circuit = cirq.Circuit(cirq.H(bit))

        # Passing a 2d operators input requires a 1d circuit input.
        sampled_expectation.SampledExpectation()(
            [reg_circuit, reg_circuit],
            operators=[[test_psum, test_psum], [test_psum, test_psum]],
            repetitions=1)

        # Passing 2d operators along with other inputs.
        sampled_expectation.SampledExpectation()(
            [symb_circuit, symb_circuit],
            symbol_names=[symbol],
            operators=[[test_psum, test_psum], [test_psum, test_psum]],
            repetitions=1)
        sampled_expectation.SampledExpectation()(
            [symb_circuit, symb_circuit],
            symbol_names=[symbol],
            symbol_values=[[0.5], [0.8]],
            operators=[[test_psum, test_psum], [test_psum, test_psum]],
            repetitions=1)

        # Ensure tiling up of circuits works as expected.
        sampled_expectation.SampledExpectation()(reg_circuit,
                                                 operators=test_psum,
                                                 repetitions=1)
        sampled_expectation.SampledExpectation()(
            reg_circuit, operators=[test_psum, test_psum], repetitions=1)

        # Ensure tiling up of symbol_values works as expected.
        sampled_expectation.SampledExpectation()(symb_circuit,
                                                 symbol_names=[symbol],
                                                 symbol_values=[[0.5], [0.8]],
                                                 operators=test_psum,
                                                 repetitions=1)
        sampled_expectation.SampledExpectation()(symb_circuit,
                                                 symbol_names=[symbol],
                                                 symbol_values=[[0.5]],
                                                 operators=test_psum,
                                                 repetitions=1)

        # Test multiple operators with integer valued repetition.
        sampled_expectation.SampledExpectation()(
            symb_circuit,
            symbol_names=[symbol],
            symbol_values=[[0.5]],
            operators=[-1.0 * cirq.Z(bit),
                       cirq.X(bit) + 2.0 * cirq.Z(bit)],
            repetitions=1)
        sampled_expectation.SampledExpectation()(
            symb_circuit,
            symbol_names=[symbol],
            symbol_values=[[0.5]],
            operators=[-1.0 * cirq.Z(bit),
                       cirq.X(bit) + 2.0 * cirq.Z(bit)],
            repetitions=[5, 1])

    def test_sampled_expectation_simple_tf_train(self):
        """Train a layer using standard tf (not keras)."""
        bit = cirq.GridQubit(0, 0)
        circuit = cirq.Circuit(cirq.Rx(sympy.Symbol('theta'))(bit))
        layer = sampled_expectation.SampledExpectation()
        optimizer = tf.optimizers.Adam(learning_rate=0.05)
        for _ in range(10):
            with tf.GradientTape() as tape:
                circuit_out = layer(circuit,
                                    symbol_names=['theta'],
                                    operators=cirq.Z(bit),
                                    repetitions=100)
                mse = tf.square(tf.reduce_sum(tf.subtract(circuit_out, -1)))
            grads = tape.gradient(mse, layer.trainable_weights)
            optimizer.apply_gradients(zip(grads, layer.trainable_weights))
        self.assertAllClose(mse.numpy(), 0, atol=1e-3)


class SampledExpectationFunctionalTests(tf.test.TestCase):
    """Test hybrid/integrated models that include a SampledExpectation layer."""

    def test_simple_param_value_input(self):
        """Train a densely connected hybrid model.

        This model will put a qubit in the zero or one state from a random state
        given the input zero or one.
        """
        bit = cirq.GridQubit(0, 0)
        symbols = sympy.symbols('x y z')
        circuit = _gen_single_bit_rotation_problem(bit, symbols)

        inputs = tf.keras.Input(shape=(1,), dtype=tf.dtypes.float64)
        datum = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        l1 = tf.keras.layers.Dense(10)(inputs)
        l2 = tf.keras.layers.Dense(3)(l1)
        outputs = sampled_expectation.SampledExpectation()(
            datum,
            symbol_names=symbols,
            operators=cirq.Z(bit),
            symbol_values=l2,
            repetitions=5000)
        model = tf.keras.Model(inputs=[datum, inputs], outputs=outputs)

        data_in = np.array([[1], [0]], dtype=np.float32)
        data_out = np.array([[1], [-1]], dtype=np.float32)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                      loss=tf.keras.losses.mean_squared_error)

        circuits = util.convert_to_tensor([circuit, circuit])

        history = model.fit(x=[circuits, data_in], y=data_out, epochs=30)
        self.assertAllClose(history.history['loss'][-1], 0, atol=0.3)

    def test_simple_op_input(self):
        """Test a simple operator input

        Learn qubit in the z+ state using two different measurement operators.
        """
        bit = cirq.GridQubit(0, 0)
        symbols = sympy.symbols('x y z')
        ops = util.convert_to_tensor([[cirq.Z(bit)], [cirq.Z(bit)]])
        n = tf.convert_to_tensor([[5000], [5000]], dtype=tf.int32)

        circuit = util.convert_to_tensor(
            [_gen_single_bit_rotation_problem(bit, symbols)] * 2)

        data_out = tf.convert_to_tensor(np.array([[1], [1]]))
        op_inp = tf.keras.Input(shape=(1,), dtype=tf.dtypes.string)
        n_inp = tf.keras.Input(shape=(1,), dtype=tf.dtypes.int32)
        circuit_inp = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        circuit_output = sampled_expectation.SampledExpectation()(
            circuit_inp,
            symbol_names=symbols,
            operators=op_inp,
            repetitions=n_inp)
        model = tf.keras.Model(inputs=[circuit_inp, op_inp, n_inp],
                               outputs=[circuit_output])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            loss=tf.keras.losses.mean_squared_error,
        )
        history = model.fit(x=[circuit, ops, n],
                            y=data_out,
                            batch_size=1,
                            epochs=3)

        self.assertAllClose(history.history['loss'][-1], 0, atol=1e-2)

    def test_simple_op_and_param_input(self):
        """Test a simple operator and parameter input.

        Train a NN to put a qubit in the z+ or x+ states based on a classical
        binary input.
        """
        bit = cirq.GridQubit(0, 0)
        symbols = sympy.symbols('x y z')
        ops = util.convert_to_tensor([[cirq.Z(bit)], [cirq.Z(bit)]])
        n = tf.convert_to_tensor([[5000], [5000]], dtype=tf.int32)
        circuits = util.convert_to_tensor(
            [_gen_single_bit_rotation_problem(bit, symbols)] * 2)
        data_in = np.array([[1], [0]])
        data_out = np.array([[1], [1]])

        data_inp = tf.keras.layers.Input(shape=(1), dtype=tf.dtypes.float32)
        op_inp = tf.keras.layers.Input(shape=(1,), dtype=tf.dtypes.string)
        n_inp = tf.keras.layers.Input(shape=(1,), dtype=tf.dtypes.int32)
        circuit_inp = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        dense_1 = tf.keras.layers.Dense(10)(data_inp)
        dense_2 = tf.keras.layers.Dense(3)(dense_1)
        circuit_output = sampled_expectation.SampledExpectation()(
            circuit_inp,
            symbol_names=symbols,
            symbol_values=dense_2,
            operators=op_inp,
            repetitions=n_inp)

        functional_model = tf.keras.Model(
            inputs=[circuit_inp, data_inp, op_inp, n_inp],
            outputs=[circuit_output])

        functional_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
            loss=tf.keras.losses.mean_squared_error)
        history = functional_model.fit(x=[circuits, data_in, ops, n],
                                       y=data_out,
                                       batch_size=2,
                                       epochs=20)
        self.assertAllClose(history.history['loss'][-1], 0, atol=3)

    def test_dnn_qnn_dnn(self):
        """Train a fully hybrid network using an SampledExpectation layer.

        Train the network to output +-5 given an input of 1 or 0. This tests
        that everything works when SampledExpectation layer is a middle layers.
        """
        bit = cirq.GridQubit(0, 0)
        symbols = sympy.symbols('x, y, z')
        circuits = util.convert_to_tensor(
            [_gen_single_bit_rotation_problem(bit, symbols)] * 2)
        data_in = np.array([[1], [0]], dtype=np.float32)
        data_out = np.array([[5], [-5]], dtype=np.float32)

        classical_input = tf.keras.Input(shape=(1,))
        circuit_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
        d1 = tf.keras.layers.Dense(10)(classical_input)
        d2 = tf.keras.layers.Dense(3)(d1)
        quantum = sampled_expectation.SampledExpectation()(
            circuit_input,
            symbol_names=symbols,
            symbol_values=d2,
            operators=cirq.Z(bit),
            repetitions=5000)
        d3 = tf.keras.layers.Dense(1)(quantum)

        model = tf.keras.Model(inputs=[circuit_input, classical_input],
                               outputs=d3)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                      loss=tf.keras.losses.mean_squared_error)
        history = model.fit(x=[circuits, data_in],
                            y=data_out,
                            batch_size=2,
                            epochs=75)
        self.assertAllClose(history.history['loss'][-1], 0, atol=4)


if __name__ == '__main__':
    tf.test.main()
