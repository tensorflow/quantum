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
"""Tests for the sample layer."""
import numpy as np
from absl.testing import parameterized
import sympy
import tensorflow as tf
import cirq

from tensorflow_quantum.python.layers.circuit_executors import sample
from tensorflow_quantum.python import util


class SampleTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for the Sample layer."""

    def test_sample_create(self):
        """Test that sample instantiates correctly."""
        sample.Sample(backend=cirq.Simulator())
        sample.Sample()
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="junk is invalid"):
            sample.Sample(backend='junk')

    def test_sample_invalid_type_inputs(self):
        """Test that sample rejects bad inputs."""
        sampler = sample.Sample()
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="repetitions not specified"):
            sampler(cirq.Circuit())
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="greater than zero"):
            sampler(cirq.Circuit(), repetitions=-1)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="cannot be parsed to int32"):
            sampler(cirq.Circuit(), repetitions='junk')

    def test_sample_invalid_shape_inputs(self):
        """Test that sample rejects bad input shapes."""
        sampler = sample.Sample()

        with self.assertRaisesRegex(TypeError,
                                    expected_regex="string or sympy.Symbol"):
            sampler(cirq.Circuit(),
                    symbol_values=[[0.5]],
                    symbol_names=[[]],
                    repetitions=10)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="rank 2. Got rank 1"):
            sampler(cirq.Circuit(),
                    symbol_values=[0.5],
                    symbol_names=['name'],
                    repetitions=10)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex="rank 1. Got rank 2"):
            sampler([[cirq.Circuit()]],
                    symbol_values=[[0.5]],
                    symbol_names=['name'],
                    repetitions=10)

        with self.assertRaisesRegex(
                TypeError, expected_regex="cannot be parsed to int32 tensor"):
            sampler([cirq.Circuit()], repetitions=[10])

    @parameterized.parameters([
        {
            'backend': 'noiseless'
        },
        {
            'backend': 'noisy'
        },
        {
            'backend': cirq.Simulator()
        },
        {
            'backend': None  # old API usage.
        }
    ])
    def test_sample_invalid_combinations(self, backend):
        """Test with valid type inputs and valid value, but incorrect combo."""
        sampler = sample.Sample(backend)
        symbol = sympy.Symbol('alpha')
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0))**symbol)
        with self.assertRaisesRegex(Exception, expected_regex=""):
            # no value provided.
            sampler([circuit, circuit], symbol_names=[symbol], repetitions=5)

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # no name provided.
            sampler([circuit, circuit],
                    symbol_names=[],
                    symbol_values=[[2.0], [3.0]],
                    repetitions=5)

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # deceptive, but the circuit shouldn't be in a list. otherwise fine.
            sampler([circuit],
                    symbol_names=['alpha'],
                    symbol_values=[[2.0], [3.0]],
                    repetitions=5)

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # wrong symbol name.
            sampler([circuit],
                    symbol_names=['alphaaaa'],
                    symbol_values=[[2.0], [3.0]],
                    repetitions=5)

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # too many symbol values provided.
            sampler(circuit,
                    symbol_names=['alpha'],
                    symbol_values=[[2.0, 4.0], [3.0, 5.0]],
                    repetitions=5)

        with self.assertRaisesRegex(Exception, expected_regex=""):
            # Wrong batch_size for symbol values.
            sampler([circuit],
                    symbol_names=['alpha'],
                    symbol_values=np.zeros((3, 1)),
                    repetitions=5)

    def test_sample_basic_inputs(self):
        """Test that sample ingests inputs correctly in simple settings."""
        sampler = sample.Sample()
        sampler(cirq.Circuit(), repetitions=10)
        sampler([cirq.Circuit()], repetitions=10)
        sampler(cirq.Circuit(),
                symbol_names=['name'],
                symbol_values=[[0.5]],
                repetitions=10)
        sampler(cirq.Circuit(),
                symbol_names=[sympy.Symbol('name')],
                symbol_values=[[0.5]],
                repetitions=10)

    def test_sample_outputs_simple(self):
        """Test the simplest call where nothing but circuits are provided."""
        sampler = sample.Sample()
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0)))
        output = sampler([circuit, circuit], repetitions=5)
        self.assertShapeEqual(np.empty((2, 5, 1)), output.to_tensor())

    # TODO(trevormccrt): add QuantumEngineSampler to this once it is available
    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                backend=['noiseless', 'noisy',
                         cirq.Simulator(), None],
                all_n_qubits=[[3, 4, 10]],
                n_samples=[1],
                symbol_names=[[], ['a', 'b']])))
    def test_sample_output(self, backend, all_n_qubits, n_samples,
                           symbol_names):
        """Test that expected output format is preserved.

        Check that any pre or post processing done inside the layers does not
        cause what is output from the layer to structurally deviate from what
        is expected.
        """
        sampler = sample.Sample(backend=backend)
        bits = cirq.GridQubit.rect(1, max(all_n_qubits))
        programs = []
        expected_outputs = []
        for n_qubits in all_n_qubits:
            programs.append(cirq.Circuit(*cirq.X.on_each(*bits[0:n_qubits])))
            expected_outputs.append([[1] * n_qubits for _ in range(n_samples)])
        symbol_values = np.random.random((len(all_n_qubits), len(symbol_names)))
        layer_output = sampler(programs,
                               symbol_names=symbol_names,
                               symbol_values=symbol_values,
                               repetitions=n_samples).to_list()

        self.assertEqual(expected_outputs, layer_output)


if __name__ == "__main__":
    tf.test.main()
