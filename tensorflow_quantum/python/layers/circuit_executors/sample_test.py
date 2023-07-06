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
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys
NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import numpy as np
from absl.testing import parameterized
import sympy
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python.layers.circuit_executors import sample
from tensorflow_quantum.python import util

RANDOM_SEED = 1234


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
            'backend': 'noiseless',
            'use_cuquantum': False,
        },
        {
            'backend': 'noisy',
            'use_cuquantum': False,
        },
        {
            'backend': cirq.Simulator(),
            'use_cuquantum': False,
        },
        {
            'backend': None,  # old API usage.
            'use_cuquantum': False,
        },
        {
            'backend': None,
            'use_cuquantum': True,
        }
    ])
    def test_sample_invalid_combinations(self, backend, use_cuquantum):
        """Test with valid type inputs and valid value, but incorrect combo."""
        if use_cuquantum and not circuit_execution_ops.is_gpu_configured():
            # GPU is not set. Ignores this sub-test.
            self.skipTest("GPU is not set. Ignoring gpu tests...")
        sampler = sample.Sample(backend, use_cuquantum=use_cuquantum)
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

    @parameterized.parameters([{
        'use_cuquantum': False,
    }, {
        'use_cuquantum': True,
    }])
    def test_sample_basic_inputs(self, use_cuquantum):
        """Test that sample ingests inputs correctly in simple settings."""
        if use_cuquantum and not circuit_execution_ops.is_gpu_configured():
            # GPU is not set. Ignores this sub-test.
            self.skipTest("GPU is not set. Ignoring gpu tests...")
        sampler = sample.Sample(use_cuquantum=use_cuquantum)
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

    @parameterized.parameters([{
        'use_cuquantum': False,
    }, {
        'use_cuquantum': True,
    }])
    def test_sample_outputs_simple(self, use_cuquantum):
        """Test the simplest call where nothing but circuits are provided."""
        if use_cuquantum and not circuit_execution_ops.is_gpu_configured():
            # GPU is not set. Ignores this sub-test.
            self.skipTest("GPU is not set. Ignoring gpu tests...")
        sampler = sample.Sample(use_cuquantum=use_cuquantum)
        circuit = cirq.Circuit(cirq.H(cirq.GridQubit(0, 0)))
        output = sampler([circuit, circuit], repetitions=5)
        self.assertShapeEqual(np.empty((2, 5, 1)), output.to_tensor())

    # TODO(trevormccrt): add ProcessorSampler to this once it is available
    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                backend=['noiseless', 'noisy',
                         cirq.Simulator(), None],
                use_cuquantum=[False, True],
                all_n_qubits=[[3, 4, 10]],
                n_samples=[1],
                symbol_names=[[], ['a', 'b']])))
    def test_sample_output(self, backend, use_cuquantum, all_n_qubits,
                           n_samples, symbol_names):
        """Test that expected output format is preserved.

        Check that any pre or post processing done inside the layers does not
        cause what is output from the layer to structurally deviate from what
        is expected.
        """
        if use_cuquantum and not circuit_execution_ops.is_gpu_configured():
            # GPU is not set. Ignores this sub-test.
            self.skipTest("GPU is not set. Ignoring gpu tests...")
        tf.random.set_seed(RANDOM_SEED)
        if use_cuquantum:
            # If use_cuquantum is True,
            if backend is not None and backend != 'noiseless':
                return
            # Passes backend=None or backend == 'noiseless' only.
        sampler = sample.Sample(backend=backend, use_cuquantum=use_cuquantum)
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
