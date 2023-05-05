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
# =============================================================================
"""Tests that specifically target tfq_simulate_ops_cu*."""
import time
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops import tfq_simulate_ops
from tensorflow_quantum.core.ops import tfq_simulate_ops_cuquantum
from tensorflow_quantum.python import util


def measure_average_runtime(
        fn,
        tag,
        num_samples=10,
        result_avg=False,
):
    """Measures average runtime for given function.

    Args:
        fn: function.
        tag: The message title.
        num_samples: The number of measurements.
        result_avg: True if the results are all averaged.

    Returns:
        The average time and the (averaged) result.
    """
    avg_time = []
    avg_res = []
    for _ in range(num_samples):
        begin_time = time.time()
        result = fn()
        duration = time.time() - begin_time
        avg_time.append(duration)
        if result_avg:
            avg_res.append(result)
    avg_time = sum(avg_time) / float(num_samples)
    print(f"\n\t{tag} time: {avg_time}\n")
    if result_avg:
        result = np.average(avg_res, axis=0)
    return avg_time, result


class SimulateExpectationCuquantumTest(tf.test.TestCase):
    """Tests tfq_simulate_expectation."""

    def test_simulate_expectation_cpu_vs_cuquantum(self):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        n_qubits = 20
        batch_size = 5
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)
        pauli_sums_tensor = util.convert_to_tensor([[x] for x in pauli_sums])

        cpu_avg_time, res_cpu = measure_average_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor),
            "Expectation CPU",
            num_samples=100,
        )

        cuquantum_avg_time, res_cuquantum = measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor),
            "Expectation cuQuantum",
            num_samples=100,
        )

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_avg_time, cuquantum_avg_time)

        # The result should be the similar within a tolerance.
        np.testing.assert_allclose(res_cpu,
                                   res_cuquantum,
                                   atol=1e-4,
                                   err_msg="""
        # If failed, the GPU architecture in this system may be unsupported.
        # Please refer to the supported architectures here.
        # https://docs.nvidia.com/cuda/cuquantum/getting_started.html#custatevec
        """)

    def test_simulate_expectation_inputs(self):
        """Make sure that the expectation op fails gracefully on bad inputs."""
        n_qubits = 5
        batch_size = 5
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1'):
            # Circuit tensor has too many dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1.'):
            # symbol_names tensor has too many dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too many dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]),
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too few dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[0],
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too few dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, util.convert_to_tensor(list(pauli_sums)))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too many dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[[x]] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # circuit tensor has the right type but invalid values.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                ['junk'] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type but invalid values.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), ['junk'],
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'qubits not found in circuit'):
            # pauli_sums tensor has the right type but invalid values.
            new_qubits = [cirq.GridQubit(5, 5), cirq.GridQubit(9, 9)]
            new_pauli_sums = util.random_pauli_sums(new_qubits, 2, batch_size)
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in new_pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # pauli_sums tensor has the right type but invalid values 2.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [['junk']] * batch_size)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # circuits tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                [1.0] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), [0.1234],
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # pauli_sums tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [[1.0]] * batch_size)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # we are missing an argument.
            # pylint: disable=no-value-for-parameter
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array)
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), [])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong op size.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums
                                       ][:int(batch_size * 0.5)]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong symbol_values size.
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[:int(batch_size * 0.5)],
                util.convert_to_tensor([[x] for x in pauli_sums]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='cirq.Channel'):
            # attempting to use noisy circuit.
            noisy_circuit = cirq.Circuit(cirq.depolarize(0.3).on_each(*qubits))
            tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                util.convert_to_tensor([noisy_circuit for _ in pauli_sums]),
                symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]))

        res = tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
            util.convert_to_tensor([cirq.Circuit() for _ in pauli_sums]),
            symbol_names, symbol_values_array.astype(np.float64),
            util.convert_to_tensor([[x] for x in pauli_sums]))
        self.assertDTypeEqual(res, np.float32)


class SimulateSampledExpectationCuquantumTest(tf.test.TestCase):
    """Tests tfq_simulate_sampled_expectation."""

    def test_simulate_sampled_expectation_cpu_vs_cuquantum(self):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        n_qubits = 20
        batch_size = 5
        symbol_names = ['alpha']
        n_samples = [[10000]] * batch_size
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)
        pauli_sums_tensor = util.convert_to_tensor([[x] for x in pauli_sums])

        cpu_avg_time, res_cpu = measure_average_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_sampled_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor,
                n_samples),
            "SampledExpectation CPU",
            num_samples=10,
            result_avg=False,
        )

        cuquantum_avg_time, res_cuquantum = measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor,
                n_samples),
            "SampledExpectation cuQuantum",
            num_samples=10,
            result_avg=False,
        )

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_avg_time, cuquantum_avg_time)

        # The result should be the similar within a tolerance.
        np.testing.assert_allclose(res_cpu,
                                   res_cuquantum,
                                   atol=0.07,
                                   err_msg="""
        # If failed, the GPU architecture in this system may be unsupported.
        # Please refer to the supported architectures here.
        # https://docs.nvidia.com/cuda/cuquantum/getting_started.html#custatevec
        """)

    def test_simulate_sampled_expectation_inputs(self):
        """Make sure sampled expectation op fails gracefully on bad inputs."""
        n_qubits = 5
        batch_size = 5
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)
        num_samples = [[10]] * batch_size

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1'):
            # Circuit tensor has too many dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1.'):
            # symbol_names tensor has too many dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too many dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]),
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too few dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[0],
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too few dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch),
                symbol_names, symbol_values_array,
                util.convert_to_tensor(list(pauli_sums)), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'pauli_sums must be rank 2.'):
            # pauli_sums tensor has too many dimensions.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                [util.convert_to_tensor([[x] for x in pauli_sums])],
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'num_samples must be rank 2'):
            # num_samples tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'num_samples must be rank 2'):
            # num_samples tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                num_samples[0])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # circuit tensor has the right type but invalid values.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                ['junk'] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type but invalid values.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), ['junk'],
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'qubits not found in circuit'):
            # pauli_sums tensor has the right type but invalid values.
            new_qubits = [cirq.GridQubit(5, 5), cirq.GridQubit(9, 9)]
            new_pauli_sums = util.random_pauli_sums(new_qubits, 2, batch_size)
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in new_pauli_sums]),
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # pauli_sums tensor has the right type but invalid values 2.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [['junk']] * batch_size, num_samples)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # circuits tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                [1.0] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), [0.1234],
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size,
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # pauli_sums tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [[1.0]] * batch_size, num_samples)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # we are missing an argument.
            # pylint: disable=no-value-for-parameter
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, num_samples)
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), [],
                num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong op size.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor([cirq.Circuit()]), symbol_names,
                symbol_values_array.astype(np.float64),
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'greater than 0'):
            # pylint: disable=too-many-function-args
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]),
                [[-1]] * batch_size)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong symbol_values size.
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[:int(batch_size * 0.5)],
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='cirq.Channel'):
            # attempting to use noisy circuit.
            noisy_circuit = cirq.Circuit(cirq.depolarize(0.3).on_each(*qubits))
            tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                util.convert_to_tensor([noisy_circuit for _ in pauli_sums]),
                symbol_names, symbol_values_array,
                util.convert_to_tensor([[x] for x in pauli_sums]), num_samples)


class SimulateSamplesCuquantumTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_simulate_samples."""

    def test_simulate_samples_cpu_vs_cuquantum(self):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        n_qubits = 20
        batch_size = 5
        symbol_names = ['alpha']
        n_samples = [100]
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        cpu_avg_time, res_cpu = measure_average_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_samples(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), n_samples),
            "Samples CPU",
            num_samples=10,
            result_avg=False,
        )

        cuquantum_avg_time, res_cuquantum = measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), n_samples),
            "Samples cuQuantum",
            num_samples=10,
            result_avg=False,
        )

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_avg_time, cuquantum_avg_time)

        res_cpu = np.average(res_cpu, axis=1)
        res_cuquantum = np.average(res_cuquantum, axis=1)

        # The result should be the similar within a tolerance.
        np.testing.assert_allclose(res_cpu,
                                   res_cuquantum,
                                   atol=0.3,
                                   err_msg="""
        # If failed, the GPU architecture in this system may be unsupported.
        # Please refer to the supported architectures here.
        # https://docs.nvidia.com/cuda/cuquantum/getting_started.html#custatevec
        """)

    def test_simulate_samples_inputs(self):
        """Make sure the sample op fails gracefully on bad inputs."""
        n_qubits = 5
        batch_size = 5
        num_samples = 10
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 1. Got rank 2'):
            # programs tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                symbol_values_array, [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 1. Got rank 2'):
            # symbol_names tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                symbol_values_array, [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 2. Got rank 3'):
            # symbol_values tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]), [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 2. Got rank 1'):
            # symbol_values tensor has the wrong shape 2.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[0], [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'rank 1. Got rank 2'):
            # num_samples tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [[num_samples]])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # programs tensor has the right type, but invalid value.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(\
                                                  ['junk'] * batch_size,
                                                  symbol_names,
                                                  symbol_values_array,
                                                  [num_samples])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type, but invalid value.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), ['junk'],
                symbol_values_array, [num_samples])

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # programs tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples([1] * batch_size,
                                                            symbol_names,
                                                            symbol_values_array,
                                                            [num_samples])

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # programs tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), [1], symbol_values_array,
                [num_samples])

        with self.assertRaisesRegex(tf.errors.UnimplementedError,
                                    'Cast string to float is not supported'):
            # programs tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size, [num_samples])

        with self.assertRaisesRegex(Exception, 'junk'):
            # num_samples tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, ['junk'])

        with self.assertRaisesRegex(TypeError, 'missing'):
            # too few tensors.
            # pylint: disable=no-value-for-parameter
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array)
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong symbol_values size.
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[:int(batch_size * 0.5)], num_samples)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='cirq.Channel'):
            # attempting to use noisy circuit.
            noisy_circuit = cirq.Circuit(cirq.depolarize(0.3).on_each(*qubits))
            tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                util.convert_to_tensor([noisy_circuit for _ in circuit_batch]),
                symbol_names, symbol_values_array, [num_samples])

    @parameterized.parameters([
        {
            'all_n_qubits': [2, 3],
            'n_samples': 10
        },
        {
            'all_n_qubits': [1, 5, 8],
            'n_samples': 10
        },
    ])
    def test_sampling_output_padding(self, all_n_qubits, n_samples):
        """Check that the sampling ops pad outputs correctly"""
        op = tfq_simulate_ops_cuquantum.tfq_simulate_samples
        circuits = []
        expected_outputs = []
        for n_qubits in all_n_qubits:
            this_expected_output = np.zeros((n_samples, max(all_n_qubits)))
            this_expected_output[:, max(all_n_qubits) - n_qubits:] = 1
            this_expected_output[:, :max(all_n_qubits) - n_qubits] = -2
            expected_outputs.append(this_expected_output)
            circuits.append(
                cirq.Circuit(*cirq.X.on_each(
                    *cirq.GridQubit.rect(1, n_qubits))))
        results = op(util.convert_to_tensor(circuits), [], [[]] * len(circuits),
                     [n_samples]).numpy()
        self.assertAllClose(expected_outputs, results)


class SimulateStateCuquantumTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_simulate_samples."""

    def test_simulate_state_cpu_vs_cuquantum(self):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        n_qubits = 20
        batch_size = 5
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        cpu_avg_time, res_cpu = measure_average_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_state(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64)),
            "State CPU",
            num_samples=10,
        )

        cuquantum_avg_time, res_cuquantum = measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_state(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64)),
            "State cuQuantum",
            num_samples=10,
        )

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_avg_time, cuquantum_avg_time)

        # The result should be the similar within a tolerance.
        np.testing.assert_allclose(res_cpu,
                                   res_cuquantum,
                                   atol=1e-4,
                                   err_msg="""
        # If failed, the GPU architecture in this system may be unsupported.
        # Please refer to the supported architectures here.
        # https://docs.nvidia.com/cuda/cuquantum/getting_started.html#custatevec
        """)

    def test_simulate_state_inputs(self):
        """Make sure the state op fails gracefully on bad inputs."""
        n_qubits = 5
        batch_size = 5
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1'):
            # programs tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor([circuit_batch]), symbol_names,
                symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1'):
            # symbol_names tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), np.array([symbol_names]),
                symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2'):
            # symbol_values tensor has the wrong shape.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2'):
            # symbol_values tensor has the wrong shape 2.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[0])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # programs tensor has the right type, but invalid value.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(['junk'] * batch_size,
                                                          symbol_names,
                                                          symbol_values_array)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type, but invalid value.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), ['junk'],
                symbol_values_array)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # programs tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_state([1] * batch_size,
                                                          symbol_names,
                                                          symbol_values_array)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), [1], symbol_values_array)

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), symbol_names,
                [['junk']] * batch_size)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # too few tensors.
            # pylint: disable=no-value-for-parameter
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), symbol_names)
            # pylint: enable=no-value-for-parameter

        # TODO (mbbrough): determine if we should allow extra arguments ?
        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # wrong symbol_values size.
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[:int(batch_size * 0.5)])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='cirq.Channel'):
            # attempting to use noisy circuit.
            noisy_circuit = cirq.Circuit(cirq.depolarize(0.3).on_each(*qubits))
            tfq_simulate_ops_cuquantum.tfq_simulate_state(
                util.convert_to_tensor([noisy_circuit for _ in circuit_batch]),
                symbol_names, symbol_values_array)

    @parameterized.parameters([
        {
            'all_n_qubits': [2, 3]
        },
        {
            'all_n_qubits': [1, 5, 8]
        },
    ])
    def test_simulate_state_output_padding(self, all_n_qubits):
        """If a tfq_simulate op is asked to simulate states given circuits
        acting on different numbers of qubits, the op should return a tensor
        padded with zeros up to the size of the largest circuit. The padding
        should be physically correct, such that samples taken from the padded
        states still match samples taken from the original circuit. """
        circuit_batch = []
        for n_qubits in all_n_qubits:
            qubits = cirq.GridQubit.rect(1, n_qubits)
            circuit_batch += util.random_circuit_resolver_batch(qubits, 1)[0]

        tfq_results = tfq_simulate_ops_cuquantum.tfq_simulate_state(
            util.convert_to_tensor(circuit_batch), [],
            [[]] * len(circuit_batch))

        # Don't use batch_util here to enforce consistent padding everywhere
        # without extra tests.
        sim = cirq.Simulator()
        manual_padded_results = []
        for circuit in circuit_batch:
            result = sim.simulate(circuit)
            wf = result.final_state_vector
            blank_state = np.ones(
                (2**max(all_n_qubits)), dtype=np.complex64) * -2
            blank_state[:wf.shape[0]] = wf
            manual_padded_results.append(blank_state)

        self.assertAllClose(tfq_results, manual_padded_results, atol=1e-5)


if __name__ == "__main__":
    tf.test.main()
