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
"""Tests that specifically target tfq_simulate_ops_cu*."""
import time
import numpy as np
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
            "CPU",
            num_samples=10,
        )

        cuquantum_avg_time, res_cuquantum = measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor),
            "cuQuantum",
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


class SimulateSampledExpectationCuquantumTest(tf.test.TestCase):
    """Tests tfq_simulate_sampled_expectation."""

    def test_simulate_sampled_expectation_cpu_vs_cuquantum(self):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        n_qubits = 20
        batch_size = 5
        symbol_names = ['alpha']
        n_samples = [[100]] * batch_size
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
            "CPU",
            num_samples=10,
        )

        cuquantum_avg_time, res_cuquantum = measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor,
                n_samples),
            "cuQuantum",
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


class SimulateSamplesCuquantumTest(tf.test.TestCase):
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
            "CPU",
            num_samples=10,
            result_avg=True,
        )

        cuquantum_avg_time, res_cuquantum = measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), n_samples),
            "cuQuantum",
            num_samples=10,
            result_avg=True,
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


class SimulateStateCuquantumTest(tf.test.TestCase):
    """Tests tfq_simulate_samples."""

    def test_simulate_state_cpu_vs_cuquantum(self):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        n_qubits = 10
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
            "CPU",
            num_samples=10,
        )

        cuquantum_avg_time, res_cuquantum = measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_state(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64)),
            "cuQuantum",
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


if __name__ == "__main__":
    tf.test.main()
