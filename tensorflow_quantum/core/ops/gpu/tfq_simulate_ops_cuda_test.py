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
"""Tests that specifically target tfq_simulate_ops_cuda."""
import time
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops import tfq_simulate_ops
from tensorflow_quantum.core.ops import tfq_simulate_ops_gpu
from tensorflow_quantum.core.ops import tfq_simulate_ops_gpu_cpu
from tensorflow_quantum.python import util


class SimulateExpectationTest(tf.test.TestCase):
    """Tests tfq_simulate_expectation."""

    def test_simulate_expectation_diff(self):
        """Make sure that cpu & gpu ops have the same results."""
        # TF 2
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) < 1:
            self.skipTest("Expected at least 1 GPU but found {} GPUs".format(
                len(gpus)))
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

        cpu_avg_time = []
        for _ in range(10):
            cpu_time = time.time()
            res_cpu = tfq_simulate_ops.tfq_simulate_expectation(
                circuit_batch_tensor,
                symbol_names, symbol_values_array.astype(np.float64),
                pauli_sums_tensor)
            cpu_time = time.time() - cpu_time
            cpu_avg_time.append(cpu_time)
        cpu_avg_time = sum(cpu_avg_time) / 10.0
        print("\n\tCPU time: ", cpu_avg_time, "\n")

        avg_cpu_with_gpu_time = []
        for _ in range(10):
            cpu_with_gpu_time = time.time()
            with tf.device("/device:GPU:0"):
                res_cpu_with_gpu = tfq_simulate_ops.tfq_simulate_expectation(
                    circuit_batch_tensor,
                    symbol_names, symbol_values_array.astype(np.float64),
                    pauli_sums_tensor)
            cpu_with_gpu_time = time.time() - cpu_with_gpu_time
            avg_cpu_with_gpu_time.append(cpu_with_gpu_time)
        avg_cpu_with_gpu_time = sum(avg_cpu_with_gpu_time) / 10.0

        # Both are CPU devices.
        self.assertEqual(res_cpu.device, res_cpu_with_gpu.device)
        np.testing.assert_allclose(res_cpu, res_cpu_with_gpu)
        print("\n\tCPU with GPU device time: ", avg_cpu_with_gpu_time, "\n")

        @tf.function
        def cpu_with_gpu_fn():
            with tf.device("/device:GPU:0"):
                return tfq_simulate_ops.tfq_simulate_expectation(
                    circuit_batch_tensor,
                    symbol_names, symbol_values_array.astype(np.float64),
                    pauli_sums_tensor)

        avg_fn_cpu_with_gpu_time = []
        for _ in range(10):
            fn_cpu_with_gpu_time = time.time()
            res_fn_cpu_with_gpu = cpu_with_gpu_fn()
            fn_cpu_with_gpu_time = time.time() - fn_cpu_with_gpu_time
            avg_fn_cpu_with_gpu_time.append(fn_cpu_with_gpu_time)
        avg_fn_cpu_with_gpu_time = sum(avg_fn_cpu_with_gpu_time) / 10.0

        # CPU & GPU devices.
        self.assertNotEqual(res_cpu.device, res_fn_cpu_with_gpu.device)
        np.testing.assert_allclose(res_cpu, res_fn_cpu_with_gpu)
        print("\n\ttf.function, CPU with GPU device time: ",
              avg_fn_cpu_with_gpu_time, "\n")

        avg_gpu_time = []
        for _ in range(10):
            gpu_time = time.time()
            res_gpu = tfq_simulate_ops_gpu_cpu.tfq_simulate_expectation(
                circuit_batch_tensor,
                symbol_names, symbol_values_array.astype(np.float64),
                pauli_sums_tensor)
            gpu_time = time.time() - gpu_time
            avg_gpu_time.append(gpu_time)
        avg_gpu_time = sum(avg_gpu_time) / 10.0
        print("\n\tGPU version time: ", avg_gpu_time, "\n")


        # This guarantees that both tensors are not in the same devices
        # (e.g. CPU vs GPU)
        # self.assertNotEqual(res.device, res_gpu.device)
        # -> this doesn't work anymore because TFQ op itself is in CPU.
        # only qsim::SimulatorCUDA is in GPU
        np.testing.assert_allclose(res_cpu, res_gpu)
        self.assertGreater(cpu_avg_time, avg_gpu_time)


if __name__ == "__main__":
    tf.test.main()
