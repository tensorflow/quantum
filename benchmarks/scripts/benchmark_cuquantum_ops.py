import os
import time
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops import tfq_simulate_ops
from tensorflow_quantum.core.ops import tfq_simulate_ops_cuquantum
from tensorflow_quantum.python import util
import flags
import benchmark_util

SEED = 63536323
SRC = os.path.dirname(os.path.realpath(__file__))
os.environ['TEST_REPORT_FILE_PREFIX'] = os.path.join(SRC, 'reports/')

class SimulateExpectationCuquantumTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_simulate_expectation."""

    def test_simulate_expectation_cpu_vs_cuquantum(self):
        """Make sure that CPU & GPU(cuquantum) ops have the same results."""
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

        cpu_avg_time, res_cpu = self._measure_average_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor),
            "Expectation CPU",
            num_samples=100,
        )

        # Benchmark time on GPU (cuquantum)
        cuquantum_avg_time, res_cuquantum = self._measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor),
            "Expectation cuQuantum",
            num_samples=100,
        )

        extras = {
            'n_qubits': 20,
            'batch_size': 5,
            'num_samples': 100,
            'cpu_avg_time': cpu_avg_time,
            'cuquantum_avg_time': cuquantum_avg_time,
        }

        name = "benchmark_simulate_expectation_cuquantum"
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": 1,
            "wall_time": cuquantum_avg_time,
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_avg_time, cuquantum_avg_time)


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

        cpu_avg_time, res_cpu = self._measure_average_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_sampled_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor,
                n_samples),
            "SampledExpectation CPU",
            num_samples=10,
            result_avg=False,
        )

        cuquantum_avg_time, res_cuquantum = self._measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor,
                n_samples),
            "SampledExpectation cuQuantum",
            num_samples=10,
            result_avg=False,
        )

        extras = {
            'n_qubits': 20,
            'batch_size': 5,
            'num_samples': 100,
            'cpu_avg_time': cpu_avg_time,
            'cuquantum_avg_time': cuquantum_avg_time,
        }

        name = "benchmark_simulate_sampled_expectation_cuquantum"
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": 1,
            "wall_time": cuquantum_avg_time,
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)


        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_avg_time, cuquantum_avg_time)


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

        cpu_avg_time, res_cpu = self._measure_average_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_samples(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), n_samples),
            "Samples CPU",
            num_samples=10,
            result_avg=False,
        )

        cuquantum_avg_time, res_cuquantum = self._measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), n_samples),
            "Samples cuQuantum",
            num_samples=10,
            result_avg=False,
        )

        extras = {
            'n_qubits': 20,
            'batch_size': 5,
            'num_samples': 10,
            'cpu_avg_time': cpu_avg_time,
            'cuquantum_avg_time': cuquantum_avg_time,
        }

        name = "benchmark_simulate_samples_cuquantum"
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": 1,
            "wall_time": cuquantum_avg_time,
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_avg_time, cuquantum_avg_time)

        res_cpu = np.average(res_cpu, axis=1)
        res_cuquantum = np.average(res_cuquantum, axis=1)


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

        cpu_avg_time, res_cpu = self._measure_average_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_state(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64)),
            "State CPU",
            num_samples=10,
        )

        cuquantum_avg_time, res_cuquantum = self._measure_average_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_state(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64)),
            "State cuQuantum",
            num_samples=10,
        )

        extras = {
            'n_qubits': n_qubits,
            'batch_size': batch_size,
            'num_samples': 10,
            'cpu_avg_time': cpu_avg_time,
            'cuquantum_avg_time': cuquantum_avg_time,
        }

        name = "benchmark_simulate_state_cuquantum"
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": 1,
            "wall_time": cuquantum_avg_time,
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_avg_time, cuquantum_avg_time)


    @staticmethod
    def _measure_average_runtime(
            self,
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

# class SimulateExpectationCuquantumBenchmark(tf.test.Benchmark):
#     """Benchmark tfq_simulate_expectation."""

#     def benchmark_simulate_expectation_cpu_vs_cuquantum(self):
#         test_obj = SimulateExpectationCuquantumTest()
#         cpu_avg_time, res_cpu = test_obj._measure_average_runtime(
#             lambda: tfq_simulate_ops.tfq_simulate_expectation(
#                 test_obj.circuit_batch_tensor, test_obj.symbol_names,
#                 test_obj.symbol_values_array.astype(np.float64),
#                 test_obj.pauli_sums_tensor),
#             "Expectation CPU",
#             num_samples=100,
#         )

#         cuquantum_avg_time, res_cuquantum = test_obj._measure_average_runtime(
#             lambda: tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
#                 test_obj.circuit_batch_tensor, test_obj.symbol_names,
#                 test_obj.symbol_values_array.astype(np.float64),
#                 test_obj.pauli_sums_tensor),
#             "Expectation cuQuantum",
#             num_samples=100,
#         )

#         extras = {
#             'n_qubits': 20,
#             'batch_size': 5,
#             'num_samples': 100,
#             'cpu_avg_time': cpu_avg_time,
#             'cuquantum_avg_time': cuquantum_avg_time,
#         }

#         name = "benchmark_simulate_expectation_cpu_vs_cuquantum"
#         full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
#                                  "{}.{}".format(self.__class__.__name__, name))
#         if os.path.exists(full_path):
#             os.remove(full_path)

#         benchmark_values = {
#             "iters": 1,
#             "wall_time": cuquantum_avg_time,
#             "extras": extras,
#             "name": name,
#         }
#         self.report_benchmark(**benchmark_values)
#         return benchmark_values


if __name__ == "__main__":
    tf.test.main()
