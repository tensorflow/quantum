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
from dataclasses import dataclass

SRC = os.path.dirname(os.path.realpath(__file__))
os.environ['TEST_REPORT_FILE_PREFIX'] = os.path.join(SRC, 'reports/')

@dataclass(frozen=True)
class BenchmarkParams:
    """Frozen dataclass to store the parameters for the benchmark"""
    n_qubits: int
    n_moments: int
    batch_size: int
    n_iters: int = 100

_test_params_1 = BenchmarkParams(n_qubits=20, n_moments=1, batch_size=5)
_test_params_2 = BenchmarkParams(n_qubits=21, n_moments=10, batch_size=5) # more depth
_test_params_3 = BenchmarkParams(n_qubits=22, n_moments=1, batch_size=5, n_iters=10)

TEST_PARAMS_EXPECTATION = [
    _test_params_1,
    # _test_params_2, # uncomment for depth params
    ]
TEST_PARAMS_SAMPLED_EXPECTATION = [
    _test_params_1,
    # _test_params_2, # uncomment for depth params
    ]
TEST_PARAMS_SAMPLES = [
    _test_params_1,
    # _test_params_2, # uncomment for depth params
    ]
TEST_PARAMS_STATE = [_test_params_3,]

def _measure_median_runtime(
        fn,
        tag,
        num_samples=10,
        result_avg=False,
):
    """Measures median runtime for given function.

    Args:
        fn: function.
        tag: The message title.
        num_samples: The number of measurements.
        result_avg: True if the results are all mediand.

    Returns:
        The median time and the (averaged) result.
    """
    median_time = []
    avg_res = []
    for _ in range(num_samples):
        begin_time = time.time()
        result = fn()
        duration = time.time() - begin_time
        median_time.append(duration)
        if result_avg:
            avg_res.append(result)
    median_time = np.median(median_time)
    print(f"\n\t{tag} time: {median_time}\n")
    if result_avg:
        result = np.average(avg_res, axis=0)
    return median_time, result


class RandomCircuitBenchmark(tf.test.Benchmark):
    """Benchmark cuquantum simulations against cpu."""

    def __init__(self, params: BenchmarkParams):
        """Pull in command line flags or use provided flags."""
        super(RandomCircuitBenchmark, self).__init__()
        # Allow input params for testing purposes.
        self.params = params

    def benchmark_expectation_cpu(self):
        """Benchmark expectation simulator on cpu."""

        n_qubits = self.params.n_qubits
        batch_size = self.params.batch_size
        circuit_depth = self.params.n_moments
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(circuit_depth, n_qubits)
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

        cpu_avg_time, _ = _measure_median_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor),
            "Expectation CPU",
            num_samples=self.params.n_iters,
        )

        extras = {
            'n_qubits': self.params.n_qubits,
            'batch_size': self.params.batch_size,
            'num_samples': self.params.n_iters,
            'median_time': cpu_avg_time,
            # 'cuquantum_avg_time': cuquantum_avg_time,
        }

        name = "benchmark_expectation_cpu"
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": 1,
            "wall_time": cpu_avg_time,
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)

        return benchmark_values


    def benchmark_expectation_cuquantum(self):
        """Benchmark expectation simulator on cpu."""

        n_qubits = self.params.n_qubits
        batch_size = self.params.batch_size
        circuit_depth = self.params.n_moments
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(circuit_depth, n_qubits)
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

        # Benchmark time on GPU (cuquantum)
        cuquantum_avg_time, _ = _measure_median_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor),
            "Expectation cuQuantum",
            num_samples=self.params.n_iters,
        )

        extras = {
            'n_qubits': self.params.n_qubits,
            'batch_size': self.params.batch_size,
            'num_samples': self.params.n_iters,
            'median_time': cuquantum_avg_time,
        }

        name = "benchmark_expectation_cuquantum"
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

        return benchmark_values
    
    def benchmark_sampled_expectation_cpu(self, params=None):
        params = params if params else self.params
        n_qubits = params.n_qubits
        batch_size = params.batch_size
        circuit_depth = params.n_moments
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(circuit_depth, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)
        n_samples = [[10000]] * batch_size

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)
        pauli_sums_tensor = util.convert_to_tensor([[x] for x in pauli_sums])

        cpu_avg_time, _ = _measure_median_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_sampled_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor,
                n_samples),
            "SampledExpectation CPU",
            num_samples=params.n_iters,
            result_avg=False,
        )

        extras = {
            'n_qubits': params.n_qubits,
            'batch_size': params.batch_size,
            'num_samples': params.n_iters,
            'median_time': cpu_avg_time,
            # 'cuquantum_avg_time': cuquantum_avg_time,
        }

        name = "benchmark_sampled_expectation_cpu"
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": 1,
            "wall_time": cpu_avg_time,
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)

        return benchmark_values
    
    def benchmark_sampled_expectation_cuquantum(self, params=None):
        params = params if params else self.params
        n_qubits = params.n_qubits
        batch_size = params.batch_size
        circuit_depth = params.n_moments
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(circuit_depth, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)
        n_samples = [[10000]] * batch_size

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        pauli_sums = util.random_pauli_sums(qubits, 3, batch_size)
        pauli_sums_tensor = util.convert_to_tensor([[x] for x in pauli_sums])

        cuquantum_avg_time, res_cuquantum = _measure_median_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_sampled_expectation(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), pauli_sums_tensor,
                n_samples),
            "SampledExpectation cuQuantum",
            num_samples=params.n_iters,
            result_avg=False,
        )

        extras = {
            'n_qubits': params.n_qubits,
            'batch_size': params.batch_size,
            'num_samples': params.n_iters,
            'median_time': cuquantum_avg_time,
            # 'cuquantum_avg_time': cuquantum_avg_time,
        }

        name = "benchmark_sampled_expectation_cuquantum"
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

        return benchmark_values

    def benchmark_samples_cpu(self, params=None):
        params = params if params else self.params
        n_qubits = params.n_qubits
        batch_size = params.batch_size
        circuit_depth = params.n_moments
        symbol_names = ['alpha']
        n_samples = [100]
        qubits = cirq.GridQubit.rect(circuit_depth, n_qubits)

        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        cpu_avg_time, _ = _measure_median_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_samples(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), n_samples),
            "Samples CPU",
            num_samples=params.n_iters,
            result_avg=False,
        )

        extras = {
            'n_qubits': params.n_qubits,
            'batch_size': params.batch_size,
            'num_samples': params.n_iters,
            'median_time': cpu_avg_time,
            # 'cuquantum_avg_time': cuquantum_avg_time,
        }

        name = "benchmark_simulate_samples_cpu"
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": 1,
            "wall_time": cpu_avg_time,
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)

        return benchmark_values

    def benchmark_samples_cuquantum(self, params=None):
        params = params if params else self.params
        n_qubits = params.n_qubits
        batch_size = params.batch_size
        circuit_depth = params.n_moments
        symbol_names = ['alpha']
        n_samples = [100]
        qubits = cirq.GridQubit.rect(circuit_depth, n_qubits)

        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        cuquantum_avg_time, _ = _measure_median_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_samples(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64), n_samples),
            "Samples cuQuantum",
            num_samples=params.n_iters,
            result_avg=False,
        )

        extras = {
            'n_qubits': params.n_qubits,
            'batch_size': params.batch_size,
            'num_samples': params.n_iters,
            'median_time': cuquantum_avg_time,
            # 'cuquantum_avg_time': cuquantum_avg_time,
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

        return benchmark_values

    def benchmark_state_cpu(self, params=None):
        params = params if params else self.params
        n_qubits = params.n_qubits
        batch_size = params.batch_size
        circuit_depth = params.n_moments
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(circuit_depth, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)


        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        cpu_avg_time, _ = _measure_median_runtime(
            lambda: tfq_simulate_ops.tfq_simulate_state(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64)),
            "State CPU",
            num_samples=params.n_iters,
        )

        extras = {
            'n_qubits': params.n_qubits,
            'batch_size': params.batch_size,
            'num_samples': params.n_iters,
            'median_time': cpu_avg_time,
        }

        name = "benchmark_simulate_state_cpu"
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": 1,
            "wall_time": cpu_avg_time,
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)

        return benchmark_values

    def benchmark_state_cuquantum(self, params=None):
        params = params if params else self.params
        n_qubits = params.n_qubits
        batch_size = params.batch_size
        circuit_depth = params.n_moments
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(circuit_depth, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        circuit_batch_tensor = util.convert_to_tensor(circuit_batch)


        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        cuquantum_avg_time, _ = _measure_median_runtime(
            lambda: tfq_simulate_ops_cuquantum.tfq_simulate_state(
                circuit_batch_tensor, symbol_names,
                symbol_values_array.astype(np.float64)),
            "State cuQuantum",
            num_samples=params.n_iters,
        )

        extras = {
            'n_qubits': params.n_qubits,
            'batch_size': params.batch_size,
            'num_samples': params.n_iters,
            'median_time': cuquantum_avg_time,
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

        return benchmark_values



class SimulateExpectationCuquantumTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_simulate_expectation."""

    @parameterized.parameters(
        TEST_PARAMS_EXPECTATION
    )
    def test_simulate_expectation_cpu_vs_cuquantum(self, params):
        """Make sure that cuquantum version is faster."""
        bench = RandomCircuitBenchmark(params)

        benchmark_cpu = bench.benchmark_expectation_cpu()
        benchmark_gpu = bench.benchmark_expectation_cuquantum()

        cpu_median_time = benchmark_cpu['extras']['median_time']
        gpu_median_time = benchmark_gpu['extras']['median_time']

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_median_time, gpu_median_time)

    @parameterized.parameters(
        TEST_PARAMS_SAMPLED_EXPECTATION
    )
    def test_simulate_sampled_expectation_cpu_vs_cuquantum(self, params):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        bench = RandomCircuitBenchmark(params)

        benchmark_cpu = bench.benchmark_sampled_expectation_cpu()
        benchmark_gpu = bench.benchmark_sampled_expectation_cuquantum()

        cpu_median_time = benchmark_cpu['extras']['median_time']
        gpu_median_time = benchmark_gpu['extras']['median_time']

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_median_time, gpu_median_time)

    @parameterized.parameters(
        TEST_PARAMS_SAMPLES
    )
    def test_simulate_samples_cpu_vs_cuquantum(self, params):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        bench = RandomCircuitBenchmark(params)

        benchmark_cpu = bench.benchmark_samples_cpu()
        benchmark_gpu = bench.benchmark_samples_cuquantum()

        cpu_median_time = benchmark_cpu['extras']['median_time']
        gpu_median_time = benchmark_gpu['extras']['median_time']

        # cuQuantum op should be faster than CPU op.
        self.assertGreater(cpu_median_time, gpu_median_time)

    @parameterized.parameters(
        TEST_PARAMS_STATE
    )
    def test_simulate_state_cpu_vs_cuquantum(self, params):
        """Make sure that cpu & gpu(cuquantum) ops have the same results."""
        bench = RandomCircuitBenchmark(params)

        benchmark_cpu = bench.benchmark_state_cpu()
        benchmark_gpu = bench.benchmark_state_cuquantum()

        cpu_median_time = benchmark_cpu['extras']['median_time']
        gpu_median_time = benchmark_gpu['extras']['median_time']

        self.assertGreater(cpu_median_time, gpu_median_time)


if __name__ == "__main__":
    tf.test.main()
