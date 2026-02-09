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
"""Benchmark differentiator methods."""
import os
import time
import string

from absl.testing import parameterized
import cirq
import tensorflow as tf
import numpy as np

from tensorflow_quantum.core.ops import tfq_simulate_ops

from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import (
    linear_combination,
    parameter_shift,
)

from benchmarks.scripts import benchmark_util
from benchmarks.scripts import flags

SRC = os.path.dirname(os.path.realpath(__file__))
os.environ['TEST_REPORT_FILE_PREFIX'] = os.path.join(SRC, 'reports/')
TEST_PARAMS_1 = flags.test_flags(n_symbols=4,
                                 n_qubits=3,
                                 n_moments=5,
                                 op_density=0.9)
TEST_PARAMS_2 = flags.test_flags(n_symbols=3,
                                 n_qubits=4,
                                 n_moments=5,
                                 op_density=0.6)


class GradientBenchmarksTest(tf.test.TestCase, parameterized.TestCase):
    """Test the Gradient benchmarking class."""

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'diff': [
                        linear_combination.ForwardDifference(),
                        linear_combination.CentralDifference(),
                        parameter_shift.ParameterShift(),
                    ],
                    'params': [TEST_PARAMS_1, TEST_PARAMS_2]
                })))
    def test_benchmark_gradient(self, diff, params):
        """Test that op constructs and runs correctly."""

        bench_name = f"GradientBenchmarks.{diff.__class__.__name__}_{params.n_qubits}_{params.n_moments}_{params.batch_size}_{params.n_symbols}"
        proto_file_path = os.path.join(SRC, "reports/", bench_name)
        self.addCleanup(os.remove, proto_file_path)

        bench = GradientBenchmarks(params=params)
        bench.setup()
        bench._benchmark_tfq_differentiator(diff, params)

        res = benchmark_util.read_benchmark_entry(proto_file_path)
        self.assertEqual(res.name, bench_name)
        self.assertEqual(
            res.extras.get("n_qubits").double_value, params.n_qubits)
        self.assertEqual(
            res.extras.get("n_moments").double_value, params.n_moments)
        self.assertEqual(
            res.extras.get("op_density").double_value, params.op_density)
        assert hasattr(res, 'iters')
        assert hasattr(res, 'wall_time')


class GradientBenchmarks(tf.test.Benchmark):
    """Benchmarks for circuit differentiation.

    Flags:
        --n_qubits --n_moments --op_density --n_runs --n_symbols --batch_size
        --n_burn
    """

    def __init__(self, params=None):
        """Pull in command line flags or use provided flags."""
        super().__init__()
        self.params = params if params else flags.FLAGS
        self.setup()

    def setup(self):
        """Persistent variational circuit, parameters, and observables."""
        qubits = cirq.GridQubit.rect(1, self.params.n_qubits)

        # Generate arbitrary symbol set without name clashes.
        symbol_names = set()
        while len(symbol_names) < self.params.n_symbols:
            symbol_names.add(''.join(
                np.random.choice(list(string.ascii_uppercase),
                                 size=4,
                                 replace=True)))
        symbol_names = list(symbol_names)

        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
            qubits=qubits,
            symbols=symbol_names,
            batch_size=self.params.batch_size,
            n_moments=self.params.n_moments,
            p=self.params.op_density)
        psums = util.random_pauli_sums(qubits, 1, self.params.batch_size)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch],
            dtype=np.float32)

        self.symbol_names = symbol_names
        self.symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
        self.programs = util.convert_to_tensor(circuit_batch)
        self.psums = util.convert_to_tensor([psums])

    def _benchmark_tfq_differentiator(self, differentiator, params):
        """Common pipeline for benchmarking and reporting."""
        # for parametrization over a single differentiator instance
        differentiator.refresh()
        op = differentiator.generate_differentiable_op(
            analytic_op=tfq_simulate_ops.tfq_simulate_expectation)

        for _ in range(params.n_burn):
            op(self.programs, self.symbol_names, self.symbol_values_tensor,
               self.psums)

        deltas = [None] * params.n_runs
        for i in range(params.n_runs):
            start = time.perf_counter()
            with tf.GradientTape() as g:
                g.watch(self.symbol_values_tensor)
                expectations = op(self.programs, self.symbol_names,
                                  self.symbol_values_tensor, self.psums)
            g.gradient(expectations, self.symbol_values_tensor)
            deltas[i] = time.perf_counter() - start

        # Name benchmark logs by differentiator classname.
        name = f"{differentiator.__class__.__name__}_{params.n_qubits}_{params.n_moments}_{params.batch_size}_{params.n_symbols}"

        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 f"{self.__class__.__name__}.{name}")
        if os.path.exists(full_path):
            os.remove(full_path)

        extras = {
            'n_qubits': params.n_qubits,
            'n_moments': params.n_moments,
            'op_density': params.op_density,
            'n_symbols': params.n_symbols,
            'batch_size': params.batch_size,
            "min_time": min(deltas),
        }

        benchmark_values = {
            "iters": params.n_runs,
            "wall_time": np.median(deltas),
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)
        return benchmark_values

    def benchmark_finite_difference_forward(self):
        """Benchmark the forward difference gradient method."""
        diff = linear_combination.ForwardDifference()
        self._benchmark_tfq_differentiator(diff, self.params)

    def benchmark_finite_difference_central(self):
        """Benchmark the central difference gradient method."""
        diff = linear_combination.CentralDifference()
        self._benchmark_tfq_differentiator(diff, self.params)

    def benchmark_parameter_shift(self):
        """Benchmark the parameter shift gradient method."""
        diff = parameter_shift.ParameterShift()
        self._benchmark_tfq_differentiator(diff, self.params)


if __name__ == "__main__":
    tf.test.main()
