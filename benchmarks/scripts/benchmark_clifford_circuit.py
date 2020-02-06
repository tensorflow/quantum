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
"""Benchmark simulators against classically simulatable circuits."""
import os
import time

from absl.testing import parameterized
import cirq
import tensorflow as tf
import numpy as np

from tensorflow_quantum.core.ops import tfq_simulate_ops
from tensorflow_quantum.core.serialize.serializer import serialize_circuit
from models.random_clifford_circuit import random_clifford_circuit
import flags
import benchmark_util

SEED = 48510234
SRC = os.path.dirname(os.path.realpath(__file__))
os.environ['TEST_REPORT_FILE_PREFIX'] = os.path.join(SRC, 'reports/')
TEST_PARAMS_1 = flags.TEST_FLAGS(n_qubits=3, n_moments=5, op_density=0.99)
TEST_PARAMS_2 = flags.TEST_FLAGS(n_qubits=4, n_moments=5, op_density=0.99)
ALL_PARAMS = [TEST_PARAMS_1, TEST_PARAMS_2]


class CliffordBenchmarksTest(tf.test.TestCase, parameterized.TestCase):
    """Test the Clifford benchmarking class."""

    @parameterized.named_parameters(
        ("params_1", TEST_PARAMS_1),
        ("params_2", TEST_PARAMS_2),
    )
    def testBenchmarkCliffordCircuitEager(self, params):
        """Test that Op constructs and runs correctly."""
        proto_file_path = os.path.join(
            SRC, "reports/",
            "CliffordBenchmarks.benchmark_clifford_circuit_{}_{}_{}".format(
                params.n_qubits, params.n_moments, params.batch_size))
        self.addCleanup(os.remove, proto_file_path)

        bench = CliffordBenchmarks(params=params)
        bench.benchmark_clifford_circuit_eager()

        res = benchmark_util.read_benchmark_entry(proto_file_path)
        self.assertEqual(
            res.name,
            "CliffordBenchmarks.benchmark_clifford_circuit_{}_{}_{}".format(
                params.n_qubits, params.n_moments, params.batch_size))
        self.assertEqual(
            res.extras.get("n_qubits").double_value, params.n_qubits)
        self.assertEqual(
            res.extras.get("n_moments").double_value, params.n_moments)
        self.assertEqual(
            res.extras.get("op_density").double_value, params.op_density)
        assert hasattr(res, 'iters')
        assert hasattr(res, 'wall_time')


class CliffordBenchmarks(tf.test.Benchmark):
    """Benchmark simulators against Clifford circuits.

    Flags:
        --n_qubits --n_moments --op_density --batch_size --n_runs --n_burn
    """

    def __init__(self, params=None):
        """Pull in command line flags or use provided flags."""
        super(CliffordBenchmarks, self).__init__()
        # Allow input params for testing purposes.
        self.params = params if params else flags.FLAGS

    def _simulate_circuit(self, circuit, params):
        # TODO: implement backend switch
        return tfq_simulate_ops.tfq_simulate_state(
            [str(serialize_circuit(circuit))] * params.batch_size, ["None"],
            [[0]] * params.batch_size)

    def benchmark_clifford_circuit_eager(self):
        """tf.test.Benchmark does not provide eager benchmarks methods."""

        qubits = cirq.GridQubit.rect(1, self.params.n_qubits)
        circuit = random_clifford_circuit(
            qubits,
            self.params.n_moments,
            self.params.op_density,
            random_state=np.random.RandomState(SEED))

        for _ in range(self.params.n_burn):
            _ = self._simulate_circuit(circuit, self.params)

        deltas = [None] * self.params.n_runs
        for i in range(self.params.n_runs):
            start = time.perf_counter()
            _ = self._simulate_circuit(circuit, self.params)
            deltas[i] = time.perf_counter() - start

        extras = {
            'n_qubits': self.params.n_qubits,
            'n_moments': self.params.n_moments,
            'op_density': self.params.op_density,
            'batch_size': self.params.batch_size,
            "min_time": min(deltas),
        }
        name = "benchmark_clifford_circuit_{}_{}_{}".format(
            self.params.n_qubits, self.params.n_moments, self.params.batch_size)

        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 "{}.{}".format(self.__class__.__name__, name))
        if os.path.exists(full_path):
            os.remove(full_path)

        benchmark_values = {
            "iters": self.params.n_runs,
            "wall_time": np.median(deltas),
            "extras": extras,
            "name": name,
        }
        self.report_benchmark(**benchmark_values)
        return benchmark_values


if __name__ == "__main__":
    tf.test.main()
