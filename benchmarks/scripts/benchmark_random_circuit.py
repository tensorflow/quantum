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
"""Benchmark simulators against classically intractable 'supremacy' circuits."""
import os
import random
import time
from typing import Iterable, Sequence, TypeVar, cast

import numpy as np
from absl.testing import parameterized
import cirq
import tensorflow as tf

from tensorflow_quantum.core.ops import tfq_simulate_ops
from tensorflow_quantum.core.serialize.serializer import serialize_circuit
from benchmarks.scripts import flags
from benchmarks.scripts import benchmark_util

SEED = 63536323
SRC = os.path.dirname(os.path.realpath(__file__))
os.environ['TEST_REPORT_FILE_PREFIX'] = os.path.join(SRC, 'reports/')
TEST_PARAMS_1 = flags.test_flags(n_rows=3, n_cols=5, n_moments=5)
TEST_PARAMS_2 = flags.test_flags(n_rows=4, n_cols=4, n_moments=20)

T = TypeVar('T')


def _choice(rand_gen: float, sequence: Sequence[T]) -> T:
    """Choose a pseudo-random element from a non-empty sequence."""
    return sequence[int(rand_gen * len(sequence))]


def _make_cz_layer(qubits: Iterable[cirq.GridQubit], layer_index: int):
    """Yield a CZ interaction pattern for the given layer index."""
    offset = layer_index % 8
    for q in qubits:
        for q2 in [
                cirq.GridQubit(q.row + 1, q.col),
                cirq.GridQubit(q.row, q.col + 1)
        ]:
            if q2 in qubits and ((q.row + q.col + offset) % 2 == 0):
                yield cirq.CZ(q, q2)


def _add_cz_layer(layer_index: int, circuit: cirq.Circuit) -> int:
    """Add the next non-empty CZ layer and return the updated layer index."""
    cz_layer = None
    while not cz_layer:
        qubits = cast(Iterable[cirq.GridQubit], circuit.all_qubits())
        cz_layer = list(_make_cz_layer(qubits, layer_index))
        layer_index += 1

    circuit.append(cz_layer, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return layer_index


def generate_boixo_2018_beyond_classical_v2_grid(n_rows: int, n_cols: int,
                                                 cz_depth: int,
                                                 seed: int) -> cirq.Circuit:
    """Local copy of ReCirq's v2 beyond-classical grid circuit generator.

    Source reference:
      https://github.com/quantumlib/ReCirq/blob/main/recirq/beyond_classical/google_v2_beyond_classical.py

    Note:
      We intentionally keep this local copy to avoid introducing broader
      dependency migration. A future cleanup can switch to direct
      ReCirq dependency once repository constraints are aligned.
    """
    qubits = [
        cirq.GridQubit(i, j) for i in range(n_rows) for j in range(n_cols)
    ]
    non_diagonal_gates = [cirq.X**(1 / 2), cirq.Y**(1 / 2)]
    rand_gen = random.Random(seed).random

    circuit = cirq.Circuit()
    circuit.append(cirq.H(qubit) for qubit in qubits)

    layer_index = 0
    if cz_depth:
        layer_index = _add_cz_layer(layer_index, circuit)
        for qubit in qubits:
            if not circuit.operation_at(qubit, 1):
                circuit.append(cirq.T(qubit),
                               strategy=cirq.InsertStrategy.EARLIEST)

    for moment_index in range(2, cz_depth + 1):
        layer_index = _add_cz_layer(layer_index, circuit)
        for qubit in qubits:
            if not circuit.operation_at(qubit, moment_index):
                last_op = circuit.operation_at(qubit, moment_index - 1)
                if last_op:
                    gate = cast(cirq.GateOperation, last_op).gate
                    if gate == cirq.CZ:
                        circuit.append(_choice(rand_gen(),
                                               non_diagonal_gates).on(qubit),
                                       strategy=cirq.InsertStrategy.EARLIEST)
                    elif gate != cirq.T:
                        circuit.append(cirq.T(qubit),
                                       strategy=cirq.InsertStrategy.EARLIEST)

    circuit.append([cirq.H(qubit) for qubit in qubits],
                   strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    return circuit


def make_random_circuit(n_rows, n_cols, depth):
    """Generate a random unparameterized circuit of fixed depth."""
    circuit = generate_boixo_2018_beyond_classical_v2_grid(
        n_rows=n_rows,
        n_cols=n_cols,
        cz_depth=max(0, depth - 2),  # Account for initial/final Hadamards.
        seed=SEED)
    return cirq.Circuit(circuit[:depth])


class RandomCircuitBenchmarksTest(tf.test.TestCase, parameterized.TestCase):
    """Test the random circuit benchmarking class."""

    @parameterized.named_parameters(
        ("params_1", TEST_PARAMS_1),
        ("params_2", TEST_PARAMS_2),
    )
    def test_benchmark_random_circuit(self, params):
        """Test that Op constructs and runs correctly."""
        proto_file_path = os.path.join(
            SRC, "reports/", "RandomCircuitBenchmarks.benchmark_random_circuit_"
            f"{params.n_rows}_{params.n_cols}_{params.n_moments}")
        self.addCleanup(os.remove, proto_file_path)

        bench = RandomCircuitBenchmarks(params=params)
        bench.benchmark_random_circuit()

        res = benchmark_util.read_benchmark_entry(proto_file_path)
        self.assertEqual(
            res.name, "RandomCircuitBenchmarks.benchmark_random_circuit_"
            f"{params.n_rows}_{params.n_cols}_{params.n_moments}")
        self.assertEqual(res.extras.get("n_rows").double_value, params.n_rows)
        self.assertEqual(res.extras.get("n_cols").double_value, params.n_cols)
        self.assertEqual(
            res.extras.get("n_moments").double_value, params.n_moments)

        assert hasattr(res, 'iters')
        assert hasattr(res, 'wall_time')

    @parameterized.named_parameters(
        ("params_1", TEST_PARAMS_1),
        ("params_2", TEST_PARAMS_2),
    )
    def test_random_circuit_params(self, params):
        """Ensure that the random circuits are structured as advertised."""
        circuit = make_random_circuit(params.n_rows, params.n_cols,
                                      params.n_moments)
        self.assertEqual(len(circuit), params.n_moments)
        self.assertEqual(len(circuit.all_qubits()),
                         params.n_rows * params.n_cols)


class RandomCircuitBenchmarks(tf.test.Benchmark):
    """Benchmark simulators against random 'supremacy' circuits.

    Flags:
        --n_rows --n_cols --n_moments --batch_size --n_runs --n_burn
    """

    def __init__(self, params=None):
        """Pull in command line flags or use provided flags."""
        super().__init__()
        # Allow input params for testing purposes.
        self.params = params if params else flags.FLAGS

    def _simulate_circuit(self, circuit, params):
        # TODO: implement backend switch
        return tfq_simulate_ops.tfq_simulate_state(
            [str(serialize_circuit(circuit))] * params.batch_size, ["None"],
            [[0]] * params.batch_size)

    def benchmark_random_circuit(self):
        """Benchmark simulator performance on
        a classically intractable circuit."""

        circuit = make_random_circuit(self.params.n_rows, self.params.n_cols,
                                      self.params.n_moments)
        for _ in range(self.params.n_burn):
            _ = self._simulate_circuit(circuit, self.params)

        deltas = [None] * self.params.n_runs
        for i in range(self.params.n_runs):
            start = time.perf_counter()
            _ = self._simulate_circuit(circuit, self.params)
            deltas[i] = time.perf_counter() - start

        extras = {
            'n_rows': self.params.n_rows,
            'n_cols': self.params.n_cols,
            'n_qubits': len(circuit.all_qubits()),
            'n_moments': self.params.n_moments,
            'batch_size': self.params.batch_size,
            "min_time": min(deltas),
        }

        name = (f"benchmark_random_circuit_{self.params.n_rows}_"
                f"{self.params.n_cols}_{self.params.n_moments}")
        full_path = os.path.join(os.environ['TEST_REPORT_FILE_PREFIX'],
                                 f"{self.__class__.__name__}.{name}")
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
