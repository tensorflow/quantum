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
"""Tests that specifically target tfq_inner_product."""
import copy
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops.math_ops import fidelity_op
from tensorflow_quantum.python import util


class FidelityTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_fidelity_op."""

    @parameterized.parameters([
        {
            'n_qubits': 5,
            'batch_size': 1,
            'inner_dim_size': 5
        },
        {
            'n_qubits': 5,
            'batch_size': 10,
            'inner_dim_size': 1
        },
        {
            'n_qubits': 10,
            'batch_size': 10,
            'inner_dim_size': 2
        },
        {
            'n_qubits': 5,
            'batch_size': 10,
            'inner_dim_size': 5
        },
    ])
    def test_correctness_with_symbols(self, n_qubits, batch_size,
                                      inner_dim_size):
        """Tests that inner_product works with symbols."""
        symbol_names = ['alpha', 'beta', 'gamma']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        other_batch = [
            util.random_circuit_resolver_batch(qubits, inner_dim_size)[0]
            for i in range(batch_size)
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names = tf.convert_to_tensor(symbol_names,
                                            dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor(symbol_values_array)

        out = fidelity_op.fidelity(programs, symbol_names, symbol_values,
                                   other_programs)

        out_arr = np.empty((batch_size, inner_dim_size), dtype=np.complex64)
        for i in range(batch_size):
            final_circuit = cirq.resolve_parameters(circuit_batch[i],
                                                    resolver_batch[i])
            final_wf = cirq.final_state_vector(final_circuit)
            for j in range(inner_dim_size):
                internal_wf = cirq.final_state_vector(other_batch[i][j])
                out_arr[i][j] = np.abs(np.vdot(final_wf, internal_wf))**2

        self.assertAllClose(out, out_arr, atol=1e-5)

    @parameterized.parameters([
        {
            'n_qubits': 5,
            'batch_size': 1,
            'inner_dim_size': 5
        },
        {
            'n_qubits': 5,
            'batch_size': 2,
            'inner_dim_size': 1
        },
        {
            'n_qubits': 10,
            'batch_size': 3,
            'inner_dim_size': 2
        },
        {
            'n_qubits': 5,
            'batch_size': 10,
            'inner_dim_size': 5
        },
    ])
    def test_correctness_without_symbols(self, n_qubits, batch_size,
                                         inner_dim_size):
        """Tests that inner_product works without symbols."""
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, _ = \
            util.random_circuit_resolver_batch(
                qubits, batch_size)

        other_batch = [
            util.random_circuit_resolver_batch(qubits, inner_dim_size)[0]
            for i in range(batch_size)
        ]

        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor([[] for _ in range(batch_size)])

        out = fidelity_op.fidelity(programs, symbol_names, symbol_values,
                                   other_programs)

        out_arr = np.empty((batch_size, inner_dim_size), dtype=np.complex64)
        for i in range(batch_size):
            final_wf = cirq.final_state_vector(circuit_batch[i])
            for j in range(inner_dim_size):
                internal_wf = cirq.final_state_vector(other_batch[i][j])
                out_arr[i][j] = np.abs(np.vdot(final_wf, internal_wf))**2

        self.assertAllClose(out, out_arr, atol=1e-5)

    def test_correctness_empty(self):
        """Tests the fidelity with empty circuits."""

        empty_circuit = util.convert_to_tensor([cirq.Circuit()])
        empty_symbols = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        empty_values = tf.convert_to_tensor([[]])
        other_program = util.convert_to_tensor([[cirq.Circuit()]])

        out = fidelity_op.fidelity(empty_circuit, empty_symbols, empty_values,
                                   other_program)
        expected = np.array([[1.0]], dtype=np.complex64)
        self.assertAllClose(out, expected)

        qubit = cirq.GridQubit(0, 0)
        non_empty_circuit = util.convert_to_tensor(
            [cirq.Circuit(cirq.X(qubit))])
        empty_symbols = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        empty_values = tf.convert_to_tensor([[]])
        other_program = util.convert_to_tensor([[cirq.Circuit()]])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'qubits not found'):
            fidelity_op.fidelity(non_empty_circuit, empty_symbols, empty_values,
                                 other_program)

    @parameterized.parameters([
        {
            'n_qubits': 5,
            'batch_size': 1,
            'inner_dim_size': 1
        },
        {
            'n_qubits': 5,
            'batch_size': 3,
            'inner_dim_size': 1
        },
    ])
    def test_tf_gradient_correctness_with_symbols(self, n_qubits, batch_size,
                                                  inner_dim_size):
        """Tests that tf.gradient of inner_product works with symbols."""
        symbol_names = ['alpha', 'beta', 'gamma']
        n_params = len(symbol_names)
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
            util.random_symbol_circuit_resolver_batch(
                qubits, symbol_names, batch_size)

        other_batch = [0 for i in range(batch_size)]
        for i in range(len(other_batch)):
            other_batch[i] = copy.deepcopy(circuit_batch)
            for j in range(len(other_batch[i])):
                other_batch[i][j] = cirq.resolve_parameters(
                    circuit_batch[i], resolver_batch[i])

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names_tensor = tf.convert_to_tensor(symbol_names,
                                                   dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor(symbol_values_array)

        with tf.GradientTape() as tape:
            tape.watch(symbol_values)
            ip = fidelity_op.fidelity(programs, symbol_names_tensor,
                                      symbol_values, other_programs)
        out = tape.gradient(ip, symbol_values)

        out_arr = np.zeros((batch_size, n_params), dtype=np.complex64)
        # dx came from _GRAD_EPS of core/src/adj_util.cc
        dx = 5e-3
        for i in range(batch_size):
            for k, name in enumerate(symbol_names):
                if name in resolver_batch[i].param_dict:
                    final_circuit_p = cirq.resolve_parameters(
                        circuit_batch[i], resolver_batch[i])
                    new_resolver = copy.deepcopy(resolver_batch[i])
                    new_resolver.param_dict[name] += dx
                    final_circuit_m = cirq.resolve_parameters(
                        circuit_batch[i], new_resolver)
                    final_wf_p = cirq.final_state_vector(final_circuit_p)
                    final_wf_m = cirq.final_state_vector(final_circuit_m)
                    # Performs central finite difference.
                    for j in range(inner_dim_size):
                        internal_wf = cirq.final_state_vector(other_batch[i][j])
                        fid_a = cirq.fidelity(final_wf_p, internal_wf)
                        fid_b = cirq.fidelity(final_wf_m, internal_wf)
                        grad_fid = (fid_b - fid_a) / dx
                        out_arr[i][k] += grad_fid

        self.assertAllClose(out, out_arr, atol=1e-3)

    @parameterized.parameters([
        {
            'n_qubits': 5,
            'batch_size': 1,
            'inner_dim_size': 5
        },
        {
            'n_qubits': 5,
            'batch_size': 3,
            'inner_dim_size': 2
        },
    ])
    def test_tf_gradient_correctness_without_symbols(self, n_qubits, batch_size,
                                                     inner_dim_size):
        """Tests that tf.gradient of inner_product works without symbols."""
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, _ = \
            util.random_circuit_resolver_batch(
                qubits, batch_size)

        other_batch = [
            util.random_circuit_resolver_batch(qubits, inner_dim_size)[0]
            for i in range(batch_size)
        ]

        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor([[] for _ in range(batch_size)])

        with tf.GradientTape() as tape:
            tape.watch(symbol_values)
            ip = fidelity_op.fidelity(programs, symbol_names, symbol_values,
                                      other_programs)
        out = tape.gradient(ip, symbol_values)
        self.assertAllClose(out, tf.zeros_like(symbol_values), atol=1e-3)

    def test_correctness_no_circuit(self):
        """Test the inner product between no circuits."""

        empty_circuit = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_symbols = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_values = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        other_program = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)

        out = fidelity_op.fidelity(empty_circuit, empty_symbols, empty_values,
                                   other_program)
        self.assertShapeEqual(np.zeros((0, 0)), out)

    def test_tf_gradient_correctness_no_circuit(self):
        """Test the inner product grad between no circuits."""

        empty_circuit = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_symbols = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_values = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        other_program = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)

        with tf.GradientTape() as tape:
            tape.watch(empty_values)
            out = fidelity_op.fidelity(empty_circuit, empty_symbols,
                                       empty_values, other_program)

        self.assertShapeEqual(np.zeros((0, 0)), out)


if __name__ == "__main__":
    tf.test.main()