# Copyright 2021 The TensorFlow Quantum Authors. All Rights Reserved.
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
"""Tests that specifically target tfq_inner_product_hessian."""
import copy
import time

import numpy as np
from absl.testing import parameterized
import sympy
import tensorflow as tf
import cirq

from tensorflow_quantum.core.ops.math_ops import inner_product_op
from tensorflow_quantum.python import util

_INVERSE_SPEEDUP = 1 / 20.0
_ATOL = 0.2
_RTOL = 0.3
_SYMBOL_NAMES = [['alpha'], ['alpha', 'beta']]
_ONE_EIGEN_GATES = [
    cirq.XPowGate,
    cirq.YPowGate,
    cirq.ZPowGate,
    cirq.HPowGate,
]
_TWO_EIGEN_GATES = [
    cirq.XXPowGate,
    cirq.YYPowGate,
    cirq.ZZPowGate,
    cirq.CZPowGate,
    cirq.CNotPowGate,
    cirq.SwapPowGate,
    cirq.ISwapPowGate,
    cirq.PhasedISwapPowGate,
    cirq.FSimGate,
]
_UNSUPPORTED_GATES = [
    cirq.PhasedXPowGate,
]


def get_gate(gate, symbol_names, qubits):
    """Generates a gate operation."""
    symbols = sympy.symbols(symbol_names)
    if len(symbols) == 1:
        a, b = symbols * 2
    else:
        a, b = symbols
    if gate == cirq.PhasedXPowGate or gate == cirq.PhasedISwapPowGate:
        return [
            gate(phase_exponent=0.1 * a, exponent=0.2 * b).on(*qubits),
            gate(phase_exponent=0.3 * b, exponent=0.4 * a).on(*qubits)
        ]
    if gate == cirq.FSimGate:
        return [
            gate(theta=0.1 * a, phi=0.2 * b).on(*qubits),
            gate(theta=0.3 * b, phi=0.4 * a).on(*qubits)
        ]

    return [
        gate(exponent=0.1 * a).on(*qubits),
        gate(exponent=0.2 * b).on(*qubits)
    ]


def get_shifted_resolved_circuit(circuit, name_j, name_k, dx_j, dx_k, resolver):
    """Generates a state vector with shifted values."""
    new_resolver = copy.deepcopy(resolver)
    new_resolver.param_dict[name_j] += dx_j
    new_resolver.param_dict[name_k] += dx_k
    return cirq.resolve_parameters(circuit, new_resolver)


def get_finite_difference_hessian(circuit, name_j, name_k, resolver):
    """Generates finite difference hessian."""
    # dx came from _GRAD_EPS of core/src/adj_util.cc
    dx = 5e-3
    inv_square_two_dx = np.asarray([1e4 + 0.j], dtype=np.complex64)
    final_circuit_pp = get_shifted_resolved_circuit(circuit, name_j, name_k, dx,
                                                    dx, resolver)
    final_circuit_mp = get_shifted_resolved_circuit(circuit, name_j, name_k,
                                                    -dx, dx, resolver)
    final_circuit_pm = get_shifted_resolved_circuit(circuit, name_j, name_k, dx,
                                                    -dx, resolver)
    final_circuit_mm = get_shifted_resolved_circuit(circuit, name_j, name_k,
                                                    -dx, -dx, resolver)
    final_wf_pp = inv_square_two_dx * cirq.final_state_vector(final_circuit_pp)
    final_wf_mp = inv_square_two_dx * cirq.final_state_vector(final_circuit_mp)
    final_wf_pm = inv_square_two_dx * cirq.final_state_vector(final_circuit_pm)
    final_wf_mm = inv_square_two_dx * cirq.final_state_vector(final_circuit_mm)
    # Performs central finite difference.
    final_wf_grad = ((final_wf_pp + final_wf_mm) - (final_wf_pm + final_wf_mp))
    return final_wf_grad


class InnerProductAdjHessianTest(tf.test.TestCase, parameterized.TestCase):
    """Tests tfq_inner_product_hessian."""

    def test_inner_product_hessian_inputs(self):
        """Makes sure that inner_product_adj_hessian fails on bad inputs."""
        n_qubits = 5
        batch_size = 5
        n_other_programs = 3
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, n_qubits)
        programs_coeffs = np.ones((batch_size,))
        other_programs_coeffs = np.ones((batch_size, n_other_programs))
        circuit_batch, resolver_batch = \
          util.random_symbol_circuit_resolver_batch(
              qubits, symbol_names, batch_size)

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        other_batch = [
            util.random_circuit_resolver_batch(qubits, n_other_programs)[0]
            for _ in range(batch_size)
        ]

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'programs must be rank 1'):
            # Circuit tensor has too many dimensions.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor([circuit_batch]),
                symbol_names, symbol_values_array,
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_names must be rank 1.'):
            # symbol_names tensor has too many dimensions.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch),
                np.array([symbol_names]), symbol_values_array,
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too many dimensions.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                np.array([symbol_values_array]),
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbol_values must be rank 2.'):
            # symbol_values_array tensor has too few dimensions.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch),
                symbol_names, symbol_values_array[0],
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'other_programs must be rank 2.'):
            # other_programs tensor has too few dimensions.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch),
                symbol_names, symbol_values_array,
                util.convert_to_tensor(circuit_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'other_programs must be rank 2.'):
            # pauli_sums tensor has too many dimensions.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in other_batch]),
                programs_coeffs, other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Unparseable proto'):
            # circuit tensor has the right type but invalid values.
            inner_product_op.inner_product_hessian(
                ['junk'] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'Could not find symbol in parameter map'):
            # symbol_names tensor has the right type but invalid values.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch),
                ['junk'], symbol_values_array,
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'not found in reference circuit'):
            # other_programs tensor has the right type but operates on
            # qubits that the reference ciruit doesn't have.
            new_qubits = [cirq.GridQubit(5, 5), cirq.GridQubit(9, 9)]
            new_circuits, _ = util.random_circuit_resolver_batch(
                new_qubits, batch_size)
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in new_circuits]),
                programs_coeffs, other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'not found in paired circuit'):
            # other_programs tensor has the right type but operates on
            # qubits that the reference ciruit doesn't have.
            new_qubits = cirq.GridQubit.rect(1, n_qubits - 1)
            new_circuits, _ = util.random_circuit_resolver_batch(
                new_qubits, batch_size)
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in new_circuits]),
                programs_coeffs, other_programs_coeffs)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # circuits tensor has the wrong type.
            inner_product_op.inner_product_hessian(
                [1.0] * batch_size, symbol_names, symbol_values_array,
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # symbol_names tensor has the wrong type.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch),
                [0.1234], symbol_values_array,
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.UnimplementedError, ''):
            # symbol_values tensor has the wrong type.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch),
                symbol_names, [['junk']] * batch_size,
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(TypeError, 'Cannot convert'):
            # other_programs tensor has the wrong type.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, [[1.0]] * batch_size, programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(TypeError, 'missing'):
            # we are missing an argument.
            # pylint: disable=no-value-for-parameter
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array, programs_coeffs, other_programs_coeffs)
            # pylint: enable=no-value-for-parameter

        with self.assertRaisesRegex(TypeError, 'positional arguments'):
            # pylint: disable=too-many-function-args
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch),
                symbol_names, symbol_values_array,
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs, [])

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # batch programs has wrong batch size.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor(other_batch[:int(batch_size * 0.5)]),
                programs_coeffs, other_programs_coeffs)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    expected_regex='do not match'):
            # batch programs has wrong batch size.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array[::int(batch_size * 0.5)],
                util.convert_to_tensor(other_batch), programs_coeffs,
                other_programs_coeffs)

        with self.assertRaisesRegex(
                tf.errors.InvalidArgumentError,
                expected_regex='Found symbols in other_programs'):
            # other_programs has symbols.
            inner_product_op.inner_product_hessian(
                util.convert_to_tensor(circuit_batch), symbol_names,
                symbol_values_array,
                util.convert_to_tensor([[x] for x in circuit_batch]),
                programs_coeffs, other_programs_coeffs)

        res = inner_product_op.inner_product_hessian(
            util.convert_to_tensor(circuit_batch), symbol_names,
            symbol_values_array.astype(np.float64),
            util.convert_to_tensor(other_batch), programs_coeffs,
            other_programs_coeffs)
        self.assertDTypeEqual(res, np.complex64)

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
        """Tests that inner_product_hessian works with symbols."""
        symbol_names = ['alpha', 'beta', 'gamma']
        n_params = len(symbol_names)
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, resolver_batch = \
          util.random_symbol_circuit_resolver_batch(
              qubits, symbol_names, batch_size,
              exclude_gates=_UNSUPPORTED_GATES)

        other_batch = [
            util.random_circuit_resolver_batch(qubits, inner_dim_size)[0]
            for _ in range(batch_size)
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])

        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names_tensor = tf.convert_to_tensor(symbol_names,
                                                   dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor(symbol_values_array)
        programs_coeffs = tf.cast(tf.random.normal((batch_size,)), tf.complex64)
        other_programs_coeffs = tf.cast(
            tf.random.normal((batch_size, inner_dim_size)), tf.complex64)

        t1 = time.time()
        out = inner_product_op.inner_product_hessian(
            programs, symbol_names_tensor, symbol_values, other_programs,
            programs_coeffs, other_programs_coeffs)
        t1 = time.time() - t1

        t2 = time.time()
        out_arr = np.zeros((batch_size, n_params, n_params), dtype=np.complex64)
        for i, resolver in enumerate(resolver_batch):
            weighted_internal_wf = None
            for l, other in enumerate(other_batch[i]):
                internal_wf = (other_programs_coeffs[i][l] *
                               cirq.final_state_vector(other))
                if l == 0:
                    weighted_internal_wf = internal_wf
                else:
                    weighted_internal_wf += internal_wf
            for j, name_j in enumerate(symbol_names):
                for k, name_k in enumerate(symbol_names):
                    final_wf_grad = get_finite_difference_hessian(
                        circuit_batch[i], name_j, name_k, resolver)
                    out_arr[i][j][k] += (
                        programs_coeffs[i] *
                        np.vdot(final_wf_grad, weighted_internal_wf))

        # Elapsed time should be less than 5% of cirq version.
        # (at least 20x speedup)
        self.assertLess(t1, t2 * _INVERSE_SPEEDUP)
        self.assertAllClose(out, out_arr, atol=_ATOL, rtol=_RTOL)

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
    def correctness_without_symbols(self, n_qubits, batch_size, inner_dim_size):
        """Tests that inner_product_hessian works without symbols."""
        qubits = cirq.GridQubit.rect(1, n_qubits)
        circuit_batch, _ = \
          util.random_circuit_resolver_batch(
              qubits, batch_size)

        other_batch = [
            util.random_circuit_resolver_batch(qubits, inner_dim_size)[0]
            for _ in range(batch_size)
        ]

        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor([[] for _ in range(batch_size)])
        progams_coeffs = np.ones((batch_size,))
        other_programs_coeffs = np.ones((batch_size, inner_dim_size))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbols must be a positive integer'):
            inner_product_op.inner_product_hessian(programs, symbol_names,
                                                   symbol_values,
                                                   other_programs,
                                                   progams_coeffs,
                                                   other_programs_coeffs)

    def correctness_empty(self):
        """Tests the inner product hessian between two empty circuits."""
        symbol_names = ['alpha', 'beta']
        n_params = len(symbol_names)
        empty_cicuit = util.convert_to_tensor([cirq.Circuit()])
        empty_symbols = tf.convert_to_tensor([], dtype=tf.dtypes.string)
        empty_values = tf.convert_to_tensor([[]])
        other_program = util.convert_to_tensor([[cirq.Circuit()]])
        program_coeffs = np.ones((1,))
        other_program_coeffs = np.ones((1, 1))

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'symbols must be a positive integer'):
            inner_product_op.inner_product_hessian(empty_cicuit, empty_symbols,
                                                   empty_values, other_program,
                                                   program_coeffs,
                                                   other_program_coeffs)

        empty_cicuit = util.convert_to_tensor([cirq.Circuit()])
        symbol_names = tf.convert_to_tensor(symbol_names,
                                            dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor([[0.0 for _ in range(2)]])
        other_program = util.convert_to_tensor([[cirq.Circuit()]])

        out = inner_product_op.inner_product_hessian(empty_cicuit, symbol_names,
                                                     symbol_values,
                                                     other_program,
                                                     program_coeffs,
                                                     other_program_coeffs)
        expected = np.zeros((1, n_params, n_params), dtype=np.complex64)
        self.assertAllClose(out, expected)

    def correctness_no_circuit(self):
        """Test the inner product hessian between no circuits."""
        empty_circuit = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_symbols = tf.raw_ops.Empty(shape=(0,), dtype=tf.string)
        empty_values = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.float32)
        other_program = tf.raw_ops.Empty(shape=(0, 0), dtype=tf.string)
        empty_program_coeffs = tf.raw_ops.Empty(shape=(0,), dtype=tf.float32)
        empty_other_program_coeffs = tf.raw_ops.Empty(shape=(0, 0),
                                                      dtype=tf.float32)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'number of symbols must be a positive'):
            # When using `tf.gradients`, a user will never encounter this error
            # thanks to the `tf.cond` inside of the custom gradient.
            _ = inner_product_op.inner_product_hessian(
                empty_circuit, empty_symbols, empty_values, other_program,
                empty_program_coeffs, empty_other_program_coeffs)


class InnerProductHessianOnGates(tf.test.TestCase, parameterized.TestCase):
    """Tests inner_product_hessian on a single gate."""

    @parameterized.parameters([{
        'gate': gate,
        'symbol_names': names
    }
                               for gate in _ONE_EIGEN_GATES + _TWO_EIGEN_GATES
                               for names in _SYMBOL_NAMES])
    def correctness_one_gate_with_symbols(self, gate, symbol_names):
        """Tests that inner_product_hessian works with one gate."""
        n_params = len(symbol_names)
        qubits = cirq.GridQubit.rect(1, 2 if gate in _TWO_EIGEN_GATES else 1)
        circuit_batch = [cirq.Circuit(get_gate(gate, symbol_names, qubits))]
        resolver_batch = [
            cirq.ParamResolver({name: 0.123 for name in symbol_names})
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])
        other_batch = [
            [cirq.Circuit(cirq.H.on_each(*qubits))] for _ in circuit_batch
        ]
        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names_tensor = tf.convert_to_tensor(symbol_names,
                                                   dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor(symbol_values_array)
        programs_coeffs = tf.cast(tf.random.normal((1,)), tf.complex64)
        other_programs_coeffs = tf.cast(tf.random.normal((1, 1)), tf.complex64)

        out = inner_product_op.inner_product_hessian(
            programs, symbol_names_tensor, symbol_values, other_programs,
            programs_coeffs, other_programs_coeffs)

        out_arr = np.zeros((1, n_params, n_params), dtype=np.complex64)
        for i, resolver in enumerate(resolver_batch):
            weighted_internal_wf = None
            for l, other in enumerate(other_batch[i]):
                internal_wf = (other_programs_coeffs[i][l] *
                               cirq.final_state_vector(other))
                if l == 0:
                    weighted_internal_wf = internal_wf
                else:
                    weighted_internal_wf += internal_wf
            for j, name_j in enumerate(symbol_names):
                for k, name_k in enumerate(symbol_names):
                    final_wf_grad = get_finite_difference_hessian(
                        circuit_batch[i], name_j, name_k, resolver)
                    out_arr[i][j][k] += (
                        programs_coeffs[i] *
                        np.vdot(final_wf_grad, weighted_internal_wf))

        self.assertAllClose(out, out_arr, atol=_ATOL, rtol=_RTOL)

    @parameterized.parameters([{
        'gate': gate,
    } for gate in _UNSUPPORTED_GATES for names in _SYMBOL_NAMES])
    def unsupported_gate_with_symbols(self, gate):
        """Tests that inner_product_hessian deals with unsupported gates."""
        symbol_names = ['alpha']
        qubits = cirq.GridQubit.rect(1, 2 if gate in _TWO_EIGEN_GATES else 1)
        circuit_batch = [cirq.Circuit(get_gate(gate, symbol_names, qubits))]
        resolver_batch = [
            cirq.ParamResolver({name: 0.123 for name in symbol_names})
        ]

        symbol_values_array = np.array(
            [[resolver[symbol]
              for symbol in symbol_names]
             for resolver in resolver_batch])
        other_batch = [
            [cirq.Circuit(cirq.H.on_each(*qubits))] for _ in circuit_batch
        ]
        programs = util.convert_to_tensor(circuit_batch)
        other_programs = util.convert_to_tensor(other_batch)
        symbol_names_tensor = tf.convert_to_tensor(symbol_names,
                                                   dtype=tf.dtypes.string)
        symbol_values = tf.convert_to_tensor(symbol_values_array)
        programs_coeffs = tf.cast(tf.random.normal((1,)), tf.complex64)
        other_programs_coeffs = tf.cast(tf.random.normal((1, 1)), tf.complex64)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                    'is currently not supported'):
            _ = inner_product_op.inner_product_hessian(
                programs, symbol_names_tensor, symbol_values, other_programs,
                programs_coeffs, other_programs_coeffs)


if __name__ == "__main__":
    tf.test.main()
