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
"""Basic tests for utility functions for SGDifferentiator"""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
import sympy
import cirq

from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import \
    stochastic_differentiator_util as sd_util
from tensorflow_quantum.python.differentiators import parameter_shift_util


def _example_circuit_helper(n_qubits, n_programs):
    n_shifts = 2
    symbol_names = ['a', 'b']
    n_symbols = len(symbol_names)
    sympy_symbols = [sympy.Symbol(s) for s in symbol_names]
    coeff = [1.0, -2.0, 3.0, -4.0, 5.0]
    q = cirq.GridQubit.rect(1, n_qubits)
    c = cirq.Circuit([
        cirq.rz(coeff[i] * sympy_symbols[i % 2]).on(q[i])
        for i in range(n_qubits)
    ])
    circuit_batch = [c] * n_programs
    symbol_values_array = np.array(
        [[i for i, _ in enumerate(symbol_names)] for _ in range(n_programs)],
        dtype=np.float32)

    symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
    programs = util.convert_to_tensor(circuit_batch)
    return programs, symbol_values_tensor, n_symbols, n_shifts


def _example_ops_helper(n_programs, n_ops):
    coeffs = [[1.0, -2.0, 3.0], [-4.0, 5.0]]
    n_qubits = 3
    q = cirq.GridQubit.rect(1, n_qubits)
    cirq_op_list = [cirq.X, cirq.Y, cirq.Z]

    def get_term_with_coefficient(coeff_list):
        # Test with multiple `cirq.PauliString`'s
        return [
            cirq.PauliString({
                q[i]: cirq_op_list[i],
            }, coefficient=coeff) for i, coeff in enumerate(coeff_list)
        ]

    psums = [[
        cirq.PauliSum.from_pauli_strings(get_term_with_coefficient(coeffs[i]))
        for i in range(n_ops)
    ]
             for _ in range(n_programs)]
    ops = util.convert_to_tensor(psums)
    return ops, psums, coeffs


class SGDifferentiatorUtilTest(tf.test.TestCase, parameterized.TestCase):
    """Test the stochastic_differentiator_util module."""

    @parameterized.parameters([{'eps': 1e-7}])
    def test_get_parse_pauli_sums(self, eps):
        """Input & output check for _get_parse_pauli_sums()."""
        n_programs = 3
        n_ops = 2
        ops, psums, coeffs = _example_ops_helper(n_programs, n_ops)

        parser = sd_util._get_parse_pauli_sums()

        # input should be tensorflow tensor.
        with self.assertRaises(ValueError):
            # psums is used instead of ops.
            parser(psums, n_programs, n_ops)

        observable_coeff = parser(ops, n_programs, n_ops)
        # shape check
        tf.assert_equal([n_programs, n_ops], tf.shape(observable_coeff))
        # value check
        true_coeff = np.array(
            [np.sum(np.abs(coeff_list)) for coeff_list in coeffs])
        self.assertAllClose(np.ones([n_programs, n_ops]) * true_coeff,
                            observable_coeff,
                            atol=eps,
                            rtol=eps)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'uniform_sampling': [True, False],
                'eps': [1e-6]
            })))
    def test_get_pdist_cost(self, uniform_sampling, eps):
        """Input & output check for _get_pdist_cost()."""
        n_programs = 3
        n_ops = 2
        ops, psums, _ = _example_ops_helper(n_programs, n_ops)

        parser = sd_util._get_parse_pauli_sums()

        # input should be tensorflow tensor.
        with self.assertRaises(ValueError):
            # psums is used instead of ops.
            parser(psums, n_programs, n_ops)

        observable_coeff = parser(ops, n_programs, n_ops)

        correction_factor_ops, pdist = \
            sd_util._get_pdist_cost(observable_coeff, uniform_sampling)
        if uniform_sampling:
            ground_truth_correction_factor = np.array([[2.0, 2.0]])
            ground_truth_pdist = np.array([[0.5, 0.5]])
        else:
            ground_truth_correction_factor = np.array([[2.5, 5.0 / 3.0]])
            # pdist is weighted by each coefficients.
            ground_truth_pdist = np.array([[0.4, 0.6]])

        self.assertAllClose(ground_truth_correction_factor,
                            correction_factor_ops,
                            atol=eps,
                            rtol=eps)
        self.assertAllClose(ground_truth_pdist, pdist, atol=eps, rtol=eps)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'uniform_sampling': [True, False],
                'eps': [0.1]
            })))
    def test_stochastic_cost_preprocessor(self, uniform_sampling, eps):
        """Input & output check for stochastic_cost_preprocessor().
        The consistency of the estimated average gradient is checked by:
        //benchmarks/scripts/differentiators:convergence_test"""
        n_programs = 3
        n_ops = 2
        ops, psums, _ = _example_ops_helper(n_programs, n_ops)

        # all inputs should be tensorflow tensors.
        with self.assertRaises(ValueError):
            # psums is used instead of ops.
            new_pauli_sums, cost_relocator, n_ops = \
                sd_util.stochastic_cost_preprocessor(
                    psums, n_programs, n_ops, uniform_sampling)

        new_pauli_sums, cost_relocator, new_n_ops = \
            sd_util.stochastic_cost_preprocessor(
                ops, n_programs, n_ops, uniform_sampling)
        # n_ops should be 1 because the only one op is sampled.
        self.assertEqual(new_n_ops, 1, "n_ops should be 1")
        ground_truth_shape = np.array([n_programs, new_n_ops], dtype=np.int32)
        tf.assert_equal(ground_truth_shape, tf.shape(new_pauli_sums))
        ground_truth_shape = np.array([n_programs, n_ops], dtype=np.int32)
        tf.assert_equal(ground_truth_shape, tf.shape(cost_relocator))

        if uniform_sampling:
            ground_truth_pdist = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
            ground_truth_cost_relocator = [[2.0, 0.0], [0.0, 2.0]]

        else:
            ground_truth_pdist = [[0.4, 0.6], [0.4, 0.6], [0.4, 0.6]]
            ground_truth_cost_relocator = [[2.5, 0.0], [0.0, 5 / 3.0]]

        # Sampling ops and estimate probabilistic distribution of them.
        cost_relocator_hist = np.zeros((n_programs, n_ops))
        n_samples = 700
        for _ in range(n_samples):
            _, cost_relocator, _ = sd_util.stochastic_cost_preprocessor(
                ops, n_programs, n_ops, uniform_sampling)
            for i, cost_per_program in enumerate(cost_relocator):
                loc = np.where(
                    np.isclose(ground_truth_cost_relocator,
                               cost_per_program))[0][0]
                cost_relocator_hist[i][loc] += 1.0

        pdist = cost_relocator_hist / n_samples
        self.assertAllClose(ground_truth_pdist, pdist, atol=eps, rtol=eps)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'uniform_sampling': [True, False],
                'eps': [1e-6]
            })))
    def test_get_pdist_shifts(self, uniform_sampling, eps):
        """value check of _get_pdist_shifts()"""
        weights = np.array([[[[0.5, -0.5], [1.5, -1.5], [2.5, -2.5]],
                             [[0.5, -0.5], [1.5, -1.5], [2.5, -2.5]],
                             [[0.5, -0.5], [1.5, -1.5], [2.5, -2.5]]],
                            [[[-1., 1.], [-2., 2.], [0., -0.]],
                             [[-1., 1.], [-2., 2.], [0., -0.]],
                             [[-1., 1.], [-2., 2.], [0., -0.]]]])
        # Transpose to [n_symbols, n_programs, n_shifts, n_param_gates]
        weights = np.transpose(weights, [0, 1, 3, 2])
        # Reshape to [sub_total_programs, n_param_gates]
        sub_total_programs = np.prod(weights.shape[:-1])
        n_param_gates = weights.shape[-1]
        weights = np.reshape(weights, [sub_total_programs, n_param_gates])

        corrected_weights, pdist = \
            sd_util._get_pdist_shifts(weights, uniform_sampling)
        if uniform_sampling:
            ground_truth_corrected_weights = np.array([[1.5, 4.5, 7.5],
                                                       [-1.5, -4.5, -7.5],
                                                       [1.5, 4.5, 7.5],
                                                       [-1.5, -4.5, -7.5],
                                                       [1.5, 4.5, 7.5],
                                                       [-1.5, -4.5, -7.5],
                                                       [-2.0, -4.0, 0.0],
                                                       [2.0, 4.0, -0.0],
                                                       [-2.0, -4.0, 0.0],
                                                       [2.0, 4.0, -0.0],
                                                       [-2.0, -4.0, 0.0],
                                                       [2.0, 4.0, -0.0]])
            ground_truth_pdist = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                                           [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                                           [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                                           [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                                           [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                                           [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                                           [0.5, 0.5, 0.0], [0.5, 0.5, 0.0],
                                           [0.5, 0.5, 0.0], [0.5, 0.5, 0.0],
                                           [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
        else:
            ground_truth_corrected_weights = np.array([[4.5, 4.5, 4.5],
                                                       [-4.5, -4.5, -4.5],
                                                       [4.5, 4.5, 4.5],
                                                       [-4.5, -4.5, -4.5],
                                                       [4.5, 4.5, 4.5],
                                                       [-4.5, -4.5, -4.5],
                                                       [-3.0, -3.0, 0.0],
                                                       [3.0, 3.0, -0.0],
                                                       [-3.0, -3.0, 0.0],
                                                       [3.0, 3.0, -0.0],
                                                       [-3.0, -3.0, 0.0],
                                                       [3.0, 3.0, -0.0]])
            # pdist is weighted by each coefficients.
            ground_truth_pdist = np.array(
                [[1.0 / 9.0, 1.0 / 3.0, 5.0 / 9.0],
                 [1.0 / 9.0, 1.0 / 3.0, 5.0 / 9.0],
                 [1.0 / 9.0, 1.0 / 3.0, 5.0 / 9.0],
                 [1.0 / 9.0, 1.0 / 3.0, 5.0 / 9.0],
                 [1.0 / 9.0, 1.0 / 3.0, 5.0 / 9.0],
                 [1.0 / 9.0, 1.0 / 3.0, 5.0 / 9.0], [1.0 / 3.0, 2.0 / 3.0, 0.0],
                 [1.0 / 3.0, 2.0 / 3.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0],
                 [1.0 / 3.0, 2.0 / 3.0, 0.0], [1.0 / 3.0, 2.0 / 3.0, 0.0],
                 [1.0 / 3.0, 2.0 / 3.0, 0.0]],
                dtype=np.float32)

        self.assertAllClose(ground_truth_corrected_weights,
                            corrected_weights,
                            atol=eps,
                            rtol=eps)
        self.assertAllClose(ground_truth_pdist, pdist, atol=eps, rtol=eps)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'uniform_sampling': [True, False],
                'eps': [0.1]
            })))
    def test_stochastic_generator_preprocessor(self, uniform_sampling, eps):
        """Input & output check for stochastic_generator_preprocessor().
        The consistency of the estimated average gradient is checked by:
        //benchmarks/scripts/differentiators:convergence_test"""
        n_qubits = 5
        n_programs = 3
        symbol_names = ['a', 'b']

        programs, symbol_values_tensor, n_symbols, n_shifts = \
            _example_circuit_helper(n_qubits, n_programs)

        new_programs_before, weights_before, shifts_before, \
        n_param_gates_before = parameter_shift_util._parameter_shift_parse_programs(
            programs, symbol_names, symbol_values_tensor, n_symbols)

        new_programs, weights, shifts, n_param_gates = \
            sd_util.stochastic_generator_preprocessor(
                new_programs_before, weights_before, shifts_before, n_programs,
                n_symbols, n_param_gates_before, n_shifts, uniform_sampling)

        # n_param_gates should be 1 because the only one generator is sampled.
        self.assertEqual(n_param_gates, 1, "n_param_gates should be 1")
        ground_truth_shape = np.array(
            [n_symbols, n_programs, n_param_gates, n_shifts], dtype=np.int32)
        tf.assert_equal(ground_truth_shape, tf.shape(new_programs))
        tf.assert_equal(ground_truth_shape, tf.shape(weights))
        tf.assert_equal(ground_truth_shape, tf.shape(shifts))

        # Estimate probability of sampling each shifts
        ground_truth_shifts = [[[1.5707964, -1.5707964],
                                [0.5235988, -0.5235988],
                                [0.31415927, -0.31415927]],
                               [[0.21460181, 1.7853982], [0.6073009, 1.3926991],
                                [1.0, 1.0]]]
        if uniform_sampling:
            ground_truth_pdist = [[0.333333, 0.333333, 0.333333],
                                  [0.5, 0.5, 0.0]]
        else:
            ground_truth_pdist = [[0.111111, 0.333333, 0.555555],
                                  [0.333333, 0.666666, 0.0]]

        shifts_hist = np.zeros((n_symbols, n_programs))
        n_samples = 700
        for _ in range(n_samples):
            _, _, shifts, _ = \
                sd_util.stochastic_generator_preprocessor(
                    new_programs_before, weights_before, shifts_before,
                    n_programs, n_symbols, n_param_gates_before, n_shifts,
                    uniform_sampling)
            for i, shifts_per_symbol in enumerate(shifts):
                for s in shifts_per_symbol:  # per program
                    loc = np.where(np.isclose(ground_truth_shifts, s))[1][0]
                    shifts_hist[i][loc] += 1.0

        shifts_pdist = shifts_hist / n_samples / n_programs
        self.assertAllClose(ground_truth_pdist,
                            shifts_pdist,
                            atol=eps,
                            rtol=eps)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'uniform_sampling': [True, False],
                'eps': [1e-6]
            })))
    def test_get_pdist_symbols(self, uniform_sampling, eps):
        """value check of _get_pdist_symbols()"""
        weights = np.array([[[[0.5, -0.5], [1.5, -1.5], [2.5, -2.5]],
                             [[0.5, -0.5], [1.5, -1.5], [2.5, -2.5]],
                             [[0.5, -0.5], [1.5, -1.5], [2.5, -2.5]]],
                            [[[-1., 1.], [-2., 2.], [0., -0.]],
                             [[-1., 1.], [-2., 2.], [0., -0.]],
                             [[-1., 1.], [-2., 2.], [0., -0.]]]])
        # Transpose to [n_param_gates, n_shifts, n_programs, n_symbols]
        weights = np.transpose(weights, [1, 2, 3, 0])
        # Reshape to [sub_total_programs, n_param_gates]
        sub_total_programs = np.prod(weights.shape[:-1])
        n_symbols = weights.shape[-1]
        weights = np.reshape(weights, [sub_total_programs, n_symbols])

        corrected_weights, pdist = sd_util._get_pdist_symbols(
            weights, uniform_sampling)
        # In this case, both pdist's of uniform_sampling=True & False are equal.
        ground_truth_corrected_weights = np.array([[0.8333333,
                                                    -2.5], [-0.8333333, 2.5],
                                                   [2.5, -5.0], [-2.5, 5.0],
                                                   [4.1666665, 0.0],
                                                   [-4.1666665, -0.0],
                                                   [0.8333333,
                                                    -2.5], [-0.8333333, 2.5],
                                                   [2.5, -5.0], [-2.5, 5.0],
                                                   [4.1666665, 0.0],
                                                   [-4.1666665, -0.0],
                                                   [0.8333333, -2.5],
                                                   [-0.8333333, 2.5],
                                                   [2.5, -5.0], [-2.5, 5.0],
                                                   [4.1666665, 0.0],
                                                   [-4.1666665, -0.0]])
        ground_truth_pdist = np.array([[0.6, 0.4]])

        self.assertAllClose(ground_truth_corrected_weights,
                            corrected_weights,
                            atol=eps,
                            rtol=eps)
        self.assertAllClose(ground_truth_pdist, pdist, atol=eps, rtol=eps)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(**{
                'uniform_sampling': [True, False],
                'eps': [0.1]
            })))
    def test_stochastic_coordinate_preprocessor(self, uniform_sampling, eps):
        """Input & output check for stochastic_coordinate_preprocessor().
        The consistency of the estimated average gradient is checked by:
        //benchmarks/scripts/differentiators:convergence_test"""
        n_qubits = 5
        n_programs = 3
        symbol_names = ['a', 'b']

        programs, symbol_values_tensor, n_symbols, n_shifts = \
            _example_circuit_helper(n_qubits, n_programs)

        n_ops = 2
        ops, psums, _ = _example_ops_helper(n_programs, n_ops)

        new_programs, weights_before, shifts, n_param_gates = \
            parameter_shift_util._parameter_shift_parse_programs(
                programs, symbol_names, symbol_values_tensor, n_symbols)

        # all inputs should be tensorflow tensors.
        with self.assertRaises(ValueError):
            # symbol_values_array is used instead of symbol_values_tensor.
            sd_util.stochastic_coordinate_preprocessor(
                new_programs, symbol_values_tensor.numpy(), ops, weights_before,
                shifts, n_programs, n_symbols, n_param_gates, n_shifts, n_ops,
                uniform_sampling)
            # psums is used instead of ops.
            sd_util.stochastic_coordinate_preprocessor(
                new_programs, symbol_values_tensor, psums, weights_before,
                shifts, n_programs, n_symbols, n_param_gates, n_shifts, n_ops,
                uniform_sampling)

        flat_programs, flat_perturbations, flat_ops, _, weights, \
        coordinate_relocator = \
            sd_util.stochastic_coordinate_preprocessor(
                new_programs, symbol_values_tensor, ops, weights_before,
                shifts, n_programs, n_symbols, n_param_gates, n_shifts,
                n_ops, uniform_sampling)

        # n_symbols should not be 1 because it doesn't fit the input format of
        # expectation_op or sampling_op.
        total_programs = n_programs * n_param_gates * n_shifts
        # flat_programs should have n_programs * n_param_gates * n_shifts * 1
        # because only one symbol is sampled now.
        self.assertAllClose([total_programs],
                            tf.shape(flat_programs),
                            atol=eps,
                            rtol=eps)
        # perturbation symbol is added, so the number of symbol should be
        # n_symbol+1
        self.assertAllClose([total_programs, n_symbols + 1],
                            tf.shape(flat_perturbations),
                            atol=eps,
                            rtol=eps)
        # shape check on flat_ops.
        self.assertAllClose([total_programs, n_ops],
                            tf.shape(flat_ops),
                            atol=eps,
                            rtol=eps)
        # resampled weights is in
        # [n_symbols, n_param_gates, n_shifts, n_programs]
        self.assertAllClose([n_symbols, n_param_gates, n_shifts, n_programs],
                            tf.shape(weights),
                            atol=eps,
                            rtol=eps)
        # resampled coordinate_relocator is in [total_programs, n_symbols]
        self.assertAllClose([total_programs, n_symbols],
                            tf.shape(coordinate_relocator),
                            atol=eps,
                            rtol=eps)

        # Estimate probability of sampling each shifts
        ground_truth_shifts = [[
            1.5707964, -1.5707964, 0.5235988, -0.5235988, 0.31415927,
            -0.31415927
        ], [0.21460181, 1.7853982, 0.6073009, 1.3926991, 1.0, 1.0]]

        ground_truth_pdist = [0.6, 0.4]

        shifts_hist = np.zeros((n_symbols,))
        n_samples = 700
        cnt = 0.0
        for _ in range(n_samples):
            _, flat_perturbations, _, _, _, _ = \
                sd_util.stochastic_coordinate_preprocessor(
                    new_programs, symbol_values_tensor, ops, weights_before,
                    shifts, n_programs, n_symbols, n_param_gates, n_shifts,
                    n_ops, uniform_sampling)

            for s in flat_perturbations[:, -1]:  # See only shift symbols.
                sym = np.where(np.isclose(ground_truth_shifts, s))[0][0]
                shifts_hist[sym] += 1.0
                cnt += 1.0

        shifts_pdist = shifts_hist / cnt
        self.assertAllClose(ground_truth_pdist,
                            shifts_pdist,
                            atol=eps,
                            rtol=eps)


if __name__ == "__main__":
    tf.test.main()
