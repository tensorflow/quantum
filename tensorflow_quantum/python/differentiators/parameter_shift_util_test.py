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
"""Basic tests for utility functions for ParameterShift"""
import cirq
import numpy as np
import sympy
import tensorflow as tf
from absl.testing import parameterized

from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import parameter_shift_util


class ParameterShiftUtilTest(tf.test.TestCase, parameterized.TestCase):
    """Test the parameter_shift_util module."""

    def test_parse_programs(self):
        """Input & output check for parse_programs()."""
        n_qubits = 5
        n_programs = 3
        n_shifts = 2
        symbol_names = ['a', 'b']
        n_symbols = len(symbol_names)
        sympy_symbols = [sympy.Symbol(s) for s in symbol_names]
        coeff = [1.0, -2.0, 3.0, -4.0, 5.0]
        # Test circuit.
        # (0, 0): ───Rz(1.0*a)────
        #
        # (0, 1): ───Rz(-2.0*b)───
        #
        # (0, 2): ───Rz(3.0*a)────
        #
        # (0, 3): ───Rz(-4.0*b)───
        #
        # (0, 4): ───Rz(5.0*a)────
        q = cirq.GridQubit.rect(1, n_qubits)
        c = cirq.Circuit()
        c.append([
            cirq.Rz(coeff[i] * sympy_symbols[i % 2]).on(q[i])
            for i in range(n_qubits)
        ])
        circuit_batch = [c] * n_programs
        symbol_values_array = np.array(
            [[i for i, _ in enumerate(symbol_names)] for _ in range(n_programs)
            ],
            dtype=np.float32)

        symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
        programs = util.convert_to_tensor(circuit_batch)

        new_programs, weights, shifts, n_param_gates = \
            parameter_shift_util.parse_programs(
                programs, symbol_names, symbol_values_tensor, n_symbols)

        # shape check
        ground_truth_shape = [n_symbols, n_programs, n_param_gates, n_shifts]
        tf.assert_equal(ground_truth_shape, tf.shape(new_programs))
        tf.assert_equal(ground_truth_shape, tf.shape(weights))
        tf.assert_equal(ground_truth_shape, tf.shape(shifts))

        # value check (1) weights
        # the first 1x3x3x2 are +/- coefficients of Rz gates with symbol 'a'.
        # they are divided by 2 in Rz.
        # [:,:,:,0] have original coefficient and [:,:,:,1] are their negatives.
        # the second 1x3x3x2 are with symbol 'b'. As we know, there are only
        # 2 'b' symbols, which makes [1,:,2,:] are zeros. (padded)
        ground_truth_weights = np.array([[[[0.5, -0.5], [1.5, -1.5],
                                           [2.5, -2.5]],
                                          [[0.5, -0.5], [1.5, -1.5],
                                           [2.5, -2.5]],
                                          [[0.5, -0.5], [1.5, -1.5],
                                           [2.5, -2.5]]],
                                         [[[-1., 1.], [-2., 2.], [0., -0.]],
                                          [[-1., 1.], [-2., 2.], [0., -0.]],
                                          [[-1., 1.], [-2., 2.], [0., -0.]]]])
        self.assertAllClose(ground_truth_weights, weights)
        # value check (2) shifts
        # Please ignore this divide-by-zero warning because it is intended.
        ground_truth_shifts = np.divide(1, ground_truth_weights) / 4.0 * np.pi
        new_symbol_values_array = np.tile(
            np.expand_dims(np.expand_dims(np.transpose(symbol_values_array,
                                                       [1, 0]),
                                          axis=-1),
                           axis=-1), [1, 1, 3, 2])
        # All inf's should be 0.0. This happens inside parse_programs()
        # with tf.math.divide_no_nan() without any warning.
        ground_truth_shifts[np.where(np.isinf(ground_truth_shifts))] = 0.0
        ground_truth_shifts = new_symbol_values_array + ground_truth_shifts
        self.assertAllClose(ground_truth_shifts, shifts)


if __name__ == "__main__":
    tf.test.main()
