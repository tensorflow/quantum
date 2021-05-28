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
"""Compute analytic gradients by using general parameter-shift rule. """
import tensorflow as tf

from tensorflow_quantum.python.differentiators import differentiator
from tensorflow_quantum.python.differentiators import parameter_shift_util


class ParameterShift(differentiator.Differentiator):
    """Calculate the general version of parameter-shift rule based gradients.

    This ParameterShift is the gradient estimator of the following paper:

    [arXiv:1905.13311](https://arxiv.org/abs/1905.13311), Gavin E. Crooks.

    This ParameterShift is used for any programs with parameterized gates.
    It internally decomposes any programs into array of gates with at most
    two distinct eigenvalues.

    >>> non_diff_op = tfq.get_expectation_op()
    >>> linear_differentiator = tfq.differentiators.ParameterShift()
    >>> # Get an expectation op, with this differentiator attached.
    >>> op = linear_differentiator.generate_differentiable_op(
    ...     analytic_op=non_diff_op
    ... )
    >>> qubit = cirq.GridQubit(0, 0)
    >>> circuit = tfq.convert_to_tensor([
    ...     cirq.Circuit(cirq.X(qubit) ** sympy.Symbol('alpha'))
    ... ])
    >>> psums = tfq.convert_to_tensor([[cirq.Z(qubit)]])
    >>> symbol_values = np.array([[0.123]], dtype=np.float32)
    >>> # Calculate tfq gradient.
    >>> symbol_values_t = tf.convert_to_tensor(symbol_values)
    >>> symbol_names = tf.convert_to_tensor(['alpha'])
    >>> with tf.GradientTape() as g:
    ...     g.watch(symbol_values_t)
    ...     expectations = op(circuit, symbol_names, symbol_values_t, psums)
    >>> # This value is now computed via the ParameterShift rule.
    >>> # https://arxiv.org/abs/1905.13311
    >>> grads = g.gradient(expectations, symbol_values_t)
    >>> grads
    tf.Tensor([[-1.1839752]], shape=(1, 1), dtype=float32)

    """

    @tf.function
    def get_gradient_circuits(self, programs, symbol_names, symbol_values):
        """See base class description."""
        # these get used a lot
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)

        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2

        # These new_programs are parameter shifted.
        # shapes: [n_symbols, n_programs, n_param_gates, n_shifts]
        (new_programs, weights, shifts,
         n_param_gates) = parameter_shift_util.parse_programs(
             programs, symbol_names, symbol_values, n_symbols)

        m_tile = n_shifts * n_param_gates * n_symbols

        # Transpose to correct shape,
        # [n_programs, n_symbols, n_param_gates, n_shifts],
        # then reshape to the correct batch size
        batch_programs = tf.reshape(tf.transpose(new_programs, [1, 0, 2, 3]),
                                    [n_programs, m_tile])
        batch_weights = tf.reshape(
            tf.transpose(weights, [1, 0, 2, 3]),
            [n_programs, n_symbols, n_param_gates * n_shifts])
        shifts = tf.reshape(tf.transpose(shifts, [1, 0, 2, 3]),
                            [n_programs, m_tile, 1])

        # Append impurity symbol into symbol name
        new_symbol_names = tf.concat([
            symbol_names,
            tf.constant([parameter_shift_util.PARAMETER_IMPURITY_NAME])
        ], 0)

        # Symbol values are the input symbol values, tiled according to
        # `batch_programs`, with the shift values appended.
        tiled_symbol_values = tf.tile(tf.expand_dims(symbol_values, 1),
                                      [1, m_tile, 1])
        batch_symbol_values = tf.concat([tiled_symbol_values, shifts], 2)

        single_program_mapper = tf.reshape(
            tf.range(n_symbols * n_param_gates * n_shifts),
            [n_symbols, n_param_gates * n_shifts])
        batch_mapper = tf.tile(tf.expand_dims(single_program_mapper, 0),
                               [n_programs, 1, 1])

        return (batch_programs, new_symbol_names, batch_symbol_values,
                batch_weights, batch_mapper)
