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
"""Compute gradients by combining function values linearly."""
import numbers

import numpy as np
import tensorflow as tf

from tensorflow_quantum.python.differentiators import differentiator


class LinearCombination(differentiator.Differentiator):
    """Differentiate a circuit with respect to its inputs by
    linearly combining values obtained by evaluating the op using parameter
    values perturbed about their forward-pass values.


    >>> my_op = tfq.get_expectation_op()
    >>> weights = [5, 6, 7]
    >>> perturbations = [0, 0.5, 0.25]
    >>> linear_differentiator = tfq.differentiators.LinearCombination(
    ...    weights, perturbations
    ... )
    >>> # Get an expectation op, with this differentiator attached.
    >>> op = linear_differentiator.generate_differentiable_op(
    ...     analytic_op=my_op
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
    ...     expectations = op(circuit, symbol_names, symbol_values_t, psums
    ... )
    >>> # Gradient would be: 5 * f(x+0) + 6 * f(x+0.5) + 7 * f(x+0.25)
    >>> grads = g.gradient(expectations, symbol_values_t)
    >>> # Note: this gradient visn't correct in value, but showcases
    >>> # the principle of how gradients can be defined in a very flexible
    >>> # fashion.
    >>> grads
    tf.Tensor([[5.089467]], shape=(1, 1), dtype=float32)

    """

    def __init__(self, weights, perturbations):
        """Instantiate this differentiator.

        Create a LinearComobinationDifferentiator. Pass in weights and
        perturbations as described below.

        Args:
            weights: Python `list` of real numbers representing linear
                combination coeffecients for each perturbed function
                evaluation.
            perturbations: Python `list` of real numbers representing
                perturbation values.
        """
        if not isinstance(weights, (np.ndarray, list, tuple)):
            raise TypeError("weights must be a numpy array, list or tuple."
                            "Got {}".format(type(weights)))
        if not all([isinstance(weight, numbers.Real) for weight in weights]):
            raise TypeError("Each weight in weights must be a real number.")
        if not isinstance(perturbations, (np.ndarray, list, tuple)):
            raise TypeError("perturbations must be a numpy array,"
                            " list or tuple. Got {}".format(type(weights)))
        if not all([
                isinstance(perturbation, numbers.Real)
                for perturbation in perturbations
        ]):
            raise TypeError("Each perturbation in perturbations must be a"
                            " real number.")
        if not len(weights) == len(perturbations):
            raise ValueError("weights and perturbations must have the same "
                             "length.")
        if len(perturbations) < 2:
            raise ValueError("Must specify at least two perturbations. "
                             "Providing only one perturbation is the same as "
                             "evaluating the circuit at a single location, "
                             "which is insufficient for differentiation.")
        self.weights = tf.constant(weights, dtype=tf.float32)
        self.n_perturbations = tf.constant(len(perturbations))
        self.perturbations = tf.constant(perturbations, dtype=tf.float32)

        # Uniqueness in particular ensures there at most one zero perturbation.
        if not len(list(set(perturbations))) == len(perturbations):
            raise ValueError("All values in perturbations must be unique.")
        mask = tf.not_equal(self.perturbations,
                            tf.zeros_like(self.perturbations))
        self.non_zero_weights = tf.boolean_mask(self.weights, mask)
        self.zero_weights = tf.boolean_mask(self.weights,
                                            tf.math.logical_not(mask))
        self.non_zero_perturbations = tf.boolean_mask(self.perturbations, mask)
        self.n_non_zero_perturbations = tf.gather(
            tf.shape(self.non_zero_perturbations), 0)

    @tf.function
    def get_gradient_circuits(self, programs, symbol_names, symbol_values):
        """See base class description."""
        n_programs = tf.gather(tf.shape(programs), 0)
        n_symbols = tf.gather(tf.shape(symbol_names), 0)

        # A new copy of each program is run for each symbol and each
        # non-zero perturbation, plus one more if there is a zero perturbation.
        # `m` represents the last index of the batch mapper.
        base_m_tile = n_symbols * self.n_non_zero_perturbations
        m_tile = tf.cond(self.n_non_zero_perturbations < self.n_perturbations,
                         lambda: base_m_tile + 1, lambda: base_m_tile)
        batch_programs = tf.tile(tf.expand_dims(programs, 1), [1, m_tile])

        # LinearCombination does not add new symbols to the gradient circuits.
        new_symbol_names = tf.identity(symbol_names)

        # Build the symbol value perturbations for a single input program.
        perts_zeros_pad = tf.zeros([self.n_non_zero_perturbations],
                                   dtype=tf.float32)
        stacked_perts = tf.stack([perts_zeros_pad, self.non_zero_perturbations])
        # Identity matrix lets us tile the perturbations and simultaneously
        # put zeros in all the symbol locations not being perturbed.
        gathered_perts = tf.gather(stacked_perts,
                                   tf.eye(n_symbols, dtype=tf.int32))
        transposed_perts = tf.transpose(gathered_perts, [0, 2, 1])
        reshaped_perts = tf.reshape(transposed_perts, [base_m_tile, n_symbols])
        symbol_zeros_pad = tf.zeros([1, n_symbols])
        single_program_perts = tf.cond(
            self.n_non_zero_perturbations < self.n_perturbations,
            lambda: tf.concat([symbol_zeros_pad, reshaped_perts], 0),
            lambda: reshaped_perts)
        # Make a copy of the perturbations tensor for each input program.
        all_perts = tf.tile(tf.expand_dims(single_program_perts, 0),
                            [n_programs, 1, 1])
        # Apply perturbations to the forward pass symbol values.
        bare_symbol_values = tf.tile(tf.expand_dims(symbol_values, 1),
                                     [1, m_tile, 1])
        batch_symbol_values = bare_symbol_values + all_perts

        # The weights for all the programs.
        tiled_weights = tf.tile(tf.expand_dims(self.non_zero_weights, 0),
                                [n_symbols, 1])
        tiled_zero_weights = tf.tile(tf.expand_dims(self.zero_weights, 0),
                                     [n_symbols, 1])
        single_program_weights = tf.concat([tiled_zero_weights, tiled_weights],
                                           1)
        # Mapping is also the same for each program.
        batch_weights = tf.tile(tf.expand_dims(single_program_weights, 0),
                                [n_programs, 1, 1])

        # The mapper selects the zero weight if it exists.
        single_program_mapper_base = tf.reshape(
            tf.range(n_symbols * self.n_non_zero_perturbations),
            [n_symbols, self.n_non_zero_perturbations])
        single_program_mapper = tf.cond(
            self.n_non_zero_perturbations < self.n_perturbations,
            lambda: tf.concat([
                tf.zeros([n_symbols, 1], dtype=tf.int32),
                single_program_mapper_base + 1
            ], 1), lambda: single_program_mapper_base)
        batch_mapper = tf.tile(tf.expand_dims(single_program_mapper, 0),
                               [n_programs, 1, 1])

        return (batch_programs, new_symbol_names, batch_symbol_values,
                batch_weights, batch_mapper)


class ForwardDifference(LinearCombination):
    """Differentiate a circuit using forward differencing.

    Forward differencing computes a derivative at a point x using only
    points larger than x (in this way, it is 'one sided'). A closed form for
    the coefficients of this derivative for an arbitrary positive error order
    is used here, which is described in the following article:
    https://www.sciencedirect.com/science/article/pii/S0377042799000886.


    >>> my_op = tfq.get_expectation_op()
    >>> linear_differentiator = tfq.differentiators.ForwardDifference(2, 0.01)
    >>> # Get an expectation op, with this differentiator attached.
    >>> op = linear_differentiator.generate_differentiable_op(
    ...     analytic_op=my_op
    ... )
    >>> qubit = cirq.GridQubit(0, 0)
    >>> circuit = tfq.convert_to_tensor([
    ...     cirq.Circuit(cirq.X(qubit) ** sympy.Symbol('alpha'))
    ... ])
    >>> psums = tfq.convert_to_tensor([[cirq.Z(qubit)]])
    >>> symbol_values_array = np.array([[0.123]], dtype=np.float32)
    >>> # Calculate tfq gradient.
    >>> symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
    >>> with tf.GradientTape() as g:
    ...     g.watch(symbol_values_tensor)
    ...     expectations = op(circuit, ['alpha'], symbol_values_tensor, psums)
    >>> # Gradient would be: -50 * f(x + 0.02) +  200 * f(x + 0.01) - 150 * f(x)
    >>> grads = g.gradient(expectations, symbol_values_tensor)
    >>> grads
    tf.Tensor([[-1.184372]], shape=(1, 1), dtype=float32)

    """

    def __init__(self, error_order=1, grid_spacing=0.001):
        """Instantiate a ForwardDifference.

        Create a ForwardDifference differentiator, passing along an error order
        and grid spacing to be used to contstruct differentiator coeffecients.

        Args:
            error_order: A positive `int` specifying the error order of this
                differentiator. This corresponds to the smallest power
                of `grid_spacing` remaining in the series that was truncated
                to generate this finite differencing expression.
            grid_spacing: A positive `float` specifying how large of a
                grid to use in calculating this finite difference.
        """
        if not (isinstance(error_order, numbers.Integral) and error_order > 0):
            raise ValueError("error_order must be a positive integer.")
        if not (isinstance(grid_spacing, numbers.Real) and grid_spacing > 0):
            raise ValueError("grid_spacing must be a positive real number.")
        self.error_order = error_order
        self.grid_spacing = grid_spacing
        grid_points_to_eval = np.arange(0, error_order + 1)
        weights = []
        for point in grid_points_to_eval:
            if point == 0:
                weight = -1 * np.sum(
                    [1 / j for j in np.arange(1, error_order + 1)])
            else:
                weight = ((-1) ** (point+1) * np.math.factorial(error_order))/\
                         (point * np.math.factorial(error_order-point)
                          * np.math.factorial(point))
            weights.append(weight / grid_spacing)
        super().__init__(weights, grid_points_to_eval * grid_spacing)


class CentralDifference(LinearCombination):
    """Differentiates a circuit using Central Differencing.

    Central differencing computes a derivative at point x using an equal
    number of points before and after x. A closed form for
    the coefficients of this derivative for an arbitrary positive error order
    is used here, which is described in the following article:
    https://www.sciencedirect.com/science/article/pii/S0377042799000886.


    >>> my_op = tfq.get_expectation_op()
    >>> linear_differentiator = tfq.differentiators.CentralDifference(2, 0.01)
    >>> # Get an expectation op, with this differentiator attached.
    >>> op = linear_differentiator.generate_differentiable_op(
    ...     analytic_op=my_op
    ... )
    >>> qubit = cirq.GridQubit(0, 0)
    >>> circuit = tfq.convert_to_tensor([
    ...     cirq.Circuit(cirq.X(qubit) ** sympy.Symbol('alpha'))
    ... ])
    >>> psums = tfq.convert_to_tensor([[cirq.Z(qubit)]])
    >>> symbol_values_array = np.array([[0.123]], dtype=np.float32)
    >>> # Calculate tfq gradient.
    >>> symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
    >>> with tf.GradientTape() as g:
    ...     g.watch(symbol_values_tensor)
    ...     expectations = op(circuit, ['alpha'], symbol_values_tensor, psums)
    >>> # Gradient would be: -50 * f(x + 0.02) +  200 * f(x + 0.01) - 150 * f(x)
    >>> grads = g.gradient(expectations, symbol_values_tensor)
    >>> grads
    tf.Tensor([[-1.1837807]], shape=(1, 1), dtype=float32)

    """

    def __init__(self, error_order=2, grid_spacing=0.001):
        """Instantiate a CentralDifference.

        Create a CentralDifference differentaitor, passing along an error order
        and grid spacing to be used to contstruct differentiator coeffecients.

        Args:
            error_order: A positive, even `int` specifying the error order
                of this differentiator. This corresponds to the smallest power
                of `grid_spacing` remaining in the series that was truncated
                to generate this finite differencing expression.
            grid_spacing: A positive `float` specifying how large of a
                grid to use in calculating this finite difference.
        """
        if not (isinstance(error_order, numbers.Integral) and
                error_order > 0 and error_order % 2 == 0):
            raise ValueError("error_order must be a positive, even integer.")
        if not (isinstance(grid_spacing, numbers.Real) and grid_spacing > 0):
            raise ValueError("grid_spacing must be a positive real number.")
        grid_points_to_eval = np.concatenate([
            np.arange(-1 * error_order / 2, 0),
            np.arange(1, error_order / 2 + 1)
        ])
        weights = []
        n = error_order / 2
        for k in grid_points_to_eval:
            numerator = (-1)**(k + 1) * np.math.factorial(n)**2
            denom = k * np.math.factorial(n - k) * np.math.factorial(n + k)
            weights.append(numerator / (denom * grid_spacing))
        super().__init__(weights, grid_points_to_eval * grid_spacing)
