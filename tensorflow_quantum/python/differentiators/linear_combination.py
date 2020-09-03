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
    >>> symbol_values_array = np.array([[0.123]], dtype=np.float32)
    >>> # Calculate tfq gradient.
    >>> symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
    >>> with tf.GradientTape() as g:
    ...     g.watch(symbol_values_tensor)
    ...     expectations = op(circuit, ['alpha'], symbol_values_tensor, psums
    ... )
    >>> # Gradient would be: 5 * f(x+0) + 6 * f(x+0.5) + 7 * f(x+0.25)
    >>> grads = g.gradient(expectations, symbol_values_tensor)
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
        if not len(list(set(perturbations))) == len(perturbations):
            raise ValueError("All values in perturbations must be unique.")
        if len(perturbations) < 2:
            # This is so that tensor squeezing does not cause a problem later.
            raise ValueError("Must specify at least two perturbations.")
        self.weights = tf.constant(weights)
        self.n_perturbations = tf.constant(len(perturbations))
        self.perturbations = tf.constant(perturbations)

    @tf.function
    def _flat_mapper_gen_inner(
        program_ind, total_input_programs, op_ind, total_input_ops, symbol_ind, total_input_symbols,
        all_weights, n_non_zero_weights):

        # The last entry of `all_weights` is the weight of the forward pass.
        # Build the map components for a single symbol.
        w_zeros = tf.zeros([n_non_zero_weights + 1])
        stacked_weights = tf.stack([w_zeros, all_weights])
        gathered_weights = tf.gather(stacked_weights, tf.one_hot(op_ind, total_input_ops, dtype=tf.int32))
        transposed_weights = tf.transpose(gathered_weights, [1, 0])
        op_zeros = tf.zeros([1, total_input_ops])
        single_symbol_map = tf.concat([op_zeros, transposed_weights], 0)

        # Build the gather indices for placing weights correctly for a given program.
        range_tile = tf.tile(tf.expand_dims(tf.range(n_non_zero_weights) + 1, 0), [total_input_symbols, 1])
        range_zero = range_tile * tf.expand_dims(tf.one_hot(symbol_ind, total_input_symbols, dtype=tf.int32), 1)
        range_unroll = tf.reshape(range_zero, [n_non_zero_weights * total_input_symbols])
        single_program_map_indices = tf.concat([tf.expand_dims(total_input_symbols, 0), range_unroll], 0)

        # Now build the full map at these indices.
        single_program_map = tf.gather(single_symbol_map, single_program_map_indices)
        prog_zero = tf.zeros([total_input_symbols * n_non_zero_weights + 1, total_input_ops])
        big_stack = tf.stack([prog_zero, single_program_map])
        big_gather = tf.gather(big_stack, tf.one_hot(program_ind, total_input_programs, dtype=tf.int32))
        return tf.reshape(
            big_gather,
            [total_input_programs * (total_input_symbols * n_non_zero_weights + 1), total_input_ops])

    @tf.function
    def _flat_mapper_gen_symb(
        self, program_ind, total_input_programs, op_ind, total_input_ops,
        total_input_symbols, all_weights, n_non_zero_weights):
        return tf.map_fn(
            lambda x: self._flat_mapper_gen_inner(
                program_ind, total_input_programs, op_ind, total_input_ops, x,
                total_input_symbols, all_weights, n_non_zero_weights),
            tf.range(total_input_symbols),
            fn_output_signature=tf.float32)

    @tf.function
    def _flat_mapper_gen_op(
        self, program_ind, total_input_programs, total_input_ops,
        total_input_symbols, all_weights, n_non_zero_weights):
        return tf.map_fn(
            lambda x: self._flat_mapper_gen_symb(
                program_ind, total_input_programs, x, total_input_ops,
                total_input_symbols, all_weights, n_non_zero_weights),
            tf.range(total_input_ops),
            fn_output_signature=tf.float32)

    @tf.function
    def _flat_mapper_gen(
        self, total_input_programs, total_input_ops,
        total_input_symbols, all_weights, n_non_zero_weights):
        return tf.map_fn(
            lambda x: self._flat_mapper_gen_op(
                x, total_input_programs, total_input_ops,
                total_input_symbols, all_weights, n_non_zero_weights),
            tf.range(total_input_programs),
            fn_output_signature=tf.float32)

    @tf.function
    def get_intermediate_logic(self, programs, symbol_names, symbol_values,
                               pauli_sums):
        """See base class description."""

        n_programs = tf.gather(tf.shape(programs), 0)
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_pauli_sums = tf.gather(tf.shape(pauli_sums), 1)

        # don't do any computation for a perturbation of zero, just use
        # forward pass values
        mask = tf.not_equal(self.perturbations,
                            tf.zeros_like(self.perturbations))
        non_zero_perturbations = tf.boolean_mask(self.perturbations, mask)
        non_zero_weights = tf.boolean_mask(self.weights, mask)
        n_non_zero_perturbations = tf.gather(tf.shape(non_zero_perturbations),
                                             0)
        # Since perturbations are unique, zero_weight can only be len 0 or 1.
        mask = tf.equal(self.perturbations, tf.zeros_like(self.perturbations))
        zero_weight_raw = tf.boolean_mask(self.weights, mask)
        n_zero_perturbations = tf.gather(tf.shape(zero_weight_raw), 0)
        zero_weight = tf.pad(zero_weight_raw, [[0, 1 - n_zero_perturbations]])
        all_weights = tf.concat([non_zero_weights, zero_weight], 0)

        # For each program, create a new program for each symbol; for each
        # symbol, create a new circuit for each nonzero perturbation, plus one
        # for the forward pass value.
        expanded_programs = tf.expand_dims(programs, 1)
        tiled_programs = tf.tile(expanded_programs,
                                 [1, n_symbols * n_non_zero_perturbations + 1])
        flat_programs = tf.reshape(tiled_programs,
                                   [n_programs * (n_symbols * n_non_zero_perturbations + 1)])

        # Symbol names are not updated for LinearCombination.
        flat_symbol_names = symbol_names

        # Tile up the given parameter values to the correct shape.
        expanded_symbol_values = tf.expand_dims(symbol_values, 1)
        tiled_symbol_values = tf.tile(expanded_symbol_values,
                                      [1, n_symbols * n_non_zero_perturbations + 1, 1])
        flat_symbol_values_original = tf.reshape(
            tiled_symbol_values,
            [n_programs * (n_symbols * n_non_zero_perturbations + 1), n_symbols])

        # Generate the perturbations tensor.
        perturbation_zeros = tf.zeros([n_non_zero_perturbations])
        symbol_zeros = tf.zeros([1, n_symbols])
        stacked_perturbations = tf.stack([perturbation_zeros, non_zero_perturbations])
        gathered_perturbations = tf.gather(stacked_perturbations, tf.eye(n_symbols, dtype=tf.int32))
        transposed_perturbations = tf.transpose(gathered_perturbations, [0, 2, 1])
        reshaped_perturbations = tf.reshape(transposed_perturbations,
                                           [n_non_zero_perturbations * n_symbols, n_symbols])
        with_zero = tf.concat([symbol_zeros, reshaped_perturbations], 0)
        flat_perturbations = tf.tile(with_zero, [n_programs, 1])

        # Apply the perturbations to the parameter values.
        flat_symbol_values = flat_symbol_values_original + flat_perturbations

        # Output pauli sums are the same as the input, just tile them up.
        expanded_pauli_sums = tf.expand_dims(pauli_sums, 1)
        tiled_pauli_sums = tf.tile(expanded_pauli_sums,
                                   [1, n_symbols * n_non_zero_perturbations + 1, 1])
        flat_pauli_sums = tf.reshape(
            tiled_pauli_sums,
            [n_programs * (n_symbols * n_non_zero_perturbations + 1), n_pauli_sums])

        # The LinearCombination weights are entered into the mapper.
        flat_mapper = self._flat_mapper_gen(
            n_programs, n_pauli_sums, n_symbols, all_weights, n_non_zero_perturbations))

        return (flat_programs, flat_symbol_names, flat_symbol_values,
                flat_pauli_sums, flat_mapper)

    @tf.function
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):

        (
            flat_programs, flat_symbol_names, flat_symbol_values,
            flat_pauli_sums, flat_mapper
        ) = self.get_intermediate_logic(programs, symbol_names, symbol_values,
                                        pauli_sums)

        flat_expectations = self.expectation_op(flat_programs, flat_symbol_names,
                                                flat_symbol_values,
                                                flat_pauli_sums)

        # Apply the mapper to build the partial derivates
        partials = tf.reduce_sum(tf.reduce_sum(
            flat_mapper * flat_expectations, -1), -1)
        # Change order to [n_symbols, n_programs, n_ops]
        partials = tf.transpose(partials, [2, 0, 1])

        # now apply the chain rule
        return tf.einsum('sco,co -> cs', partials, grad)

    @tf.function
    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):

        (
            flat_programs, flat_symbol_names, flat_symbol_values,
            flat_pauli_sums, flat_mapper
        ) = self.get_intermediate_logic(programs, symbol_names, symbol_values,
                                        pauli_sums)

        flat_expectations = self.expectation_op(flat_programs, flat_symbol_names,
                                                flat_symbol_values,
                                                flat_pauli_sums, num_samples)

        # Apply the mapper to build the partial derivates
        partials = tf.reduce_sum(tf.reduce_sum(
            flat_mapper * flat_expectations, -1), -1)
        # Change order to [n_symbols, n_programs, n_ops]
        partials = tf.transpose(partials, [2, 0, 1])

        # now apply the chain rule
        return tf.einsum('sco,co -> cs', partials, grad)


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
