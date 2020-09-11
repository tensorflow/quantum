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
        self.weights = tf.constant(weights, dtype=tf.float32)
        self.n_perturbations = tf.constant(len(perturbations))
        self.perturbations = tf.constant(perturbations, dtype=tf.float32)

    @tf.function
    def _measurement_mapper_gen_inner(self, op_ind, total_input_ops, symbol_ind,
                                      total_input_symbols, all_weights,
                                      n_non_zero_weights):
        """The last entry of `all_weights` is the weight of the forward pass."""
        # Build the map components for a single symbol.
        w_zeros = tf.zeros([n_non_zero_weights + 1])
        stacked_weights = tf.stack([w_zeros, all_weights])
        gathered_weights = tf.gather(
            stacked_weights, tf.one_hot(op_ind, total_input_ops,
                                        dtype=tf.int32))
        transposed_weights = tf.transpose(gathered_weights, [1, 0])
        op_zeros = tf.zeros([1, total_input_ops])
        single_symbol_map = tf.concat([op_zeros, transposed_weights], 0)

        # Build the gather indices to place weights for a given program.
        range_tile = tf.tile(
            tf.expand_dims(tf.range(n_non_zero_weights) + 1, 0),
            [total_input_symbols, 1])
        range_zero = range_tile * tf.expand_dims(
            tf.one_hot(symbol_ind, total_input_symbols, dtype=tf.int32), 1)
        range_unroll = tf.reshape(range_zero,
                                  [n_non_zero_weights * total_input_symbols])
        single_program_map_indices = tf.concat(
            [tf.expand_dims(n_non_zero_weights + 1, 0), range_unroll], 0)

        # Now build the full map at these indices.
        single_program_map = tf.gather(single_symbol_map,
                                       single_program_map_indices)
        return single_program_map

    @tf.function
    def _measurement_mapper_gen_symb(self, op_ind, total_input_ops,
                                     total_input_symbols, all_weights,
                                     n_non_zero_weights):
        return tf.map_fn(lambda x: self._measurement_mapper_gen_inner(
            op_ind, total_input_ops, x, total_input_symbols, all_weights,
            n_non_zero_weights),
                         tf.range(total_input_symbols),
                         fn_output_signature=tf.float32)

    @tf.function
    def _measurement_mapper_gen(self, total_input_ops, total_input_symbols,
                                all_weights, n_non_zero_weights):
        return tf.map_fn(lambda x: self._measurement_mapper_gen_symb(
            x, total_input_ops, total_input_symbols, all_weights,
            n_non_zero_weights),
                         tf.range(total_input_ops),
                         fn_output_signature=tf.float32)

    @tf.function
    def get_intermediate_logic(self, programs, symbol_names, symbol_values,
                               pauli_sums, num_samples):
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
        batch_programs = tf.tile(expanded_programs,
                                 [1, n_symbols * n_non_zero_perturbations + 1])

        # Symbol names are not updated for LinearCombination.
        batch_symbol_names = tf.tile(tf.expand_dims(symbol_names, 0),
                                     [n_programs, 1])

        # Tile up the given parameter values to the correct shape.
        expanded_symbol_values = tf.expand_dims(symbol_values, 1)
        batch_symbol_values_original = tf.tile(
            expanded_symbol_values,
            [1, n_symbols * n_non_zero_perturbations + 1, 1])

        # Generate the perturbations tensor.
        perturbation_zeros = tf.zeros([n_non_zero_perturbations],
                                      dtype=tf.float32)
        symbol_zeros = tf.zeros([1, n_symbols])
        stacked_perturbations = tf.stack(
            [perturbation_zeros, non_zero_perturbations])
        gathered_perturbations = tf.gather(stacked_perturbations,
                                           tf.eye(n_symbols, dtype=tf.int32))
        transposed_perturbations = tf.transpose(gathered_perturbations,
                                                [0, 2, 1])
        reshaped_perturbations = tf.reshape(
            transposed_perturbations,
            [n_non_zero_perturbations * n_symbols, n_symbols])
        with_zero = tf.concat([symbol_zeros, reshaped_perturbations], 0)
        batch_perturbations = tf.tile(tf.expand_dims(with_zero, 0),
                                      [n_programs, 1, 1])

        # Apply the perturbations to the parameter values.
        batch_symbol_values = batch_symbol_values_original + batch_perturbations

        # Output pauli sums are the same as the input, just tile them up.
        expanded_pauli_sums = tf.expand_dims(pauli_sums, 1)
        batch_pauli_sums = tf.tile(
            expanded_pauli_sums,
            [1, n_symbols * n_non_zero_perturbations + 1, 1])

        # pauli_sums and num_samples should be tiled identically.
        expanded_num_samples = tf.expand_dims(num_samples, 1)
        batch_num_samples = tf.tile(
            expanded_num_samples,
            [1, n_symbols * n_non_zero_perturbations + 1, 1])

        # The LinearCombination weights are entered into the mapper.
        single_program_mapper = self._measurement_mapper_gen(
            n_pauli_sums, n_symbols, all_weights, n_non_zero_perturbations)
        batch_mapper = tf.tile(tf.expand_dims(single_program_mapper, 0),
                               [n_programs, 1, 1, 1, 1])

        return (batch_programs, batch_symbol_names, batch_symbol_values,
                batch_pauli_sums, batch_num_samples, batch_mapper)

    @tf.function
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):

        # these get used a lot
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)

        # STEP 1: Generate required inputs for executor
        # in this case I can do this with existing tensorflow ops if i'm clever

        # don't do any computation for a perturbation of zero, just use
        # forward pass values
        mask = tf.not_equal(self.perturbations,
                            tf.zeros_like(self.perturbations))
        non_zero_perturbations = tf.boolean_mask(self.perturbations, mask)
        non_zero_weights = tf.boolean_mask(self.weights, mask)
        n_non_zero_perturbations = tf.gather(tf.shape(non_zero_perturbations),
                                             0)

        # tile up symbols to [n_non_zero_perturbations, n_programs, n_symbols]
        perturbation_tiled_symbols = tf.tile(
            tf.expand_dims(symbol_values, 0),
            tf.stack([n_non_zero_perturbations, 1, 1]))

        def create_3d_perturbation(i, perturbation_values):
            """Generate a tensor the same shape as perturbation_tiled_symbols
             containing the perturbations specified by perturbation_values."""
            ones = tf.cast(
                tf.concat([
                    tf.zeros(tf.stack([n_non_zero_perturbations, n_programs, i
                                      ])),
                    tf.ones(tf.stack([n_non_zero_perturbations, n_programs, 1
                                     ])),
                    tf.zeros(
                        tf.stack([
                            n_non_zero_perturbations, n_programs,
                            tf.subtract(n_symbols, tf.add(i, 1))
                        ]))
                ],
                          axis=2), perturbation_values.dtype)
            return tf.einsum('kij,k->kij', ones, perturbation_values)

        def generate_perturbation(i):
            """Perturb each value in the ith column of
             perturbation_tiled_symbols.
            """
            return tf.add(
                perturbation_tiled_symbols,
                tf.cast(create_3d_perturbation(i, non_zero_perturbations),
                        perturbation_tiled_symbols.dtype))

        # create a 4d tensor with the following dimensions:
        # [n_symbols, n_perturbations, n_programs, n_symbols]
        # the zeroth dimension represents the fact that we have to apply
        # a perturbation in the direction of every parameter individually.
        # the first dimension represents the number of perturbations that we
        # have to apply, and the inner 2 dimensions represent the standard
        # input format to the expectation ops
        all_perturbations = tf.map_fn(generate_perturbation,
                                      tf.range(n_symbols),
                                      dtype=tf.float32)

        # reshape everything to fit into expectation op correctly
        total_programs = tf.multiply(
            tf.multiply(n_programs, n_non_zero_perturbations), n_symbols)
        # tile up and then reshape to order programs correctly
        flat_programs = tf.reshape(
            tf.tile(
                tf.expand_dims(programs, 0),
                tf.stack([tf.multiply(n_symbols, n_non_zero_perturbations),
                          1])), [total_programs])
        flat_perturbations = tf.reshape(all_perturbations, [
            tf.multiply(tf.multiply(n_symbols, n_non_zero_perturbations),
                        n_programs), n_symbols
        ])
        # tile up and then reshape to order ops correctly
        flat_ops = tf.reshape(
            tf.tile(
                tf.expand_dims(pauli_sums, 0),
                tf.stack(
                    [tf.multiply(n_symbols, n_non_zero_perturbations), 1, 1])),
            [total_programs, n_ops])

        # STEP 2: calculate the required expectation values
        expectations = self.expectation_op(flat_programs, symbol_names,
                                           flat_perturbations, flat_ops)

        # STEP 3: generate gradients according to the results

        # we know the rows are grouped according to which parameter
        # was perturbed, so reshape to reflect that
        grouped_expectations = tf.reshape(
            expectations,
            [n_symbols,
             tf.multiply(n_non_zero_perturbations, n_programs), -1])

        # now we can calculate the partial of the circuit output with
        # respect to each perturbed parameter
        def rearrange_expectations(grouped):

            def split_vertically(i):
                return tf.slice(grouped, [tf.multiply(i, n_programs), 0],
                                [n_programs, n_ops])

            return tf.map_fn(split_vertically,
                             tf.range(n_non_zero_perturbations),
                             dtype=tf.float32)

        # reshape so that expectations calculated on different programs are
        # separated by a dimension
        rearranged_expectations = tf.map_fn(rearrange_expectations,
                                            grouped_expectations)

        # now we will calculate all of the partial derivatives

        nonzero_partials = tf.einsum(
            'spco,p->sco', rearranged_expectations,
            tf.cast(non_zero_weights, rearranged_expectations.dtype))

        # now add the contribution of a zero term if required

        # find any zero terms
        mask = tf.equal(self.perturbations, tf.zeros_like(self.perturbations))
        zero_weight = tf.boolean_mask(self.weights, mask)
        n_zero_perturbations = tf.gather(tf.shape(zero_weight), 0)

        # this will have shape [n_symbols, n_programs, n_ops]
        partials = tf.cond(
            tf.equal(n_zero_perturbations, 0), lambda: nonzero_partials,
            lambda: nonzero_partials + tf.multiply(
                tf.tile(tf.expand_dims(forward_pass_vals, axis=0),
                        tf.stack([n_symbols, 1, 1])),
                tf.cast(tf.gather(zero_weight, 0), forward_pass_vals.dtype)))

        # now apply the chain rule
        return tf.einsum('sco,co -> cs', partials, grad)

    @tf.function
    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):

        # these get used a lot
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)

        # STEP 1: Generate required inputs for executor
        # in this case I can do this with existing tensorflow ops if i'm clever

        # don't do any computation for a perturbation of zero, just use
        # forward pass values
        mask = tf.not_equal(self.perturbations,
                            tf.zeros_like(self.perturbations))
        non_zero_perturbations = tf.boolean_mask(self.perturbations, mask)
        non_zero_weights = tf.boolean_mask(self.weights, mask)
        n_non_zero_perturbations = tf.gather(tf.shape(non_zero_perturbations),
                                             0)

        # tile up symbols to [n_non_zero_perturbations, n_programs, n_symbols]
        perturbation_tiled_symbols = tf.tile(
            tf.expand_dims(symbol_values, 0),
            tf.stack([n_non_zero_perturbations, 1, 1]))

        def create_3d_perturbation(i, perturbation_values):
            """Generate a tensor the same shape as perturbation_tiled_symbols
             containing the perturbations specified by perturbation_values."""
            ones = tf.cast(
                tf.concat([
                    tf.zeros(tf.stack([n_non_zero_perturbations, n_programs, i
                                      ])),
                    tf.ones(tf.stack([n_non_zero_perturbations, n_programs, 1
                                     ])),
                    tf.zeros(
                        tf.stack([
                            n_non_zero_perturbations, n_programs,
                            tf.subtract(n_symbols, tf.add(i, 1))
                        ]))
                ],
                          axis=2), perturbation_values.dtype)
            return tf.einsum('kij,k->kij', ones, perturbation_values)

        def generate_perturbation(i):
            """Perturb each value in the ith column of
             perturbation_tiled_symbols.
            """
            return tf.add(
                perturbation_tiled_symbols,
                tf.cast(create_3d_perturbation(i, non_zero_perturbations),
                        perturbation_tiled_symbols.dtype))

        # create a 4d tensor with the following dimensions:
        # [n_symbols, n_perturbations, n_programs, n_symbols]
        # the zeroth dimension represents the fact that we have to apply
        # a perturbation in the direction of every parameter individually.
        # the first dimension represents the number of perturbations that we
        # have to apply, and the inner 2 dimensions represent the standard
        # input format to the expectation ops
        all_perturbations = tf.map_fn(generate_perturbation,
                                      tf.range(n_symbols),
                                      dtype=tf.float32)

        # reshape everything to fit into expectation op correctly
        total_programs = tf.multiply(
            tf.multiply(n_programs, n_non_zero_perturbations), n_symbols)
        # tile up and then reshape to order programs correctly
        flat_programs = tf.reshape(
            tf.tile(
                tf.expand_dims(programs, 0),
                tf.stack([tf.multiply(n_symbols, n_non_zero_perturbations),
                          1])), [total_programs])
        flat_perturbations = tf.reshape(all_perturbations, [
            tf.multiply(tf.multiply(n_symbols, n_non_zero_perturbations),
                        n_programs), n_symbols
        ])
        # tile up and then reshape to order ops correctly
        flat_ops = tf.reshape(
            tf.tile(
                tf.expand_dims(pauli_sums, 0),
                tf.stack(
                    [tf.multiply(n_symbols, n_non_zero_perturbations), 1, 1])),
            [total_programs, n_ops])
        flat_num_samples = tf.reshape(
            tf.tile(
                tf.expand_dims(num_samples, 0),
                tf.stack(
                    [tf.multiply(n_symbols, n_non_zero_perturbations), 1, 1])),
            [total_programs, n_ops])

        # STEP 2: calculate the required expectation values
        expectations = self.expectation_op(flat_programs, symbol_names,
                                           flat_perturbations, flat_ops,
                                           flat_num_samples)

        # STEP 3: generate gradients according to the results

        # we know the rows are grouped according to which parameter
        # was perturbed, so reshape to reflect that
        grouped_expectations = tf.reshape(
            expectations,
            [n_symbols,
             tf.multiply(n_non_zero_perturbations, n_programs), -1])

        # now we can calculate the partial of the circuit output with
        # respect to each perturbed parameter
        def rearrange_expectations(grouped):

            def split_vertically(i):
                return tf.slice(grouped, [tf.multiply(i, n_programs), 0],
                                [n_programs, n_ops])

            return tf.map_fn(split_vertically,
                             tf.range(n_non_zero_perturbations),
                             dtype=tf.float32)

        # reshape so that expectations calculated on different programs are
        # separated by a dimension
        rearranged_expectations = tf.map_fn(rearrange_expectations,
                                            grouped_expectations)

        # now we will calculate all of the partial derivatives

        nonzero_partials = tf.einsum(
            'spco,p->sco', rearranged_expectations,
            tf.cast(non_zero_weights, rearranged_expectations.dtype))

        # now add the contribution of a zero term if required

        # find any zero terms
        mask = tf.equal(self.perturbations, tf.zeros_like(self.perturbations))
        zero_weight = tf.boolean_mask(self.weights, mask)
        n_zero_perturbations = tf.gather(tf.shape(zero_weight), 0)

        # this will have shape [n_symbols, n_programs, n_ops]
        partials = tf.cond(
            tf.equal(n_zero_perturbations, 0), lambda: nonzero_partials,
            lambda: nonzero_partials + tf.multiply(
                tf.tile(tf.expand_dims(forward_pass_vals, axis=0),
                        tf.stack([n_symbols, 1, 1])),
                tf.cast(tf.gather(zero_weight, 0), forward_pass_vals.dtype)))

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
