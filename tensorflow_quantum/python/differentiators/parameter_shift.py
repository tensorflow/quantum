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
    >>> symbol_values_array = np.array([[0.123]], dtype=np.float32)
    >>> # Calculate tfq gradient.
    >>> symbol_values_tensor = tf.convert_to_tensor(symbol_values_array)
    >>> with tf.GradientTape() as g:
    ...     g.watch(symbol_values_tensor)
    ...     expectations = op(circuit, ['alpha'], symbol_values_tensor, psums)
    >>> # This value is now computed via the ParameterShift rule.
    >>> # https://arxiv.org/abs/1905.13311
    >>> grads = g.gradient(expectations, symbol_values_tensor)
    >>> grads
    tf.Tensor([[-1.1839752]], shape=(1, 1), dtype=float32)

    """

    def _get_padded_weights(self, weights, symbol_ind, total_symbols):
        return tf.map_fn(
            lambda x: tf.pad(tf.gather(x, [symbol_ind]), [[
                symbol_ind, total_symbols - symbol_ind - 1
            ], [0, 0]]), weights)

    def _get_padded_expanded_weights(self, expanded_weights, op_ind, total_ops):
        return tf.pad(
            expanded_weights,
            [[0, 0], [0, 0], [0, 0], [op_ind, total_ops - op_ind - 1]])

    @tf.function
    def get_intermediate_logic(self, programs, symbol_names, symbol_values,
                               pauli_sums, num_samples):
        """See base class description for args and returns.

        The gradient calculations follows the following steps:

        1. Compute the decomposition of the incoming circuits so that we have
            their generator information (done using cirq in a tf.py_function)
        2. Use formula (31) from paper inside of TensorFlow to calculate
            gradients from all the decomposed circuits.
        3. Sum up terms and reshape for the total gradient that is compatible
            with TensorFlow.

        **CAUTION**
        Analytic gradient measurements based on this ParameterShift generally
        run at least K(=2) times SLOW than the original circuit.
        On top of it, since all parameters of gates are shifted individually,
        the time complexity is linear in the number of parameterized gates L.
        So, you will see O(KL) slower time & space complexity than the original
        forward pass measurements.
        """
        n_programs = tf.gather(tf.shape(programs), 0)
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_pauli_sums = tf.gather(tf.shape(pauli_sums), 1)

        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2

        # Deserialize programs and parse the whole parameterized gates
        # new_programs has [n_symbols, n_programs, n_param_gates, n_shifts]
        # These new_programs are programs with parameter-shift rule applied.
        (new_programs, weights, shifts,
         n_param_gates) = parameter_shift_util.parse_programs(
             programs, symbol_names, symbol_values, n_symbols)

        # Transpose and reshape new_programs to the new shape
        # [n_programs, n_symbols, n_param_gates, n_shifts].
        # Also, the weights and shifts should be the same for every program.
        new_programs = tf.transpose(new_programs, [1, 0, 2, 3])
        weights = tf.transpose(weights, [1, 0, 2, 3])
        shifts = tf.transpose(shifts, [1, 0, 2, 3])
        batch_programs = tf.reshape(
            new_programs, [n_programs, n_symbols * n_param_gates * n_shifts])

        # Append impurity symbol into symbol_names and tile.
        new_symbol_names = tf.concat([
            symbol_names,
            tf.expand_dims(tf.constant(
                parameter_shift_util._PARAMETER_IMPURITY_NAME),
                           axis=0)
        ],
                                     axis=0)
        batch_symbol_names = tf.tile(tf.expand_dims(new_symbol_names, 0),
                                     [n_programs, 1])

        # Construct the new parameter values.
        expanded_symbol_values = tf.expand_dims(symbol_values, 1)
        tiled_symbol_values = tf.tile(
            expanded_symbol_values,
            [1, n_symbols * n_param_gates * n_shifts, 1])
        tiled_shifts = tf.reshape(
            shifts, [n_programs, n_symbols * n_param_gates * n_shifts, 1])
        batch_symbol_values = tf.concat([tiled_symbol_values, tiled_shifts], -1)

        # Construct the new measurements.
        expanded_pauli_sums = tf.expand_dims(pauli_sums, 1)
        batch_pauli_sums = tf.tile(expanded_pauli_sums,
                                   [1, n_symbols * n_param_gates * n_shifts, 1])

        n_tile = tf.cast(tf.gather(tf.shape(batch_programs), 1), dtype=tf.int32)
        batch_num_samples = tf.tile(tf.expand_dims(num_samples, 1),
                                    tf.stack([1, n_tile, 1]))

        # Reshape the weights.
        shaped_weights = tf.reshape(
            weights, [n_programs, n_symbols, n_param_gates * n_shifts])
        reshaped_weights = tf.map_fn(
            lambda x: self._get_padded_weights(shaped_weights, x, n_symbols),
            tf.range(n_symbols),
            fn_output_signature=tf.float32)
        transposed_weights = tf.transpose(reshaped_weights, [1, 0, 2, 3])
        outer_reshaped_weights = tf.reshape(
            transposed_weights,
            [n_programs, n_symbols, n_symbols * n_param_gates * n_shifts])
        expanded_weights = tf.expand_dims(
            tf.expand_dims(outer_reshaped_weights, -1), 0)
        tiled_expanded_weights = tf.tile(expanded_weights,
                                         [n_pauli_sums, 1, 1, 1, 1])
        padded_weights = tf.map_fn(
            lambda x: self._get_padded_expanded_weights(x[0], x[1], n_pauli_sums
                                                       ),
            (tiled_expanded_weights, tf.range(n_pauli_sums)),
            fn_output_signature=tf.float32)
        batch_mapper = tf.transpose(padded_weights, [1, 0, 2, 3, 4])

        return (batch_programs, batch_symbol_names, batch_symbol_values,
                batch_pauli_sums, batch_num_samples, batch_mapper)

    @tf.function
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):
        """Calculate the gradient.
        The gradient calculations follows the following steps:
        1. Compute the decomposition of the incoming circuits so that we have
            their generator information (done using cirq in a tf.py_function)
        2. Use formula (31) from paper inside of TensorFlow to calculate
            gradients from all the decomposed circuits.
        3. Sum up terms and reshape for the total gradient that is compatible
            with TensorFlow.
        **CAUTION**
        Analytic gradient measurements based on this ParameterShift generally
        run at least K(=2) times SLOWER than the original circuit.
        On top of it, since all parameters of gates are shifted individually,
        the time complexity is linear in the number of parameterized gates L.
        So, you will see O(KL) slower time & space complexity than the original
        forward pass measurements.
        Args:
            programs: `tf.Tensor` of strings with shape [batch_size] containing
                the string representations of the circuits to be executed.
            symbol_names: `tf.Tensor` of strings with shape [n_params], which
                is used to specify the order in which the values in
                `symbol_values` should be placed inside of the circuits in
                `programs`.
            symbol_values: `tf.Tensor` of real numbers with shape
                [batch_size, n_params] specifying parameter values to resolve
                into the circuits specified by programs, following the ordering
                dictated by `symbol_names`.
            pauli_sums: `tf.Tensor` of strings with shape [batch_size, n_ops]
                containing the string representation of the operators that will
                be used on all of the circuits in the expectation calculations.
            forward_pass_vals: `tf.Tensor` of real numbers with shape
                [batch_size, n_ops] containing the output of the forward pass
                through the op you are differentiating.
            grad: `tf.Tensor` of real numbers with shape [batch_size, n_ops]
                representing the gradient backpropagated to the output of the
                op you are differentiating through.
        Returns:
            Backward gradient values for each program & each pauli sum. It has
            the shape of [batch_size, n_symbols].
        """

        # these get used a lot
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)
        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2

        # STEP 1: Generate required inputs for executor
        # Deserialize programs and parse the whole parameterized gates
        # new_programs has [n_symbols, n_param_gates, n_shifts, n_programs].
        # These new_programs has programs that parameter-shift rule is applied,
        # so those programs has
        (new_programs, weights, shifts,
         n_param_gates) = parameter_shift_util.parse_programs(
             programs, symbol_names, symbol_values, n_symbols)

        # Reshape & transpose new_programs, weights and shifts to fit into
        # the input format of tensorflow_quantum simulator.
        # [n_symbols, n_param_gates, n_shifts, n_programs]
        new_programs = tf.transpose(new_programs, [0, 2, 3, 1])
        weights = tf.transpose(weights, [0, 2, 3, 1])
        shifts = tf.transpose(shifts, [0, 2, 3, 1])

        # reshape everything to fit into expectation op correctly
        total_programs = n_programs * n_shifts * n_param_gates * n_symbols
        # tile up and then reshape to order programs correctly
        flat_programs = tf.reshape(new_programs, [total_programs])
        flat_shifts = tf.reshape(shifts, [total_programs])

        # tile up and then reshape to order ops correctly
        n_tile = n_shifts * n_param_gates * n_symbols
        flat_perturbations = tf.concat([
            tf.reshape(
                tf.tile(tf.expand_dims(symbol_values, 0),
                        tf.stack([n_tile, 1, 1])), [total_programs, n_symbols]),
            tf.expand_dims(flat_shifts, axis=1)
        ],
                                       axis=1)
        flat_ops = tf.reshape(
            tf.tile(tf.expand_dims(pauli_sums, 0), tf.stack([n_tile, 1, 1])),
            [total_programs, n_ops])
        # Append impurity symbol into symbol name
        new_symbol_names = tf.concat([
            symbol_names,
            tf.expand_dims(tf.constant(
                parameter_shift_util._PARAMETER_IMPURITY_NAME),
                           axis=0)
        ],
                                     axis=0)

        # STEP 2: calculate the required expectation values
        expectations = self.expectation_op(flat_programs, new_symbol_names,
                                           flat_perturbations, flat_ops)

        # STEP 3: generate gradients according to the results

        # we know the rows are grouped according to which parameter
        # was perturbed, so reshape to reflect that
        grouped_expectations = tf.reshape(
            expectations,
            [n_symbols, n_shifts * n_programs * n_param_gates, -1])

        # now we can calculate the partial of the circuit output with
        # respect to each perturbed parameter
        def rearrange_expectations(grouped):

            def split_vertically(i):
                return tf.slice(grouped, [i * n_programs, 0],
                                [n_programs, n_ops])

            return tf.map_fn(split_vertically,
                             tf.range(n_param_gates * n_shifts),
                             dtype=tf.float32)

        # reshape so that expectations calculated on different programs are
        # separated by a dimension
        rearranged_expectations = tf.map_fn(rearrange_expectations,
                                            grouped_expectations)

        # now we will calculate all of the partial derivatives
        partials = tf.einsum(
            'spco,spc->sco', rearranged_expectations,
            tf.cast(
                tf.reshape(weights,
                           [n_symbols, n_param_gates * n_shifts, n_programs]),
                rearranged_expectations.dtype))

        # now apply the chain rule
        return tf.einsum('sco,co -> cs', partials, grad)

    @tf.function
    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):
        """Calculate the gradient.
        The gradient calculations follows the following steps:
        1. Compute the decomposition of the incoming circuits so that we have
            their generator information (done using cirq in a tf.py_function)
        2. Use formula (31) from paper inside of TensorFlow to calculate
            gradients from all the decomposed circuits.
        3. Sum up terms and reshape for the total gradient that is compatible
            with TensorFlow.
        **CAUTION**
        Analytic gradient measurements based on this ParameterShift generally
        run at least K(=2) times SLOW than the original circuit.
        On top of it, since all parameters of gates are shifted individually,
        the time complexity is linear in the number of parameterized gates L.
        So, you will see O(KL) slower time & space complexity than the original
        forward pass measurements.
        Args:
            programs: `tf.Tensor` of strings with shape [batch_size] containing
                the string representations of the circuits to be executed.
            symbol_names: `tf.Tensor` of strings with shape [n_params], which
                is used to specify the order in which the values in
                `symbol_values` should be placed inside of the circuits in
                `programs`.
            symbol_values: `tf.Tensor` of real numbers with shape
                [batch_size, n_params] specifying parameter values to resolve
                into the circuits specified by programs, following the ordering
                dictated by `symbol_names`.
            pauli_sums: `tf.Tensor` of strings with shape [batch_size, n_ops]
                containing the string representation of the operators that will
                be used on all of the circuits in the expectation calculations.
            num_samples: `tf.Tensor` of positiver integers indicating the number
                of samples used per term to calculate the expectation value
                in the forward pass.
            forward_pass_vals: `tf.Tensor` of real numbers with shape
                [batch_size, n_ops] containing the output of the forward pass
                through the op you are differentiating.
            grad: `tf.Tensor` of real numbers with shape [batch_size, n_ops]
                representing the gradient backpropagated to the output of the
                op you are differentiating through.
        Returns:
            Backward gradient values for each program & each pauli sum. It has
            the shape of [batch_size, n_symbols].
        """

        # these get used a lot
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)
        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2

        # STEP 1: Generate required inputs for executor
        # Deserialize programs and parse the whole parameterized gates
        # new_programs has [n_symbols, n_param_gates, n_shifts, n_programs].
        # These new_programs has programs that parameter-shift rule is applied,
        # so those programs has
        (new_programs, weights, shifts,
         n_param_gates) = parameter_shift_util.parse_programs(
             programs, symbol_names, symbol_values, n_symbols)

        # Reshape & transpose new_programs, weights and shifts to fit into
        # the input format of tensorflow_quantum simulator.
        # [n_symbols, n_param_gates, n_shifts, n_programs]
        new_programs = tf.transpose(new_programs, [0, 2, 3, 1])
        weights = tf.transpose(weights, [0, 2, 3, 1])
        shifts = tf.transpose(shifts, [0, 2, 3, 1])

        # reshape everything to fit into expectation op correctly
        total_programs = n_programs * n_shifts * n_param_gates * n_symbols
        # tile up and then reshape to order programs correctly
        flat_programs = tf.reshape(new_programs, [total_programs])
        flat_shifts = tf.reshape(shifts, [total_programs])

        # tile up and then reshape to order ops correctly
        n_tile = n_shifts * n_param_gates * n_symbols
        flat_perturbations = tf.concat([
            tf.reshape(
                tf.tile(tf.expand_dims(symbol_values, 0),
                        tf.stack([n_tile, 1, 1])), [total_programs, n_symbols]),
            tf.expand_dims(flat_shifts, axis=1)
        ],
                                       axis=1)
        flat_ops = tf.reshape(
            tf.tile(tf.expand_dims(pauli_sums, 0), tf.stack([n_tile, 1, 1])),
            [total_programs, n_ops])
        flat_num_samples = tf.reshape(
            tf.tile(tf.expand_dims(num_samples, 0), tf.stack([n_tile, 1, 1])),
            [total_programs, n_ops])
        # Append impurity symbol into symbol name
        new_symbol_names = tf.concat([
            symbol_names,
            tf.expand_dims(tf.constant(
                parameter_shift_util._PARAMETER_IMPURITY_NAME),
                           axis=0)
        ],
                                     axis=0)

        # STEP 2: calculate the required expectation values
        expectations = self.expectation_op(flat_programs, new_symbol_names,
                                           flat_perturbations, flat_ops,
                                           flat_num_samples)

        # STEP 3: generate gradients according to the results

        # we know the rows are grouped according to which parameter
        # was perturbed, so reshape to reflect that
        grouped_expectations = tf.reshape(
            expectations,
            [n_symbols, n_shifts * n_programs * n_param_gates, -1])

        # now we can calculate the partial of the circuit output with
        # respect to each perturbed parameter
        def rearrange_expectations(grouped):

            def split_vertically(i):
                return tf.slice(grouped, [i * n_programs, 0],
                                [n_programs, n_ops])

            return tf.map_fn(split_vertically,
                             tf.range(n_param_gates * n_shifts),
                             dtype=tf.float32)

        # reshape so that expectations calculated on different programs are
        # separated by a dimension
        rearranged_expectations = tf.map_fn(rearrange_expectations,
                                            grouped_expectations)

        # now we will calculate all of the partial derivatives
        partials = tf.einsum(
            'spco,spc->sco', rearranged_expectations,
            tf.cast(
                tf.reshape(weights,
                           [n_symbols, n_param_gates * n_shifts, n_programs]),
                rearranged_expectations.dtype))

        # now apply the chain rule
        return tf.einsum('sco,co -> cs', partials, grad)
