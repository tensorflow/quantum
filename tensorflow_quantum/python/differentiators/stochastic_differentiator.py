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
"""Compute gradients by using stochastic generator.
For the test of this SGDifferentiator's consistency & convergence, please see:
//benchmarks/scripts/differentiators:convergence_test
"""
import tensorflow as tf

from tensorflow_quantum.python.differentiators import differentiator, \
    parameter_shift_util, stochastic_differentiator_util as sd_util


class SGDifferentiator(differentiator.Differentiator):
    """Stochastic generator based differentiator class.
    SGDifferentiator allows you to get the sampled gradient value from three
    different stochastic processes:
    - parameter coordinate sampling
        Choose one of the symbols of the given programs and perform coordinate
        descent optimization.
        e.g. if a program has parameters ['a','b','c'], choose 'a' w.r.t given
            probability and get the partial derivative of the direction 'a' only
    - parameter-shift rule generators sampling
        e.g. Given symbols, there could be many operators sharing the same
            symbol, X**'a', Y**'a', Z**'a'. Choose Y**'a' w.r.t given
            probability and get the partial derivative of the generator.
    - cost Hamiltonian sampling
        e.g. if there are cost Hamiltonians such as ['Z1',Z2',Z3'], then choose
            'Z2' w.r.t given probability and get the partial derivative of the
            Hamiltonian observable only.
    and the expectation value of the sampled gradient value converges into
    the true ground truth gradient value.
    This Stochastic Generator Differentiator is the modified gradient estimator
    of the following two papers:
    - [arXiv:1901.05374](https://arxiv.org/abs/1901.05374), Harrow et al.
    - [arXiv:1910.01155](https://arxiv.org/abs/1910.01155), Sweke et al.

    >>> # Get an expectation op.
    >>> my_op = tfq.get_expectation_op()
    >>> # Attach a differentiator.
    >>> my_dif = tfq.differentiators.SGDifferentiator()
    >>> op = my_dif.generate_differentiable_op(
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
    >>> # This value is now computed via the stochastic processes described in:
    >>> # https://arxiv.org/abs/1901.05374
    >>> # https://arxiv.org/abs/1910.01155
    >>> grads = g.gradient(expectations, symbol_values_tensor)
    >>> # the result is non-deterministic in general, but in this special case,
    >>> # it has only one result.
    >>> grads
    <tf.Tensor: shape=(1, 1), dtype=float32, numpy=[[-1.1839752]]>

    """

    def __init__(self,
                 stochastic_coordinate=True,
                 stochastic_generator=True,
                 stochastic_cost=True,
                 uniform_sampling=False):
        """Instantiate this differentiator.
        Create a SGDifferentiator.
        Args:
            stochastic_coordinate: Python `bool` to determine if
                sampling on coordinate is performed or not. Default to True.
            stochastic_generator: Python `bool` to determine if
                sampling on generator is performed or not. Default to True.
            stochastic_cost: Python `bool` to determine if sampling on
                cost Hamiltonian is performed or not. Default to True.
            uniform_sampling: Python `bool` to determine the
                probabilistic distributions on the sampling targets.
                Default to False.
        """

        def _boolean_type_check(variable, variable_name):
            if variable != True and variable != False:
                raise TypeError("{} must be boolean: Got {} {}".format(
                    variable_name, variable, type(variable)))

        _boolean_type_check(stochastic_coordinate, "stochastic_coordinate")
        _boolean_type_check(stochastic_generator, "stochastic_generator")
        _boolean_type_check(stochastic_cost, "stochastic_cost")
        _boolean_type_check(uniform_sampling, "uniform_sampling")

        self.stochastic_coordinate = stochastic_coordinate
        self.stochastic_generator = stochastic_generator
        self.stochastic_cost = stochastic_cost
        self.uniform_sampling = uniform_sampling

    def get_intermediate_logic(self, programs, symbol_names, symbol_values, pauli_sums):
        n_symbols = tf.gather(tf.shape(symbol_values), 1)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_shifts = 2

        # STEP 1: Generate required inputs for executor by using parsers

        # Deserialize programs and parse the whole parameterized gates
        # new_programs has [n_symbols, n_programs, n_param_gates, n_shifts].
        new_programs, weights, shifts, n_param_gates = \
            parameter_shift_util.parse_programs(
                programs, symbol_names, symbol_values, n_symbols)

        if self.stochastic_generator:
            # Result : [n_symbols, n_programs, n_param_gates=1, n_shifts].
            new_programs, weights, shifts, n_param_gates = \
                sd_util.stochastic_generator_preprocessor(
                    new_programs, weights, shifts, n_programs, n_symbols,
                    n_param_gates, n_shifts, self.uniform_sampling)

        # Reshape & transpose new_programs, weights and shifts to fit into
        # the input format of tensorflow_quantum simulator.
        # [n_symbols, n_param_gates, n_shifts, n_programs]
        new_programs = tf.transpose(new_programs, [0, 2, 3, 1])
        weights = tf.transpose(weights, [0, 2, 3, 1])
        shifts = tf.transpose(shifts, [0, 2, 3, 1])

        cost_relocator = None
        if self.stochastic_cost:
            # Result : pauli_sums [n_programs, n_ops] -> [n_programs, n_ops=1]
            pauli_sums, cost_relocator, n_ops = \
                sd_util.stochastic_cost_preprocessor(
                    pauli_sums, n_programs, n_ops, self.uniform_sampling)

        coordinate_relocator = None
        if self.stochastic_coordinate:
            flat_programs, flat_perturbations, flat_ops, _, weights, \
            coordinate_relocator = sd_util.stochastic_coordinate_preprocessor(
                new_programs, symbol_values, pauli_sums, weights, shifts,
                n_programs, n_symbols, n_param_gates, n_shifts, n_ops,
                self.uniform_sampling)
        else:
            # reshape everything to fit into expectation op correctly
            total_programs = n_programs * n_shifts * n_symbols * n_param_gates
            # tile up and then reshape to order programs correctly
            flat_programs = tf.reshape(new_programs, [total_programs])
            flat_shifts = tf.reshape(shifts, [total_programs])

            # tile up and then reshape to order ops correctly
            n_tile = n_shifts * n_symbols * n_param_gates
            flat_perturbations = tf.concat([
                tf.reshape(
                    tf.tile(tf.expand_dims(symbol_values, 0),
                            tf.stack([n_tile, 1, 1])),
                    [total_programs, n_symbols]),
                tf.expand_dims(flat_shifts, axis=1)
            ],
                                           axis=1)
            flat_ops = tf.reshape(
                tf.tile(tf.expand_dims(pauli_sums, 0),
                        tf.stack([n_tile, 1, 1])), [total_programs, n_ops])

        # Append impurity symbol into symbol name
        new_symbol_names = tf.concat([
            symbol_names,
            tf.expand_dims(tf.constant(
                parameter_shift_util._PARAMETER_IMPURITY_NAME),
                           axis=0)
        ],
                                     axis=0)
        return flat_programs, new_symbol_names, weights, flat_perturbations, flat_ops, cost_relocator, coordinate_relocator, n_param_gates


    @tf.function
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):
        """Compute the sampled gradient with cascaded stochastic processes.
        The gradient calculations follows the following steps:
        1. Compute the decomposition of the incoming circuits so that we have
            their generator information (done using cirq in a tf.py_function)
        2. Construct probability distributions & perform stochastic processes
            to select parameter-shift terms.
            - Stochastic generator : sampling on parameter-shifted gates.
            - Stochastic coordinate : sampling on symbols.
            - Stochastic cost : sampling on pauli sums
        3. Sum up terms and reshape for the total gradient that is compatible
            with tensorflow differentiation.
        Args:
            programs: `tf.Tensor` of strings with shape [n_programs] containing
                the string representations of the circuits to be executed.
            symbol_names: `tf.Tensor` of strings with shape [n_symbols], which
                is used to specify the order in which the values in
                `symbol_values` should be placed inside of the circuits in
                `programs`.
            symbol_values: `tf.Tensor` of real numbers with shape
                [n_programs, n_symbols] specifying parameter values to resolve
                into the circuits specified by programs, following the ordering
                dictated by `symbol_names`.
            pauli_sums : `tf.Tensor` of strings with shape [n_programs, n_ops]
                representing output observables for each program.
            forward_pass_vals : `tf.Tensor` of real numbers for forward pass
                values with the shape of [n_programs, n_ops]
            grad : `tf.Tensor` of real numbers for backpropagated gradient
                values from the upper layer with the shape of
                [n_programs, n_ops]
        Returns:
            A `tf.Tensor` of real numbers for sampled gradients from the above
            samplers with the shape of [n_programs, n_symbols]
        """
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)
        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2
        flat_programs, new_symbol_names, weights, flat_perturbations, flat_ops, cost_relocator, coordinate_relocator, n_param_gates = self.get_intermediate_logic(
            programs, symbol_names, symbol_values, pauli_sums)
        total_programs = n_param_gates * n_programs * n_shifts * n_symbols
        n_tile = n_shifts * n_param_gates * n_symbols

        # STEP 2: calculate the required expectation values
        expectations = self.expectation_op(flat_programs, new_symbol_names,
                                           flat_perturbations, flat_ops)

        # STEP 3: generate gradients according to the results
        if self.stochastic_coordinate:
            # Transpose to the original shape
            # [n_symbols, n_programs, n_param_gates, n_shifts]
            #
            # coordinate_relocator has [sub_total_programs, n_symbols](=ij)
            # expectations has [sub_total_programs, n_ops](=ik)
            # einsum -> [n_ops, n_symbols, sub_total_programs](=kji)
            expectations = tf.einsum(
                'ij,ik->kji', tf.cast(coordinate_relocator, dtype=tf.float64),
                tf.cast(expectations, dtype=tf.float64))
            # Transpose to [n_symbols, sub_total_programs, n_ops]
            expectations = tf.transpose(expectations, [1, 2, 0])

        # we know the rows are grouped according to which parameter
        # was perturbed, so reshape to reflect that
        grouped_expectations = tf.reshape(
            tf.cast(expectations, dtype=tf.float64),
            [n_symbols, n_shifts * n_programs * n_param_gates, -1])

        # now we can calculate the partial of the circuit output with
        # respect to each perturbed parameter
        def rearrange_expectations(grouped):

            def split_vertically(i):
                return tf.slice(grouped, [i * n_programs, 0],
                                [n_programs, n_ops])

            return tf.map_fn(split_vertically,
                             tf.range(n_param_gates * n_shifts),
                             dtype=tf.float64)

        # reshape so that expectations calculated on different programs are
        # separated by a dimension
        rearranged_expectations = tf.map_fn(rearrange_expectations,
                                            grouped_expectations,
                                            dtype=tf.float64)

        # now we will calculate all of the partial derivatives
        # s: symbol, p: perturbation, c: circuit, o: ops
        partials = tf.einsum(
            'spco,spc->sco', rearranged_expectations,
            tf.cast(tf.reshape(
                weights, [n_symbols, n_param_gates * n_shifts, n_programs]),
                    dtype=tf.float64))

        if self.stochastic_cost:
            # Reshape to the original n_ops shape
            # partials: [n_symbols, n_programs, n_ops=1]
            # cost_relocator: [n_programs, original_n_ops]
            # Result: [n_symbols, n_programs, original_n_ops]
            partials = partials * tf.stop_gradient(
                tf.cast(cost_relocator, dtype=tf.float64))

        # now apply the chain rule
        # cast partials back to float32
        return tf.cast(
            tf.einsum('sco,co -> cs', partials,
                      tf.cast(grad, dtype=tf.float64)), tf.float32)

    @tf.function
    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):
        """Compute the sampled gradient with cascaded stochastic processes.
        The gradient calculations follows the following steps:
        1. Compute the decomposition of the incoming circuits so that we have
            their generator information (done using cirq in a tf.py_function)
        2. Construct probability distributions & perform stochastic processes
            to select parameter-shift terms.
            - Stochastic generator : sampling on parameter-shifted gates.
            - Stochastic coordinate : sampling on symbols.
            - Stochastic cost : sampling on pauli sums
        3. Sum up terms and reshape for the total gradient that is compatible
            with tensorflow differentiation.
        Args:
            programs: `tf.Tensor` of strings with shape [n_programs] containing
                the string representations of the circuits to be executed.
            symbol_names: `tf.Tensor` of strings with shape [n_symbols], which
                is used to specify the order in which the values in
                `symbol_values` should be placed inside of the circuits in
                `programs`.
            symbol_values: `tf.Tensor` of real numbers with shape
                [n_programs, n_symbols] specifying parameter values to resolve
                into the circuits specified by programs, following the ordering
                dictated by `symbol_names`.
            num_samples: `tf.Tensor` of positive integers representing the
                number of samples per term in each term of pauli_sums used
                during the forward pass.
            pauli_sums : `tf.Tensor` of strings with shape [n_programs, n_ops]
                representing output observables for each program.
            forward_pass_vals : `tf.Tensor` of real numbers for forward pass
                values with the shape of [n_programs, n_ops]
            grad : `tf.Tensor` of real numbers for backpropagated gradient
                values from the upper layer with the shape of
                [n_programs, n_ops]
        Returns:
            A `tf.Tensor` of real numbers for sampled gradients from the above
            samplers with the shape of [n_programs, n_symbols]
        """
        n_symbols = tf.gather(tf.shape(symbol_names), 0)
        n_programs = tf.gather(tf.shape(programs), 0)
        n_ops = tf.gather(tf.shape(pauli_sums), 1)
        # Assume cirq.decompose() generates gates with at most two distinct
        # eigenvalues, which results in two parameter shifts.
        n_shifts = 2
        flat_programs, new_symbol_names, weights, flat_perturbations, flat_ops, cost_relocator, coordinate_relocator, n_param_gates = self.get_intermediate_logic(
            programs, symbol_names, symbol_values, pauli_sums)
        total_programs = n_param_gates * n_programs * n_shifts * n_symbols
        n_tile = n_shifts * n_param_gates * n_symbols
        flat_num_samples = tf.reshape(
            tf.tile(tf.expand_dims(num_samples, 0),
                    tf.stack([n_tile, 1, 1])), [total_programs, n_ops])

        # STEP 2: calculate the required expectation values
        expectations = self.expectation_op(flat_programs, new_symbol_names,
                                           flat_perturbations, flat_ops,
                                           flat_num_samples)

        # STEP 3: generate gradients according to the results
        if self.stochastic_coordinate:
            # Transpose to the original shape
            # [n_symbols, n_programs, n_param_gates, n_shifts]
            #
            # coordinate_relocator has [sub_total_programs, n_symbols](=ij)
            # expectations has [sub_total_programs, n_ops](=ik)
            # einsum -> [n_ops, n_symbols, sub_total_programs](=kji)
            expectations = tf.einsum(
                'ij,ik->kji', tf.cast(coordinate_relocator, dtype=tf.float64),
                tf.cast(expectations, dtype=tf.float64))
            # Transpose to [n_symbols, sub_total_programs, n_ops]
            expectations = tf.transpose(expectations, [1, 2, 0])

        # we know the rows are grouped according to which parameter
        # was perturbed, so reshape to reflect that
        grouped_expectations = tf.reshape(
            tf.cast(expectations, dtype=tf.float64),
            [n_symbols, n_shifts * n_programs * n_param_gates, -1])

        # now we can calculate the partial of the circuit output with
        # respect to each perturbed parameter
        def rearrange_expectations(grouped):

            def split_vertically(i):
                return tf.slice(grouped, [i * n_programs, 0],
                                [n_programs, n_ops])

            return tf.map_fn(split_vertically,
                             tf.range(n_param_gates * n_shifts),
                             dtype=tf.float64)

        # reshape so that expectations calculated on different programs are
        # separated by a dimension
        rearranged_expectations = tf.map_fn(rearrange_expectations,
                                            grouped_expectations,
                                            dtype=tf.float64)

        # now we will calculate all of the partial derivatives
        # s: symbol, p: perturbation, c: circuit, o: ops
        partials = tf.einsum(
            'spco,spc->sco', rearranged_expectations,
            tf.cast(tf.reshape(
                weights, [n_symbols, n_param_gates * n_shifts, n_programs]),
                    dtype=tf.float64))

        if self.stochastic_cost:
            # Reshape to the original n_ops shape
            # partials: [n_symbols, n_programs, n_ops=1]
            # cost_relocator: [n_programs, original_n_ops]
            # Result: [n_symbols, n_programs, original_n_ops]
            partials = partials * tf.stop_gradient(
                tf.cast(cost_relocator, dtype=tf.float64))

        # now apply the chain rule
        # cast partials back to float32
        return tf.cast(
            tf.einsum('sco,co -> cs', partials,
                      tf.cast(grad, dtype=tf.float64)), tf.float32)
