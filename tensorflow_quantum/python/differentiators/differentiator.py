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
"""Testing consistency in values across differentiation methods."""
import abc
import inspect
import functools

import tensorflow as tf


def catch_empty_inputs(func):
    """Helper function for differentiators to correctly handle empty cases.

    Adds support to decorated function for the case when `programs` or
    `symbol_values` is empty which requires output to be
    `tf.zeros_like(symbol_values)`.
    """

    @functools.wraps(func)
    def new_diff(*args, **kwargs):
        # args[1]=programs. args[2]=symbol_names. args[3]=symbol_values
        programs = args[1]
        symbol_names = args[2]
        symbol_values = args[3]
        empty_args = tf.equal(tf.size(programs), 0)
        empty_vals = tf.equal(tf.size(symbol_values), 0)
        empty_symbols = tf.equal(tf.size(symbol_names), 0)

        ret_zero = tf.logical_or(empty_args, empty_vals)
        ret_zero = tf.logical_or(ret_zero, empty_symbols)
        return tf.cond(ret_zero, lambda: tf.zeros_like(symbol_values),
                       lambda: func(*args, **kwargs))

    return new_diff


class Differentiator(metaclass=abc.ABCMeta):
    """Interface that defines how to specify gradients for a quantum circuit.

    This abstract class allows for the creation of gradient calculation
    procedures for (expectation values from) quantum circuits, with
    respect to a set of input parameter values. This allows one
    to backpropagate through a quantum circuit.
    """

    def generate_differentiable_op(self, *, sampled_op=None, analytic_op=None):
        """Generate a differentiable op by attaching self to an op.

        This function returns a `tf.function` that passes values through to
        `forward_op` during the forward pass and this differentiator (`self`) to
        backpropagate through the op during the backward pass. If sampled_op
        is provided the differentiators `differentiate_sampled` method will
        be invoked (which requires sampled_op to be a sample based expectation
        op with num_samples input tensor). If analytic_op is provided the
        differentiators `differentiate_analytic` method will be invoked (which
        requires analytic_op to be an analytic based expectation op that does
        NOT have num_samples as an input). If both sampled_op and analytic_op
        are provided an exception will be raised.

        ***CAUTION***

        This `generate_differentiable_op()` can be called only ONCE because
        of the `one differentiator per op` policy. You need to call `refresh()`
        to reuse this differentiator with another op.

        Args:
            sampled_op: A `callable` op that you want to make differentiable
                using this differentiator's `differentiate_sampled` method.
            analytic_op: A `callable` op that you want to make differentiable
                using this differentiators `differentiate_analytic` method.

        Returns:
            A `callable` op that who's gradients are now registered to be
            a call to this differentiators `differentiate_*` function.

        """
        if hasattr(self, 'expectation_op'):
            raise TypeError('This differentiator is already used for other '
                            'op by calling generate_differentiable_op before. '
                            'You need to call `refresh()` to reuse this '
                            'differentiator with another op.')

        if sampled_op is None and analytic_op is None:
            raise ValueError('generate_differentiable_op requires a sample '
                             'based expectation op to be provided with arg '
                             '\'sampled_op\', or an analytically '
                             'calculated expectation op to be provided with '
                             'arg \'analytic_op\'.')

        if sampled_op is not None and analytic_op is not None:
            raise ValueError('generate_differentiable_op was given both a '
                             'sampled_op and analytic_op. '
                             'Please provide analytic_op if the '
                             'operation you wish to make differentiable is '
                             'analytical. Otherwise provide '
                             'sampled_op if the operation you want '
                             'to make differentiable is sample based.')

        if not callable(sampled_op) and not callable(analytic_op):
            raise TypeError('Provided arguments must be callable tensorflow '
                            'ops.')

        # TODO (mbbrough): find a better workaround than this to ensure
        #   that the correct sample based expectation wasn't accidentally
        #   put inside of the analytical_op argument or vice versa.
        #   right all that is checked is that the desire op signatures
        #   are substrings of the given op signature.
        if analytic_op is not None:
            signature = inspect.signature(analytic_op).parameters
            expected_signature = [
                'programs', 'symbol_names', 'symbol_values', 'pauli_sums'
            ]
            for key in expected_signature:
                if not any(key in s for s in signature):
                    raise ValueError('unexpected signature for analytic_op. '
                                     'Given arg: {}.'.format(str(key)) + ''
                                     'The signature should contain: {}.'.format(
                                         list(expected_signature)) + ''
                                     ' Given: {}'.format(list(signature)) + ''
                                     'Note: noisy ops should use sampled_op')

            if 'num_samples' in signature:
                raise ValueError('found num_samples in analytic_op. Please '
                                 'ensure that you are providing an analytical '
                                 'expectation op in the analytic_op arg.'
                                 'Note: noisy ops should use sampled_op')

        if sampled_op is not None:
            signature = inspect.signature(sampled_op).parameters
            expected_signature = [
                'programs', 'symbol_names', 'symbol_values', 'pauli_sums',
                'num_samples'
            ]
            for key in expected_signature:
                if not any(key in s for s in signature):
                    raise ValueError('unexpected signature for sampled_op. '
                                     'Given arg: {}.'.format(str(key)) + ''
                                     'The signature should contain: {}.'.format(
                                         list(expected_signature)))

        @tf.custom_gradient
        def op_wrapper_analytic(programs, symbol_names, symbol_values,
                                pauli_sums):
            forward_pass_vals = analytic_op(programs, symbol_names,
                                            symbol_values, pauli_sums)

            def gradient(grad):
                return self._differentiate_ana(programs, symbol_names,
                                               symbol_values, pauli_sums,
                                               forward_pass_vals, grad)

            return forward_pass_vals, gradient

        @tf.custom_gradient
        def op_wrapper_sampled(programs, symbol_names, symbol_values,
                               pauli_sums, num_samples):
            forward_pass_vals = sampled_op(programs, symbol_names,
                                           symbol_values, pauli_sums,
                                           num_samples)

            def gradient(grad):
                return self._differentiate_sam(programs, symbol_names,
                                               symbol_values, pauli_sums,
                                               num_samples, forward_pass_vals,
                                               grad)

            return forward_pass_vals, gradient

        self.expectation_op = analytic_op
        return_func = op_wrapper_analytic
        if analytic_op is None:
            self.expectation_op = sampled_op
            return_func = op_wrapper_sampled

        return return_func

    def _differentiate_ana(self, programs, symbol_names, symbol_values,
                           pauli_sums, forward_pass_vals, grad):
        return None, None, self.differentiate_analytic(
            programs, symbol_names, symbol_values,
            pauli_sums, forward_pass_vals, grad), \
               None

    def _differentiate_sam(self, programs, symbol_names, symbol_values,
                           pauli_sums, num_samples, forward_pass_vals, grad):
        return None, None, self.differentiate_sampled(
            programs, symbol_names, symbol_values,
            pauli_sums, num_samples, forward_pass_vals, grad), \
               None, None

    def refresh(self):
        """Refresh this differentiator in order to use it with other ops."""
        # Now that self.expectation_op is removed, users can call
        # generate_differentiable_op() again.
        if hasattr(self, 'expectation_op'):
            del self.expectation_op
        return self

    @abc.abstractmethod
    def get_gradient_circuits(self, programs, symbol_names, symbol_values):
        """Return circuits to compute gradients for given forward pass circuits.

        Prepares (but does not execute) all intermediate circuits needed to
        calculate the gradients for the given forward pass circuits specified by
        `programs`, `symbol_names`, and `symbol_values`. The returned
        `tf.Tensor` objects give all necessary information to recreate the
        internal logic of the differentiator.

        This base class defines the standard way to use the outputs of this
        function to obtain either analytic gradients or sample gradients.
        Below is code that is copied directly from the `differentiate_analytic`
        default implementation, which is then compared to how one could
        automatically get this gradient.  The point is that the derivatives of
        some functions cannot be calculated via the available auto-diff (such
        as when the function is not expressible efficiently as a PauliSum),
        and then one would need to use `get_gradient_circuits` the manual way.

        Suppose we have some inputs `programs`, `symbol_names`, and
        `symbol_values`.  To get the derivative of the expectation values of a
        tensor of PauliSums `pauli_sums` with respect to these inputs, do:


        >>> diff = <some differentiator>()
        >>> (
        ...     batch_programs, new_symbol_names, batch_symbol_values,
        ...     batch_weights, batch_mapper
        ... ) = diff.get_gradient_circuits(
        ...     programs, symbol_names, symbol_values)
        >>> exp_layer = tfq.layers.Expectation()
        >>> batch_pauli_sums = tf.tile(
        ...     tf.expand_dims(pauli_sums, 1),
        ...     [1, tf.shape(batch_programs)[1], 1])
        >>> n_batch_programs = tf.reduce_prod(tf.shape(batch_programs))
        >>> n_symbols = tf.shape(new_symbol_names)[0]
        >>> n_ops = tf.shape(pauli_sums)[1]
        >>> batch_expectations = tfq.layers.Expectation()(
        ...     tf.reshape(batch_programs, [n_batch_programs]),
        ...     symbol_names=new_symbol_names,
        ...     symbol_values=tf.reshape(
        ...         batch_symbol_values, [n_batch_programs, n_symbols]),
        ...     operators=tf.reshape(
        ...         batch_pauli_sums, [n_batch_programs, n_ops]))
        >>> batch_expectations = tf.reshape(
        ...     batch_expectations, tf.shape(batch_pauli_sums))
        >>> batch_jacobian = tf.map_fn(
        ...     lambda x: tf.einsum('km,kmp->kp', x[0], tf.gather(x[1], x[2])),
        ...     (batch_weights, batch_expectations, batch_mapper),
        ...     fn_output_signature=tf.float32)
        >>> grad_manual = tf.reduce_sum(batch_jacobian, -1)


        To perform the same gradient calculation automatically:


        >>> with tf.GradientTape() as g:
        >>>     g.watch(symbol_values)
        >>>     exact_outputs = tfq.layers.Expectation()(
        ...         programs, symbol_names=symbol_names,
        ...         symbol_values=symbol_values, operators=pauli_sums)
        >>> grad_auto = g.gradient(exact_outputs, symbol_values)
        >>> tf.math.reduce_all(grad_manual == grad_auto).numpy()
        True


        NOTE: this feature is intended for advanced users who need more
        flexibility than the standard workflow allows.


        Args:
            programs: `tf.Tensor` of strings with shape [batch_size] containing
                the string representations of the circuits to be executed during
                the forward pass.
            symbol_names: `tf.Tensor` of strings with shape [n_params], which is
                used to specify the order in which the values in `symbol_values`
                should be placed inside of the circuits in `programs`.
            symbol_values: `tf.Tensor` of real numbers with shape
                [batch_size, n_params] specifying parameter values to resolve
                into the circuits specified by programs during the forward pass,
                following the ordering dictated by `symbol_names`.

        Returns:
            batch_programs: 2-D `tf.Tensor` of strings representing circuits to
                run to evaluate the gradients. The first dimension is the length
                of the input `programs`. At each index `i` in the first
                dimension is the tensor of circuits required to evaluate the
                gradient of the input circuit `programs[i]`.  The size of the
                second dimension is determined by the inheriting differentiator.
            new_symbol_names: `tf.Tensor` of strings, containing the name of
                every symbol used in every circuit in `batch_programs`. The
                length is determined by the inheriting differentiator.
            batch_symbol_values: 3-D `tf.Tensor` of DType `tf.float32`
                containing values to fill in to every parameter in every
                circuit. The first two dimensions are the same shape as
                `batch_programs`; the last dimension is the length of
                `new_symbol_names`. Thus, at each index `i` in the first
                dimension is the 2-D tensor of parameter values to fill in to
                `batch_programs[i]`.
            batch_weights: 3-D `tf.Tensor` of DType `tf.float32` which defines
                how much weight to give to each program when computing the
                derivatives.  First dimension is the length of the input
                `programs`, second dimension is the length of the input
                `symbol_names`, and the third dimension is determined by the
                inheriting differentiator.
            batch_mapper: 3-D `tf.Tensor` of DType `tf.int32` which defines
                how to map expectation values of the circuits generated by this
                differentiator to the derivatives of the original circuits.
                It says which indices of the returned programs are relevant for
                the derivative of each symbol, for use by `tf.gather`.
                The first dimension is the length of the input `programs`, the
                second dimension is the length of the input `symbol_names`,
                and the third dimension is the length of the last dimension of
                the output `batch_weights`.
        """

    @catch_empty_inputs
    @tf.function
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):
        """Differentiate a circuit with analytical expectation.

        This is called at graph runtime by TensorFlow. `differentiate_analytic`
        calls he inheriting differentiator's `get_gradient_circuits` and uses
        those components to construct the gradient.

        Note: the default implementation does not use `forward_pass_vals`; the
        inheriting differentiator is free to override the default implementation
        and use this argument if desired.

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
            A `tf.Tensor` with the same shape as `symbol_values` representing
            the gradient backpropageted to the `symbol_values` input of the op
            you are differentiating through.
        """
        (batch_programs, new_symbol_names, batch_symbol_values, batch_weights,
         batch_mapper) = self.get_gradient_circuits(programs, symbol_names,
                                                    symbol_values)
        m_i = tf.shape(batch_programs)[1]
        batch_pauli_sums = tf.tile(tf.expand_dims(pauli_sums, 1), [1, m_i, 1])
        n_batch_programs = tf.reduce_prod(tf.shape(batch_programs))
        n_symbols = tf.shape(new_symbol_names)[0]
        n_ops = tf.shape(pauli_sums)[1]
        batch_expectations = self.expectation_op(
            tf.reshape(batch_programs, [n_batch_programs]), new_symbol_names,
            tf.reshape(batch_symbol_values, [n_batch_programs, n_symbols]),
            tf.reshape(batch_pauli_sums, [n_batch_programs, n_ops]))
        batch_expectations = tf.reshape(batch_expectations,
                                        tf.shape(batch_pauli_sums))

        # has shape [n_programs, n_symbols, n_ops]
        batch_jacobian = tf.map_fn(
            lambda x: tf.einsum('sm,smo->so', x[0], tf.gather(x[1], x[2])),
            (batch_weights, batch_expectations, batch_mapper),
            fn_output_signature=tf.float32)

        # now apply the chain rule
        return tf.einsum('pso,po->ps', batch_jacobian, grad)

    @catch_empty_inputs
    @tf.function
    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):
        """Differentiate a circuit with sampled expectation.

        This is called at graph runtime by TensorFlow. `differentiate_sampled`
        calls he inheriting differentiator's `get_gradient_circuits` and uses
        those components to construct the gradient.

        Note: the default implementation does not use `forward_pass_vals`; the
        inheriting differentiator is free to override the default implementation
        and use this argument if desired.

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
            num_samples: `tf.Tensor` of positive integers representing the
                number of samples per term in each term of pauli_sums used
                during the forward pass.
            forward_pass_vals: `tf.Tensor` of real numbers with shape
                [batch_size, n_ops] containing the output of the forward pass
                through the op you are differentiating.
            grad: `tf.Tensor` of real numbers with shape [batch_size, n_ops]
                representing the gradient backpropagated to the output of the
                op you are differentiating through.

        Returns:
            A `tf.Tensor` with the same shape as `symbol_values` representing
            the gradient backpropageted to the `symbol_values` input of the op
            you are differentiating through.
        """
        (batch_programs, new_symbol_names, batch_symbol_values, batch_weights,
         batch_mapper) = self.get_gradient_circuits(programs, symbol_names,
                                                    symbol_values)
        m_i = tf.shape(batch_programs)[1]
        batch_pauli_sums = tf.tile(tf.expand_dims(pauli_sums, 1), [1, m_i, 1])
        batch_num_samples = tf.tile(tf.expand_dims(num_samples, 1), [1, m_i, 1])
        n_batch_programs = tf.reduce_prod(tf.shape(batch_programs))
        n_symbols = tf.shape(new_symbol_names)[0]
        n_ops = tf.shape(pauli_sums)[1]
        batch_expectations = self.expectation_op(
            tf.reshape(batch_programs, [n_batch_programs]), new_symbol_names,
            tf.reshape(batch_symbol_values, [n_batch_programs, n_symbols]),
            tf.reshape(batch_pauli_sums, [n_batch_programs, n_ops]),
            tf.reshape(batch_num_samples, [n_batch_programs, n_ops]))
        batch_expectations = tf.reshape(batch_expectations,
                                        tf.shape(batch_pauli_sums))

        # has shape [n_programs, n_symbols, n_ops]
        batch_jacobian = tf.map_fn(
            lambda x: tf.einsum('sm,smo->so', x[0], tf.gather(x[1], x[2])),
            (batch_weights, batch_expectations, batch_mapper),
            fn_output_signature=tf.float32)

        # now apply the chain rule
        return tf.einsum('pso,po->ps', batch_jacobian, grad)
