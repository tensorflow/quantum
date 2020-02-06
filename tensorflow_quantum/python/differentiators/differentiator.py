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

import tensorflow as tf


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
                                     ' Given: {}'.format(list(signature)))

            if 'num_samples' in signature:
                raise ValueError('found num_samples in analytic_op. Please '
                                 'ensure that you are providing an analytical '
                                 'expectation op in the analytic_op arg.')

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
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):
        """Specify how to differentiate a circuit with analytical expectation.

        This is called at graph runtime by TensorFlow. `differentiate_analytic`
        should calculate the gradient of a batch of circuits and return it
        formatted as indicated below. See
        `tfq.differentiators.ForwardDifference` for an example.

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

    @abc.abstractmethod
    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):
        """Specify how to differentiate a circuit with sampled expectation.

        This is called at graph runtime by TensorFlow. `differentiate_sampled`
        should calculate the gradient of a batch of circuits and return it
        formatted as indicated below. See
        `tfq.differentiators.ForwardDifference` for an example.

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
