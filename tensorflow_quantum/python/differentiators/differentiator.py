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
    def get_gradient_circuits(self, programs, symbol_names, symbol_values):
        """Transforms programs to allow gradient calculations on real QPUs.

        In the classical setting, backpropagation can be used to compute
        derivatives of an objective function f(x) with respect to its inputs x.
        Backpropagation works by recursively applying the chain rule to the
        graph of computations defining f.

        Moving towards the quantum setting, suppose there is some objective
        function f(|s(x)>), where |s(x)> is the state produced by some quantum
        circuit with parameters x. If one simulates the circuit on a classical
        computer, then one can similarly apply the chain rule to f(|s(x)>) to
        obtain the derivative; this is the approach taken by the adjoint
        differentiator in TFQ.

        If the quantum state |s(x)> is computed on a quantum processing unit
        (QPU), backpropagation is no longer possible. Instead, the classical
        computer, where the objective function resides, can only access
        bitstrings sampled from states produced on the QPU; derivatives must be
        computed from these samples instead. This function exposes the circuits,
        symbols, and values required for computing such gradients on real QPUs.

        NOTE: most users should simply hand a differentiator to the circuit
        executor of their choice (for example, see the docs for
        `tfq.layers.Expectation`, which accepts a `differentiator` initializer).
        This function is intended for advanced users who need more flexibility
        than the standard workflow allows.

        Args:
            programs: `tf.Tensor` of strings with shape [batch_size] containing
                the string representations of the circuits to be executed.
            symbol_names: `tf.Tensor` of strings with shape [n_params], which is
                used to specify the order in which the values in `symbol_values`
                should be placed inside of the circuits in `programs`.
            symbol_values: `tf.Tensor` of real numbers with shape
                [batch_size, n_params] specifying parameter values to resolve
                into the circuits specified by programs, following the ordering
                dictated by `symbol_names`.

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
            batch_mapper: 3-D `tf.Tensor` of DType `tf.float32` which defines
                how to map expectation values of the circuits generated by this
                differentiator to the derivatives of the original circuits.
                The first dimension is the length of the input `programs`, the
                second dimension is the length of the input `symbol_names`,
                and the third dimension is the length of the second dimension of
                the output `batch_programs`.

        If the caller has a list of functions `f` with vector-valued outputs of
        length [n_vals] to apply to a batch of circuits, then the derivative of
        the computation will have shape [batch_size, n_params, n_vals]. Then in
        terms of the inputs and outputs of `get_gradient_circuits`, each entry
        of the derivative can be computed as

        d(<programs[i](symbol_values[i])|
            f[i]|programs[i](symbol_values[i])>)
        / d(symbol_names[k]) = sum_m (batch_mapper[i, k, m] *
            (<batch_programs[i, m](batch_symbol_values[i, m])|
                f[i]|batch_programs[i, m](batch_symbol_values[i, mi])>)

        where d represents taking the partial derivative.
        """


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
