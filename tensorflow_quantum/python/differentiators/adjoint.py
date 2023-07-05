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
import tensorflow as tf

from tensorflow_quantum.core.ops import tfq_adj_grad_op
from tensorflow_quantum.python.differentiators import differentiator


class Adjoint(differentiator.Differentiator):
    """Differentiate a circuit with respect to its inputs by adjoint method.

    **Caution** This differentiator is only compatible with analytic expectation
    calculations and the native C++ ops (`backend = None`). The methods used by
    this differentiation techniques can not be realized easily on a real device.

    The Adjoint differentiator follows along with the methods described here:
    [arXiv:1912.10877](https://arxiv.org/abs/1912.10877) and
    [ doi: 10.1111/j.1365-246X.2006.02978.x](
    https://academic.oup.com/gji/article-pdf/167/2/495/1492368/167-2-495.pdf).
    The Adjoint method differentiates the input circuits in roughly one forward
    and backward pass over the circuits, to calculate the gradient of
    a symbol only a constant number of gate operations need to be applied to the
    circuits state. When the number of parameters in a circuit is very large,
    this differentiator performs much better than all the others found in TFQ.


    >>> my_op = tfq.get_expectation_op()
    >>> adjoint_differentiator = tfq.differentiators.Adjoint()
    >>> # Get an expectation op, with this differentiator attached.
    >>> op = adjoint_differentiator.generate_differentiable_op(
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
    >>> grads = g.gradient(expectations, symbol_values_t)
    >>> grads
    tf.Tensor([[-1.1839]], shape=(1, 1), dtype=float32)

    """

    def generate_differentiable_op(self, *, sampled_op=None, analytic_op=None):
        """Generate a differentiable op by attaching self to an op.

        See `tfq.differentiators.Differentiator`. This has been partially
        re-implemented by the Adjoint differentiator to disallow the
        `sampled_op` input.


        Args:
            sampled_op: A `callable` op that you want to make differentiable
                using this differentiator's `differentiate_sampled` method.
            analytic_op: A `callable` op that you want to make differentiable
                using this differentiators `differentiate_analytic` method.

        Returns:
            A `callable` op that who's gradients are now registered to be
            a call to this differentiators `differentiate_*` function.

        """
        if sampled_op is not None:
            raise ValueError("sample base backends are not supported by the "
                             "Adjoint method, please use analytic expectation"
                             " or choose another differentiator.")

        return super().generate_differentiable_op(analytic_op=analytic_op)

    @tf.function
    def get_gradient_circuits(self, programs, symbol_names, symbol_values):
        """See base class description."""
        raise NotImplementedError(
            "Adjoint differentiator cannot run on a real QPU, "
            "therefore it has no accessible gradient circuits.")

    @differentiator.catch_empty_inputs
    @tf.function
    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):
        return tfq_adj_grad_op.tfq_adj_grad(programs, symbol_names,
                                            symbol_values, pauli_sums, grad)

    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              pauli_sums, num_samples, forward_pass_vals, grad):
        raise NotImplementedError(
            "Adjoint state methods are not supported in sample based settings."
            " Please use analytic expectation calculation or a different "
            "tfq.differentiator.")
