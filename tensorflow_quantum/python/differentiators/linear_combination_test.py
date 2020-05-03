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
"""Basic tests for the LinearCombinationDifferentiator"""
import numpy as np
from absl.testing import parameterized
import tensorflow as tf
import sympy
import cirq

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python import util
from tensorflow_quantum.python.differentiators import linear_combination


def _simple_op_inputs():
    qubit = cirq.GridQubit(0, 0)
    symbol = 'alpha'
    circuit = cirq.Circuit(cirq.Y(qubit)**sympy.Symbol(symbol))
    op = cirq.X(qubit)
    value = 0.3
    n_samples = 2000

    # Return inputs prepped for expectation ops.
    # circuit, symbol_names, values, ops, n_samples
    # along with expected feedforward expectation
    # and expected gradient.
    return (util.convert_to_tensor([circuit]), tf.convert_to_tensor([symbol]),
            tf.convert_to_tensor([[value]]), util.convert_to_tensor([[op]]),
            tf.convert_to_tensor([[n_samples]]),
            tf.convert_to_tensor([[np.sin(np.pi * value)]]),
            tf.convert_to_tensor([[np.pi * np.cos(np.pi * value)]]))


class LinearCombinationTest(tf.test.TestCase, parameterized.TestCase):
    """Test the LinearCombination based Differentiators."""

    def test_linear_combination_instantiate(self):
        """Test LinearCombinationDifferentiator type checking."""
        linear_combination.LinearCombination([1, 1], [1, 0])
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="weights must be"):
            linear_combination.LinearCombination("junk", [1, 0])
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="perturbations must be"):
            linear_combination.LinearCombination([1, 1], "junk")
        with self.assertRaisesRegex(TypeError,
                                    expected_regex="weight in weights"):
            linear_combination.LinearCombination([1, "junk"], [1, 0])
        with self.assertRaisesRegex(
                TypeError, expected_regex="perturbation in perturbations"):
            linear_combination.LinearCombination([1, 1], [1, "junk"])
        with self.assertRaisesRegex(ValueError, expected_regex="length"):
            linear_combination.LinearCombination([1, 1, 1], [1, 0])
        with self.assertRaisesRegex(ValueError, expected_regex="unique"):
            linear_combination.LinearCombination([1, 1], [1, 1])

    def test_forward_instantiate(self):
        """Test ForwardDifference type checking."""
        linear_combination.ForwardDifference()
        linear_combination.ForwardDifference(1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive integer"):
            linear_combination.ForwardDifference(0.1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive integer"):
            linear_combination.ForwardDifference(-1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive integer"):
            linear_combination.ForwardDifference(0, 0.1)
        with self.assertRaisesRegex(ValueError, expected_regex="grid_spacing"):
            linear_combination.ForwardDifference(1, -0.1)
        with self.assertRaisesRegex(ValueError, expected_regex="grid_spacing"):
            linear_combination.ForwardDifference(1, 1j)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'order_coef_perturbs': [(1, (-1, 1), (
                        0, 1)), (2, (-3 / 2, 2, -1 / 2), (0, 1, 2))],
                    'grid_spacing': [0.1, 0.01, 0.5, 1, 0.05]
                })))
    def test_forward_coeffecients(self, order_coef_perturbs, grid_spacing):
        """Test that ForwardDifference produces the right coeffecients for
        common first and second order cases."""
        order = order_coef_perturbs[0]
        expected_std_coeffs = order_coef_perturbs[1]
        expected_perturbations = order_coef_perturbs[2]
        forward = linear_combination.ForwardDifference(order, grid_spacing)
        self.assertAllClose(
            np.array(expected_std_coeffs) / grid_spacing, forward.weights)
        self.assertAllClose(
            np.array(expected_perturbations) * grid_spacing,
            forward.perturbations)

    def test_central_instantiate(self):
        """Test CentralDifference type checking."""
        linear_combination.CentralDifference()
        linear_combination.CentralDifference(2, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive, even"):
            linear_combination.CentralDifference(0.1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive, even"):
            linear_combination.CentralDifference(-1, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive, even"):
            linear_combination.CentralDifference(0, 0.1)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex="positive, even"):
            linear_combination.CentralDifference(1, 0.1)
        with self.assertRaisesRegex(ValueError, expected_regex="grid_spacing"):
            linear_combination.CentralDifference(2, -0.1)
        with self.assertRaisesRegex(ValueError, expected_regex="grid_spacing"):
            linear_combination.CentralDifference(2, 1j)

    @parameterized.parameters(
        list(
            util.kwargs_cartesian_product(
                **{
                    'order_coef_perturbs': [(2, (-1 / 2, 1 / 2), (-1, 1)),
                                            (4, (1 / 12, -8 / 12, 8 / 12,
                                                 -1 / 12), (-2, -1, 1, 2))],
                    'grid_spacing': [0.1, 0.01, 0.5, 1, 0.05]
                })))
    def test_central_coefficients(self, order_coef_perturbs, grid_spacing):
        """Test that CentralDifference produces the right coefficients for
        common first and second order cases."""
        order = order_coef_perturbs[0]
        expected_std_coeffs = order_coef_perturbs[1]
        expected_perturbations = order_coef_perturbs[2]
        forward = linear_combination.CentralDifference(order, grid_spacing)
        self.assertAllClose(
            np.array(expected_std_coeffs) / grid_spacing, forward.weights)
        self.assertAllClose(
            np.array(expected_perturbations) * grid_spacing,
            forward.perturbations)

    @parameterized.parameters([{
        'diff': linear_combination.ForwardDifference()
    }, {
        'diff': linear_combination.CentralDifference()
    }])
    def test_analytic_functional(self, diff):
        """Test that the differentiate_analytic function WORKS."""
        differentiable_op = diff.generate_differentiable_op(
            analytic_op=circuit_execution_ops.get_expectation_op())
        circuit, names, values, ops, _, true_f, true_g = _simple_op_inputs()
        with tf.GradientTape() as g:
            g.watch(values)
            res = differentiable_op(circuit, names, values, ops)

        # Just check that it computes without failing.
        self.assertAllClose(true_f, res, atol=1e-2, rtol=1e-2)
        self.assertAllClose(true_g,
                            g.gradient(res, values),
                            atol=1e-2,
                            rtol=1e-2)

    @parameterized.parameters([{
        'diff': linear_combination.ForwardDifference()
    }, {
        'diff': linear_combination.CentralDifference()
    }])
    def test_sampled_functional(self, diff):
        """Test that the differentiate_sampled function WORKS."""
        differentiable_op = diff.generate_differentiable_op(
            sampled_op=circuit_execution_ops.get_sampled_expectation_op())
        circuit, names, values, ops, n_samples, true_f, true_g = \
            _simple_op_inputs()
        with tf.GradientTape() as g:
            g.watch(values)
            res = differentiable_op(circuit, names, values, ops, n_samples)

        # Just check that it computes without failing.
        self.assertAllClose(true_f, res, atol=1e-1, rtol=1e-1)
        self.assertAllClose(true_g,
                            g.gradient(res, values),
                            atol=1e-1,
                            rtol=1e-1)


if __name__ == "__main__":
    tf.test.main()
