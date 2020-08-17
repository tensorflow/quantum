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
"""Tests for the differentiator abstract class."""
import tensorflow as tf
from tensorflow_quantum.python.differentiators import differentiator


class WorkingDifferentiator(differentiator.Differentiator):
    """test."""

    def get_intermediate_logic(self, programs, symbol_names, symbol_values,
                               pauli_sums):
        """test."""

    def differentiate_analytic(self, programs, symbol_names, symbol_values,
                               pauli_sums, forward_pass_vals, grad):
        """test."""

    def differentiate_sampled(self, programs, symbol_names, symbol_values,
                              num_samples, pauli_sums, forward_pass_vals, grad):
        """test."""


class DifferentiatorTest(tf.test.TestCase):
    """Test that we can properly subclass differentiator."""

    def test_subclass(self):
        """Test that the BaseDifferentiator can be subclassed."""
        WorkingDifferentiator()

    def test_subclass_missing_differentiate(self):
        """Test that BaseDifferentiator enforces abstract method existence."""

        class BrokenDifferentiator(differentiator.Differentiator):
            """test."""

        with self.assertRaisesRegex(TypeError, expected_regex="instantiate"):
            BrokenDifferentiator()

    def test_generate_differentiable_op(self):
        """test the type checking on this method."""
        WorkingDifferentiator().generate_differentiable_op(
            analytic_op=lambda programs, symbol_names, symbol_values,
            pauli_sums: 1)
        WorkingDifferentiator().generate_differentiable_op(
            sampled_op=lambda programs, symbol_names, symbol_values, pauli_sums,
            num_samples: 1)
        with self.assertRaisesRegex(TypeError, expected_regex='callable'):
            WorkingDifferentiator().generate_differentiable_op(analytic_op=1)
        with self.assertRaisesRegex(ValueError, expected_regex='given both'):
            WorkingDifferentiator().generate_differentiable_op(
                analytic_op=lambda: 1, sampled_op=lambda: 1)
        with self.assertRaisesRegex(ValueError, expected_regex='analytic_op'):
            WorkingDifferentiator().generate_differentiable_op(
                analytic_op=lambda programs, symbol_names, symbol_values: 1)
        with self.assertRaisesRegex(
                ValueError, expected_regex='num_samples in analytic_op'):
            WorkingDifferentiator().generate_differentiable_op(
                analytic_op=lambda programs, symbol_names, symbol_values,
                pauli_sums, num_samples: 1)
        with self.assertRaisesRegex(ValueError, expected_regex='sampled_op'):
            WorkingDifferentiator().generate_differentiable_op(
                sampled_op=lambda programs, symbol_names, pauli_sums: 1)

    def test_single_op_link(self):
        """Tests if the `one-differentiator-per-op` policy is working well."""
        wd = WorkingDifferentiator()
        wd.generate_differentiable_op(analytic_op=lambda programs, symbol_names,
                                      symbol_values, pauli_sums: 1)
        with self.assertRaisesRegex(TypeError, expected_regex='already used'):
            wd.generate_differentiable_op(
                analytic_op=lambda programs, symbol_names, symbol_values,
                pauli_sums: 1)
            wd.generate_differentiable_op(
                sampled_op=lambda programs, symbol_names, symbol_values,
                pauli_sums: 1)
        wd.refresh()
        wd.generate_differentiable_op(analytic_op=lambda programs, symbol_names,
                                      symbol_values, pauli_sums: 1)


if __name__ == '__main__':
    tf.test.main()
