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
"""Util tests for tensorflow quantum optimizers"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from absl.testing import parameterized
import numpy as np
from tensorflow_quantum.python.optimizers.utils import function_factory


class RotosolveMinimizerTest(tf.test.TestCase, parameterized.TestCase):
    """Tests for optimizer utils"""

    def test_function_factory(self):
        """Test the function_factory"""

        class LinearModel(object):
            """A simple tensorflow linear model"""

            def __init__(self):
                self.w = tf.Variable(0.0)
                self.b = tf.Variable(0.0)

            def __call__(self, x):
                return self.w * x + self.b

        model = LinearModel()

        def loss(target_y, predicted_y):
            return tf.reduce_mean(tf.square(target_y - predicted_y))

        xs = [[1], [2]]
        ys = [[9], [14]]  # y = x * 5 + 4

        new_func = function_factory(model, loss, xs, ys)

        loss_1 = new_func(np.asarray([2, 3]))  # ys = 5, 7 / loss = 32.5
        loss_2 = new_func(np.asarray([5, 4]))  # ys = 9, 14 / loss = 0

        self.assertNear(loss_1, 32.5, 1e-6)
        self.assertNear(loss_2, 0, 1e-6)
