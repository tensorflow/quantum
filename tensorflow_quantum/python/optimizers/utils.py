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
"""Utils for tensorflow quantum optimizers
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from operator import mul
from functools import reduce
import tensorflow as tf


def function_factory(model, loss, train_x, train_y):
    """A factory to create a function required by tfq.optimizer.rotosolve.
    This function is originally defined for l-bgfs minimizer for tensorflow
    probability package.

    Args:
        model : an instance of `tf.keras.Model` or its subclasses.
        loss : a function with signature loss_value = loss(pred_y, true_y).
        train_x : the input part of training data.
        train_y : the output part of training data.

    Returns:
        A function that has a signature of:
            loss_value = f(model_parameters).
    """

    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = []  # stitch indices
    part = []  # partition indices

    for i, shape in enumerate(shapes):
        n = reduce(mul, shape)
        idx.append(tf.reshape(tf.range(count, count + n, dtype=tf.int32),
                              shape))
        part.extend([i] * n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.

        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's
                trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def exposed_func(params_1d):
        """A function that can be used by tfp.optimizer.rotosolve_minimize.

        This function is created by function_factory.

        Args:
           params_1d [in]: a 1D tf.Tensor.

        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # update the parameters in the model
        assign_new_model_parameters(params_1d)
        # calculate the loss
        loss_value = loss(model(train_x, training=True), train_y)
        exposed_func.iter.assign_add(1)

        return loss_value

    # store these information as members so we can use them outside the scope
    exposed_func.iter = tf.Variable(0)
    exposed_func.idx = idx
    exposed_func.part = part
    exposed_func.shapes = shapes
    exposed_func.assign_new_model_parameters = assign_new_model_parameters

    return exposed_func
