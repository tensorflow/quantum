# Copyright 2021 The TensorFlow Quantum Authors. All Rights Reserved.
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
"""The SPSA minimization algorithm."""
import tensorflow as tf
import numpy as np


def prefer_static_shape(x):
    """Return static shape of tensor `x` if available,

    else `tf.shape(x)`.

    Args:
        x: `tf.Tensor` (already converted).
    Returns:
        Numpy array (if static shape is obtainable), else `tf.Tensor`.
    """
    return prefer_static_value(tf.shape(x))


def prefer_static_value(x):
    """Return static value of tensor `x` if available, else `x`.

    Args:
        x: `tf.Tensor` (already converted).
    Returns:
        Numpy array (if static value is obtainable), else `tf.Tensor`.
    """
    static_x = tf.get_static_value(x)
    if static_x is not None:
        return static_x
    return x


class SPSAOptimizerResults(tf.experimental.ExtensionType):
    """ExtentionType of SPSA Optimizer tf.while_loop() inner state."""
    converged: tf.Tensor
    # Scalar boolean tensor indicating whether the minimum
    # was found within tolerance.
    num_iterations: tf.Tensor
    # The number of iterations of the SPSA update.
    num_objective_evaluations: tf.Tensor
    # The total number of objective
    # evaluations performed.
    position: tf.Tensor
    # A tensor containing the last argument value found
    # during the search. If the search converged, then
    # this value is the argmin of the objective function.
    # A tensor containing the value of the objective from
    # previous iteration
    objective_value_prev: tf.Tensor
    # Save the evaluated value of the objective function
    # from the previous iteration
    objective_value: tf.Tensor
    # A tensor containing the value of the objective
    # function at the `position`. If the search
    # converged, then this is the (local) minimum of
    # the objective function.
    tolerance: tf.Tensor
    # Define the stop criteria. Iteration will stop when the
    # objective value difference between two iterations is
    # smaller than tolerance
    learning_rate: tf.Tensor
    # Specifies the learning rate
    alpha: tf.Tensor
    # Specifies scaling of the learning rate
    perturb: tf.Tensor
    # Specifies the size of the perturbations
    gamma: tf.Tensor
    # Specifies scaling of the size of the perturbations
    blocking: tf.Tensor
    # If true, then the optimizer will only accept updates that improve
    # the objective function.
    allowed_increase: tf.Tensor

    # Specifies maximum allowable increase in objective function
    # (only applies if blocking is true).

    def to_dict(self):
        """Transforms immutable data to mutable dictionary."""
        return {
            "converged": self.converged,
            "num_iterations": self.num_iterations,
            "num_objective_evaluations": self.num_objective_evaluations,
            "position": self.position,
            "objective_value": self.objective_value,
            "objective_value_prev": self.objective_value_prev,
            "tolerance": self.tolerance,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "perturb": self.perturb,
            "gamma": self.gamma,
            "blocking": self.blocking,
            "allowed_increase": self.allowed_increase,
        }


def _get_initial_state(initial_position, tolerance, expectation_value_function,
                       learning_rate, alpha, perturb, gamma, blocking,
                       allowed_increase):
    """Create SPSAOptimizerResults with initial state of search."""
    init_args = {
        "converged": tf.Variable(False),
        "num_iterations": tf.Variable(0),
        "num_objective_evaluations": tf.Variable(0),
        "position": tf.Variable(initial_position),
        "objective_value":
            (tf.cast(expectation_value_function(initial_position), tf.float32)),
        "objective_value_prev": tf.Variable(np.inf),
        "tolerance": tolerance,
        "learning_rate": tf.Variable(learning_rate),
        "alpha": tf.Variable(alpha),
        "perturb": tf.Variable(perturb),
        "gamma": tf.Variable(gamma),
        "blocking": tf.Variable(blocking),
        "allowed_increase": tf.Variable(allowed_increase),
    }
    return SPSAOptimizerResults(**init_args)


def minimize(expectation_value_function,
             initial_position,
             tolerance=1e-5,
             max_iterations=200,
             alpha=0.602,
             learning_rate=1.0,
             perturb=1.0,
             gamma=0.101,
             blocking=False,
             allowed_increase=0.5,
             seed=None,
             name=None):
    """Applies the SPSA algorithm.

    The SPSA algorithm can be used to minimize a noisy function. See:

    [SPSA website](https://www.jhuapl.edu/SPSA/)

    Usage:

    Here is an example of optimize a function which consists the
    summation of a few quadratics.

    >>> n = 5  # Number of quadratics
    >>> coefficient = tf.random.uniform(minval=0, maxval=1, shape=[n])
    >>> min_value = 0
    >>> func = func = lambda x : tf.math.reduce_sum(np.power(x, 2) * \
            coefficient)
    >>> # Optimize the function with SPSA, start with random parameters
    >>> result = tfq.optimizers.spsa_minimize(func, np.random.random(n))
    >>> result.converged
    tf.Tensor(True, shape=(), dtype=bool)
    >>> result.objective_value
    tf.Tensor(0.0013349084, shape=(), dtype=float32)

    Args:
        expectation_value_function:  Python callable that accepts a real
            valued tf.Tensor with shape [n] where n is the number of function
            parameters. The return value is a real `tf.Tensor` Scalar
            (matching shape `[1]`).
        initial_position: Real `tf.Tensor` of shape `[n]`. The starting
            point, or points when using batching dimensions, of the search
            procedure. At these points the function value and the gradient
            norm should be finite.
        tolerance: Scalar `tf.Tensor` of real dtype. Specifies the tolerance
            for the procedure. If the supremum norm between two iteration
            vector is below this number, the algorithm is stopped.
        learning_rate: Scalar `tf.Tensor` of real dtype.
            Specifies the learning rate.
        alpha: Scalar `tf.Tensor` of real dtype. Specifies scaling of the
            learning rate.
        perturb: Scalar `tf.Tensor` of real dtype. Specifies the size of the
            perturbations.
        gamma: Scalar `tf.Tensor` of real dtype. Specifies scaling of the
            size of the perturbations.
        blocking: Boolean. If true, then the optimizer will only accept
            updates that improve the objective function.
        allowed_increase: Scalar `tf.Tensor` of real dtype. Specifies maximum
            allowable increase in objective function (only applies if blocking
            is true).
        seed: (Optional) Python integer. Used to create a random seed for the
            perturbations.
        name: (Optional) Python `str`. The name prefixed to the ops created
            by this function. If not supplied, the default name 'minimize'
            is used.

    Returns:
        optimizer_results: A SPSAOptimizerResults object contains the
            result of the optimization process.
    """

    with tf.name_scope(name or 'minimize'):
        if seed is not None:
            generator = tf.random.Generator.from_seed(seed)
        else:
            generator = tf.random

        initial_position = tf.convert_to_tensor(initial_position,
                                                name='initial_position',
                                                dtype='float32')
        dtype = initial_position.dtype.base_dtype
        tolerance = tf.convert_to_tensor(tolerance,
                                         dtype=dtype,
                                         name='grad_tolerance')
        max_iterations = tf.convert_to_tensor(max_iterations,
                                              name='max_iterations')

        learning_rate_init = tf.convert_to_tensor(learning_rate,
                                                  name='initial_a',
                                                  dtype='float32')
        perturb_init = tf.convert_to_tensor(perturb,
                                            name='initial_c',
                                            dtype='float32')

        def _spsa_once(state):
            """Caclulate single SPSA gradient estimation

            Args:
                state: A SPSAOptimizerResults object stores the
                       current state of the minimizer.

            Returns:
                states: A list which the first element is the new state
            """
            delta_shift = tf.cast(
                2 * generator.uniform(shape=state.position.shape,
                                      minval=0,
                                      maxval=2,
                                      dtype=tf.int32) - 1, tf.float32)
            v_m = expectation_value_function(state.position -
                                             state.perturb * delta_shift)
            v_p = expectation_value_function(state.position +
                                             state.perturb * delta_shift)

            gradient_estimate = (v_p - v_m) / (2 * state.perturb) * delta_shift
            update = state.learning_rate * gradient_estimate
            next_state_params = state.to_dict()
            next_state_params.update({
                "num_objective_evaluations":
                    state.num_objective_evaluations + 2,
            })

            current_obj = tf.cast(expectation_value_function(state.position -
                                                             update),
                                  dtype=tf.float32)
            if state.objective_value_prev + \
                state.allowed_increase >= current_obj or not state.blocking:
                next_state_params.update({
                    "position": state.position - update,
                    "objective_value_prev": state.objective_value,
                    "objective_value": current_obj
                })

            return [SPSAOptimizerResults(**next_state_params)]

        # The `state` here is a `SPSAOptimizerResults` tuple with
        # values for the current state of the algorithm computation.
        def _cond(state):
            """Continue if iterations remain and stopping condition
            is not met."""
            return (state.num_iterations < max_iterations) \
                   and (not state.converged)

        def _body(state):
            """Main optimization loop."""
            new_learning_rate = learning_rate_init / (
                (tf.cast(state.num_iterations + 1, tf.float32) +
                 0.01 * tf.cast(max_iterations, tf.float32))**state.alpha)
            new_perturb = perturb_init / (tf.cast(state.num_iterations + 1,
                                                  tf.float32)**state.gamma)

            pre_state_params = state.to_dict()
            pre_state_params.update({
                "learning_rate": new_learning_rate,
                "perturb": new_perturb,
            })

            post_state = _spsa_once(SPSAOptimizerResults(**pre_state_params))[0]
            post_state_params = post_state.to_dict()
            post_state_params.update({
                "num_iterations":
                    post_state.num_iterations + 1,
                "converged":
                    (tf.abs(state.objective_value - state.objective_value_prev)
                     < state.tolerance),
            })
            return [SPSAOptimizerResults(**post_state_params)]

        initial_state = _get_initial_state(initial_position, tolerance,
                                           expectation_value_function,
                                           learning_rate, alpha, perturb, gamma,
                                           blocking, allowed_increase)

        return tf.while_loop(cond=_cond,
                             body=_body,
                             loop_vars=[initial_state],
                             parallel_iterations=1)[0]
