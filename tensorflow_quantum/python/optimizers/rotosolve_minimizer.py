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
"""The rotosolve minimization algorithm"""
import collections
import numpy as np
import tensorflow as tf


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


RotosolveOptimizerResults = collections.namedtuple(
    'RotosolveOptimizerResults',
    [
        'converged',
        # Scalar boolean tensor indicating whether the minimum
        # was found within tolerance.
        'num_iterations',
        # The number of iterations of the rotosolve update.
        'num_objective_evaluations',
        # The total number of objective
        # evaluations performed.
        'position',
        # A tensor containing the last argument value found
        # during the search. If the search converged, then
        # this value is the argmin of the objective function.
        # A tensor containing the value of the objective from
        # previous iteration
        'objective_value_previous_iteration',
        # Save the evaluated value of the objective function
        # from the previous iteration
        'objective_value',
        # A tensor containing the value of the objective
        # function at the `position`. If the search
        # converged, then this is the (local) minimum of
        # the objective function.
        'tolerance',
        # Define the stop criteria. Iteration will stop when the
        # objective value difference between two iterations is
        # smaller than tolerance
        'solve_param_i',
        # The parameter index where rotosolve is currently
        # modifying. Reserved for internal use.
    ])


def _get_initial_state(initial_position, tolerance, expectation_value_function):
    """Create RotosolveOptimizerResults with initial state of search."""
    init_args = {
        "converged": tf.Variable(False),
        "num_iterations": tf.Variable(0),
        "num_objective_evaluations": tf.Variable(0),
        "position": tf.Variable(initial_position),
        "objective_value": tf.Variable(0.),
        "objective_value_previous_iteration": tf.Variable(0.),
        "tolerance": tolerance,
        "solve_param_i": tf.Variable(0)
    }
    return RotosolveOptimizerResults(**init_args)


def minimize(expectation_value_function,
             initial_position,
             tolerance=1e-5,
             max_iterations=50,
             name=None):
    """Applies the rotosolve algorithm.

    The rotosolve algorithm can be used to minimize a linear combination

    of quantum measurement expectation values. See the following paper:

    [arXiv:1903.12166](https://arxiv.org/abs/1903.12166), Ken M. Nakanishi.
    [arXiv:1905.09692](https://arxiv.org/abs/1905.09692), Mateusz Ostaszewski.

    Usage:

    Here is an example of optimize a function which consists summation of
    a few sinusoids.

    >>> n = 10  # Number of sinusoids
    >>> coefficient = tf.random.uniform(shape=[n])
    >>> min_value = -tf.math.reduce_sum(tf.abs(coefficient))
    >>> func = lambda x:tf.math.reduce_sum(tf.sin(x) * coefficient)
    >>> # Optimize the function with rotosolve, start with random parameters
    >>> result = tfq.optimizers.rotosolve_minimize(func, np.random.random(n))
    >>> result.converged
    tf.Tensor(True, shape=(), dtype=bool)
    >>> result.objective_value
    tf.Tensor(-4.7045116, shape=(), dtype=float32)

    Args:
        expectation_value_function:  A Python callable that accepts
            a point as a real `tf.Tensor` and returns a `tf.Tensor`s
            of real dtype containing the value of the function.
            The function to be minimized. The input is of shape `[n]`,
            where `n` is the size of the trainable parameters.
            The return value is a real `tf.Tensor` Scalar (matching shape
            `[1]`).  This must be a linear combination of quantum
            measurement expectation value, otherwise this algorithm cannot
            work.
        initial_position: Real `tf.Tensor` of shape `[n]`. The starting
            point, or points when using batching dimensions, of the search
            procedure. At these points the function value and the gradient
            norm should be finite.
        tolerance: Scalar `tf.Tensor` of real dtype. Specifies the tolerance
            for the procedure. If the supremum norm between two iteration
            vector is below this number, the algorithm is stopped.
        name: (Optional) Python `str`. The name prefixed to the ops created
            by this function. If not supplied, the default name 'minimize'
            is used.

    Returns:
        optimizer_results: A RotosolveOptimizerResults object contains the
            result of the optimization process.
    """

    with tf.name_scope(name or 'minimize'):
        initial_position = tf.convert_to_tensor(initial_position,
                                                name='initial_position',
                                                dtype='float32')
        dtype = initial_position.dtype.base_dtype
        tolerance = tf.convert_to_tensor(tolerance,
                                         dtype=dtype,
                                         name='grad_tolerance')
        max_iterations = tf.convert_to_tensor(max_iterations,
                                              name='max_iterations')

        def _rotosolve_one_parameter_once(state):
            """Rotosolve a single parameter once.

            Args:
                state: A RotosolveOptimizerResults object stores the
                       current state of the minimizer.

            Returns:
                states: A list which the first element is the new state
            """
            delta_shift = tf.scatter_nd([[state.solve_param_i]],
                                        [tf.constant(np.pi / 2, dtype=dtype)],
                                        prefer_static_shape(state.position))

            # Evaluate three different point for curve fitting
            v_l, v_n, v_r = expectation_value_function(
                state.position - delta_shift), \
                state.objective_value, \
                expectation_value_function(state.position + delta_shift)

            # Use the analytical solution to find the optimized position
            delta_update = -np.pi / 2 - \
                tf.math.atan2(2 * v_n - v_l - v_r, v_r - v_l)

            delta_update_tensor = tf.scatter_nd(
                [[state.solve_param_i]], [delta_update],
                prefer_static_shape(state.position))

            state.solve_param_i.assign_add(1)
            state.position.assign(
                tf.math.floormod(state.position + delta_update_tensor,
                                 np.pi * 2))

            state.objective_value_previous_iteration.assign(
                state.objective_value)
            state.objective_value.assign(
                expectation_value_function(state.position))

            return [state]

        def _rotosolve_all_parameters_once(state):
            """Iterate over all parameters and rotosolve each single

            of them once.

            Args:
                state: A RotosolveOptimizerResults object stores the
                       current state of the minimizer.

            Returns:
                states: A list which the first element is the new state
            """

            def _cond_internal(state_cond):
                return state_cond.solve_param_i < \
                       prefer_static_shape(state_cond.position)[0]

            state.num_objective_evaluations.assign_add(1)

            return tf.while_loop(
                cond=_cond_internal,
                body=_rotosolve_one_parameter_once,
                loop_vars=[state],
                parallel_iterations=1,
            )

        # The `state` here is a `RotosolveOptimizerResults` tuple with
        # values for the current state of the algorithm computation.
        def _cond(state):
            """Continue if iterations remain and stopping condition
            is not met."""
            return (state.num_iterations < max_iterations) \
                   and (not state.converged)

        def _body(state):
            """Main optimization loop."""

            state.solve_param_i.assign(0)

            _rotosolve_all_parameters_once(state)

            state.num_iterations.assign_add(1)
            state.converged.assign(
                tf.abs(state.objective_value -
                       state.objective_value_previous_iteration) <
                state.tolerance)

            return [state]

        initial_state = _get_initial_state(initial_position, tolerance,
                                           expectation_value_function)

        initial_state.objective_value.assign(
            expectation_value_function(initial_state.position))

        return tf.while_loop(cond=_cond,
                             body=_body,
                             loop_vars=[initial_state],
                             parallel_iterations=1)[0]
