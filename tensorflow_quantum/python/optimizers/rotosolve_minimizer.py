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
"""The rotosolve minimization algorithm
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import math

import collections


def prefer_static_shape(x):
    """Return static shape of tensor `x` if available, else `tf.shape(x)`.
    Args:
      x: `Tensor` (already converted).
    Returns:
      Numpy array (if static shape is obtainable), else `Tensor`.
    """
    return prefer_static_value(tf.shape(x))


def prefer_static_value(x):
    """Return static value of tensor `x` if available, else `x`.
    Args:
      x: `Tensor` (already converted).
    Returns:
      Numpy array (if static value is obtainable), else `Tensor`.
    """
    static_x = tf.get_static_value(x)
    if static_x is not None:
        return static_x
    return x


RotosolveOptimizerResults = collections.namedtuple(
    'RotosolveOptimizerResults', [
        'converged',  # Scalar boolean tensor indicating whether the minimum
        # was found within tolerance.
        'num_iterations',  # The number of iterations of the BFGS update.
        'num_objective_evaluations',  # The total number of objective evaluations performed.
        'position',  # A tensor containing the last argument value found
        # during the search. If the search converged, then
        # this value is the argmin of the objective function.
        'last_objective_value',  # A tensor containing the value of the objective from previous iteration
        'objective_value',  # A tensor containing the value of the objective
        # function at the `position`. If the search
        # converged, then this is the (local) minimum of
        # the objective function.
        'tolerance',  # Define the stop criteria. Iteration will stop when the objective value difference
        # between two iterations is smaller than tolerance 
        'solve_param_i',  # The parameter index where rotosolve is currently modifying. Reserved for internal use.
    ])


def _get_initial_state(initial_position,
                       tolerance, expectation_value_function):
    """Create RotosolveOptimizerResults with initial state of search ."""
    init_args = {
        "converged": tf.Variable(False),
        "num_iterations": tf.Variable(0),
        "num_objective_evaluations": expectation_value_function.iter,
        "position": tf.Variable(initial_position),
        "objective_value": tf.Variable(0.),
        "last_objective_value": tf.Variable(0.),
        "tolerance": tolerance,
        "solve_param_i": tf.Variable(0)
    }
    return RotosolveOptimizerResults(**init_args)


def minimize(expectation_value_function,
             initial_position,
             tolerance=1e-8,
             max_iterations=50,
             name=None):
    """Applies the rotosolve algorithm to minimize a linear combination of quantum measurement
        expectation values. See arXiv:1903.12166, arXiv:1905.09692

    ### Usage:

    The following example demonstrates the Rotosolve optimizer attempting to find the
    minimum for two qubit ansatz expectation value.

    ```python

    # In this example we train a circuit perform an XOR operation
    X = np.asarray([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ], dtype=float)

    Y = np.asarray([
        [-1], [1], [1], [-1]
    ], dtype=float)

    def convert_to_circuit(input_data):
        # Encode into quantum datapoint.
        values = np.ndarray.flatten(input_data)
        qubits = cirq.GridQubit.rect(1, 2)
        circuit = cirq.Circuit()
        for i, value in enumerate(values):
            if value:
                circuit.append(cirq.X(qubits[i]))
        return circuit

    x_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in X])

    # Create two qubits
    q0, q1 = cirq.GridQubit.rect(1, 2)

    # Create an anzatz on these qubits.
    a, b = sympy.symbols('a b') # parameters for the circuit
    circuit = cirq.Circuit(
        cirq.rx(a).on(q0),
        cirq.ry(b).on(q1), cirq.CNOT(control=q0, target=q1))

    # Build the Keras model.
    model = tf.keras.Sequential([
        # The input is the data-circuit, encoded as a tf.string
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        # The PQC layer returns the expected value of the readout gate, range [-1,1].
        tfq.layers.PQC(circuit, cirq.Z(q1)),
    ])

    def hinge_loss(y_true, y_pred):
        # Here we use hinge loss as the cost function
        tf.reduce_mean(tf.cast(1 - y_true * y_pred, tf.float32))

    result = rotosolve_minimizer.minimize( rotosolve_minimizer.function_factory(model, hinge_loss, x_circ, Y),
                                              np.random.rand([2])) # Initial guess of the parameter

    print(result)
    ```

    Args:
      expectation_value_function:  A Python callable that accepts a point as a
        real `Tensor` and returns a tuple of `Tensor`s of real dtype containing
        the value of the function and its gradient at that point. The function
        to be minimized. The input is of shape `[..., n]`, where `n` is the size
        of the domain of input points, and all others are batching dimensions.
        The first component of the return value is a real `Tensor` of matching
        shape `[...]`. The second component (the gradient) is also of shape
        `[..., n]` like the input value to the function. This must be a linear
        combination of quantum measurement expectation value, otherwise this 
        algorithm cannot work.
      initial_position: Real `Tensor` of shape `[..., n]`. The starting point, or
        points when using batching dimensions, of the search procedure. At these
        points the function value and the gradient norm should be finite.
      tolerance: Scalar `Tensor` of real dtype. Specifies the gradient tolerance
        for the procedure. If the supremum norm of the gradient vector is below
        this number, the algorithm is stopped.
      name: (Optional) Python str. The name prefixed to the ops created by this
        function. If not supplied, the default name 'minimize' is used.

    Returns:
      optimizer_results: A RotosolveOptimizerResults object contains the result of the optimization process.
    """

    with tf.name_scope(name or 'minimize'):
        initial_position = tf.convert_to_tensor(
            initial_position, name='initial_position', dtype='float32')
        dtype = initial_position.dtype.base_dtype
        tolerance = tf.convert_to_tensor(
            tolerance, dtype=dtype, name='grad_tolerance')
        max_iterations = tf.convert_to_tensor(max_iterations, name='max_iterations')

        def _rotosolve_one_parameter_once(state):
            """
            Rotosolve a single parameter.
            """
            delta_shift = tf.reshape(tf.cast(tf.sparse.to_dense(
                tf.sparse.SparseTensor([[state.solve_param_i, 0]], [math.pi / 2],
                                       [prefer_static_shape(state.position)[0], 1])), dtype=dtype),
                prefer_static_shape(state.position))

            # Evaluate three different point for curve fitting
            v_l, v_n, v_r = expectation_value_function(state.position - delta_shift), \
                            state.objective_value, \
                            expectation_value_function(state.position + delta_shift)

            # Use the analytical solution to find the optimized position
            delta_update = -math.pi / 2 - tf.math.atan2(2 * v_n - v_l - v_r, v_r - v_l)

            delta_update_tensor = tf.reshape(tf.cast(tf.sparse.to_dense(
                tf.sparse.SparseTensor([[state.solve_param_i, 0]], [delta_update],
                                       [prefer_static_shape(state.position)[0], 1])), dtype=dtype),
                prefer_static_shape(state.position))

            state.solve_param_i.assign_add(1)
            state.position.assign(tf.math.floormod(state.position + delta_update_tensor, math.pi * 2))

            state.last_objective_value.assign(state.objective_value)
            state.objective_value.assign(expectation_value_function(state.position))

            return [state]

        def _rotosolve_all_parameters_once(state):
            """
            Loop iterate over all parameters and update each one of them
            """

            def _cond_internal(state_cond):
                return state_cond.solve_param_i < prefer_static_shape(state_cond.position)[0]

            return tf.while_loop(
                cond=_cond_internal, body=_rotosolve_one_parameter_once, loop_vars=[state], parallel_iterations=1,
            )

        # The `state` here is a `RotosolveOptimizerResults` tuple with values for the
        # current state of the algorithm computation.
        def _cond(state):
            """Continue if iterations remain and stopping condition is not met."""
            return (state.num_iterations < max_iterations) and (not state.converged)

        def _body(state):
            """Main optimization loop."""

            state.solve_param_i.assign(0)

            _rotosolve_all_parameters_once(state)

            state.num_iterations.assign_add(1)
            state.converged.assign(tf.abs(state.objective_value - state.last_objective_value) < state.tolerance)

            return [state]

        initial_state = _get_initial_state(initial_position, tolerance, expectation_value_function)

        initial_state.objective_value.assign(expectation_value_function(initial_state.position))

        return tf.while_loop(
            cond=_cond,
            body=_body,
            loop_vars=[initial_state],
            parallel_iterations=1)[0]
