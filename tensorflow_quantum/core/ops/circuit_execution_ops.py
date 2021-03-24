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
"""A module for user-facing generators of tfq ops."""
import enum

import cirq

from tensorflow_quantum.core.ops import (cirq_ops, tfq_simulate_ops,
                                         tfq_utility_ops)
from tensorflow_quantum.python import quantum_context


class TFQStateVectorSimulator(enum.Enum):
    """Enum to make specifying TFQ simulators user-friendly."""
    expectation = tfq_simulate_ops.tfq_simulate_expectation
    samples = tfq_simulate_ops.tfq_simulate_samples
    state = tfq_simulate_ops.tfq_simulate_state
    sampled_expectation = tfq_simulate_ops.tfq_simulate_sampled_expectation


def _check_quantum_concurrent(quantum_concurrent):
    if not isinstance(quantum_concurrent, bool):
        raise TypeError("quantum_concurrent must be type bool."
                        " Given: {}".format(str(type(quantum_concurrent))))


def get_expectation_op(
        backend=None,
        *,
        quantum_concurrent=quantum_context.get_quantum_concurrent_op_mode()):
    """Get a TensorFlow op that will calculate batches of expectation values.

    This function produces a non-differentiable TF op that will calculate
    batches of expectation values given tensor batches of `cirq.Circuit`s,
    parameter values, and `cirq.PauliSum` operators to measure.


    >>> # Simulate circuits with C++.
    >>> my_op = tfq.get_expectation_op()
    >>> # Prepare some inputs.
    >>> qubit = cirq.GridQubit(0, 0)
    >>> my_symbol = sympy.Symbol('alpha')
    >>> my_circuit_tensor = tfq.convert_to_tensor([
    ...     cirq.Circuit(cirq.H(qubit) ** my_symbol)
    ... ])
    >>> my_values = np.array([[0.123]])
    >>> my_paulis = tfq.convert_to_tensor([[
    ...     3.5 * cirq.X(qubit) - 2.2 * cirq.Y(qubit)
    ... ]])
    >>> # This op can now be run with:
    >>> output = my_op(
    ...     my_circuit_tensor, ['alpha'], my_values, my_paulis)
    >>> output
    tf.Tensor([[0.71530885]], shape=(1, 1), dtype=float32)


    In order to make the op differentiable, a `tfq.differentiator` object is
    needed. see `tfq.differentiators` for more details. Below is a simple
    example of how to make my_op from the above code block differentiable:

    >>> diff = tfq.differentiators.ForwardDifference()
    >>> my_differentiable_op = diff.generate_differentiable_op(
    ...     analytic_op=my_op
    ... )


    Args:
        backend: Optional Python `object` that specifies what backend this op
            should use when evaluating circuits. Can be
            `cirq.DensityMatrixSimulator` or any
            `cirq.sim.simulator.SimulatesExpectationValues`. If not provided the
            default C++ analytical expectation calculation op is returned.
        quantum_concurrent: Optional Python `bool`. True indicates that the
            returned op should not block graph level parallelism on itself when
            executing. False indicates that graph level parallelism on itself
            should be blocked. Defaults to value specified in
            `tfq.get_quantum_concurrent_op_mode` which defaults to True
            (no blocking). This flag is only needed for advanced users when
            using TFQ for very large simulations, or when running on a real
            chip.

    Returns:
        A `callable` with the following signature:

        ```op(programs, symbol_names, symbol_values, pauli_sums)```

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

        Returns:
            `tf.Tensor` with shape [batch_size, n_ops] that holds the
                expectation value for each circuit with each op applied to it
                (after resolving the corresponding parameters in).
    """

    # TODO (mbbrough): investigate how the above docstring renders.
    _check_quantum_concurrent(quantum_concurrent)

    op = None
    if backend is None:
        op = TFQStateVectorSimulator.expectation

    # TODO(zaqqwerty): remove DM check after cirq #3964
    if isinstance(backend, (cirq.sim.simulator.SimulatesExpectationValues,
                            cirq.DensityMatrixSimulator)):
        op = cirq_ops._get_cirq_analytical_expectation(backend)

    if op is not None:
        if quantum_concurrent is True:
            # Return an op that does not block graph level parallelism.
            return lambda programs, symbol_names, symbol_values, pauli_sums: \
                op(programs, symbol_names, symbol_values, pauli_sums)

        # Return an op that does block graph level parallelism.
        return lambda programs, symbol_names, symbol_values, pauli_sums: \
            quantum_context._GLOBAL_OP_LOCK.execute(lambda: op(
                programs, symbol_names, symbol_values, pauli_sums))

    if isinstance(backend, (cirq.SimulatesSamples, cirq.Sampler)):
        raise NotImplementedError("Sample-based expectation is not supported."
                                  " Use "
                                  "tf.get_sampled_expectation_op() instead.")

    raise TypeError("Backend {} is invalid. Expected a "
                    "cirq.sim.simulator.SimulatesExpectationValues "
                    "or None.".format(backend))


def get_sampling_op(
        backend=None,
        *,
        quantum_concurrent=quantum_context.get_quantum_concurrent_op_mode()):
    """Get a Tensorflow op that produces samples from given quantum circuits.

    This function produces a non-differentiable op that will calculate
    batches of circuit samples given tensor batches of `cirq.Circuit`s,
    parameter values, and a scalar telling the op how many samples to take.


    >>> # Simulate circuits with cirq.
    >>> my_op = tfq.get_sampling_op(backend=cirq.sim.Simulator())
    >>> # Simulate circuits with C++.
    >>> my_second_op = tfq.get_sampling_op()
    >>> # Prepare some inputs.
    >>> qubit = cirq.GridQubit(0, 0)
    >>> my_symbol = sympy.Symbol('alpha')
    >>> my_circuit_tensor = tfq.convert_to_tensor(
    ...     [cirq.Circuit(cirq.X(qubit)**my_symbol)])
    >>> my_values = np.array([[2.0]])
    >>> n_samples = np.array([10])
    >>> # This op can now be run to take samples.
    >>> output = my_second_op(
    ...     my_circuit_tensor, ['alpha'], my_values, n_samples)
    >>> output
    <tf.RaggedTensor [[[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]]>


    Args:
        backend: Optional Python `object` that specifies what backend this op
            should use when evaluating circuits. Can be any `cirq.Sampler`. If
            not provided the default C++ sampling op is returned.
        quantum_concurrent: Optional Python `bool`. True indicates that the
            returned op should not block graph level parallelism on itself when
            executing. False indicates that graph level parallelism on itself
            should be blocked. Defaults to value specified in
            `tfq.get_quantum_concurrent_op_mode` which defaults to True
            (no blocking). This flag is only needed for advanced users when
            using TFQ for very large simulations, or when running on a real
            chip.

    Returns:
        A `callable` with the following signature:

        ```op(programs, symbol_names, symbol_values, num_samples)```

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
        num_samples: `tf.Tensor` with one element indicating the number of
            samples to draw.

        Returns:
            `tf.Tensor` with shape
                [batch_size, num_samples, <ragged> n_qubits] that
                holds samples (as boolean values) for each circuit.
    """

    # TODO (mbbrough): investigate how the above docstring renders.
    _check_quantum_concurrent(quantum_concurrent)

    op = None
    if backend is None:
        op = TFQStateVectorSimulator.samples

    if isinstance(backend, cirq.Sampler):
        op = cirq_ops._get_cirq_samples(backend)

    if op is not None:
        if quantum_concurrent is True:
            # Return an op that does not block graph level parallelism.
            return lambda programs, symbol_names, symbol_values, num_samples: \
                tfq_utility_ops.padded_to_ragged(
                    op(programs, symbol_names, symbol_values, num_samples))

        return lambda programs, symbol_names, symbol_values, num_samples: \
            quantum_context._GLOBAL_OP_LOCK.execute(
                lambda: tfq_utility_ops.padded_to_ragged(op(
                    programs, symbol_names, symbol_values, num_samples)))

    raise TypeError("Backend {} is invalid. Expected a Cirq.Sampler "
                    "or None.".format(backend))


def get_state_op(
        backend=None,
        *,
        quantum_concurrent=quantum_context.get_quantum_concurrent_op_mode()):
    """Get a TensorFlow op that produces states from given quantum circuits.

    This function produces a non-differentiable op that will calculate
    batches of state tensors given tensor batches of `cirq.Circuit`s and
    parameter values.


    >>> # Simulate circuits with cirq.
    >>> my_op = tfq.get_state_op(backend=cirq.DensityMatrixSimulator())
    >>> # Simulate circuits with C++.
    >>> my_second_op = tfq.get_state_op()
    >>> # Prepare some inputs.
    >>> qubit = cirq.GridQubit(0, 0)
    >>> my_symbol = sympy.Symbol('alpha')
    >>> my_circuit_tensor = tfq.convert_to_tensor([
    ...     cirq.Circuit(cirq.Y(qubit) ** my_symbol)
    ... ])
    >>> my_values = np.array([[0.5]])
    >>> # This op can now be run to calculate the state.
    >>> output = my_second_op(my_circuit_tensor, ['alpha'], my_values)
    >>> output
    <tf.RaggedTensor [[(0.5+0.5j), (0.5+0.5j)]]>


    Args:
        backend: Optional Python `object` that specifies what backend this op
            should use when evaluating circuits. Can be any
            `cirq.SimulatesFinalState`. If not provided, the default C++
            state vector simulator will be used.
        quantum_concurrent: Optional Python `bool`. True indicates that the
            returned op should not block graph level parallelism on itself when
            executing. False indicates that graph level parallelism on itself
            should be blocked. Defaults to value specified in
            `tfq.get_quantum_concurrent_op_mode` which defaults to True
            (no blocking). This flag is only needed for advanced users when
            using TFQ for very large simulations, or when running on a real
            chip.

    Returns:
        A `callable` with the following signature:

        ```op(programs, symbol_names, symbol_values)```

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

        Returns:
            `tf.Tensor` with shape [batch_size, <ragged> size of state] that
            contains the state information of the circuit.
    """

    # TODO (mbbrough): investigate how the above docstring renders.
    _check_quantum_concurrent(quantum_concurrent)

    op = None
    if backend is None:
        op = TFQStateVectorSimulator.state

    if isinstance(backend, (cirq.SimulatesFinalState)):
        op = cirq_ops._get_cirq_simulate_state(backend)

    if op is not None:
        if quantum_concurrent is True:
            # Return an op that does not block graph level parallelism.
            return lambda programs, symbol_names, symbol_values: \
                tfq_utility_ops.padded_to_ragged(
                    op(programs, symbol_names, symbol_values))

        # Return an op that does block graph level parallelism.
        return lambda programs, symbol_names, symbol_values: \
            quantum_context._GLOBAL_OP_LOCK.execute(
                lambda: tfq_utility_ops.padded_to_ragged(op(
                    programs, symbol_names, symbol_values)))

    raise TypeError("Backend {} is invalid. Expected a Cirq.SimulatesFinalState"
                    " or None.".format(backend))


def get_sampled_expectation_op(
        backend=None,
        *,
        quantum_concurrent=quantum_context.get_quantum_concurrent_op_mode()):
    """Get a TensorFlow op that will calculate sampled expectation values.

    This function produces a non-differentiable TF op that will calculate
    batches of expectation values given tensor batches of `cirq.Circuit`s,
    parameter values, and `cirq.PauliSum` operators to measure.
    Expectation is estimated by taking num_samples shots per term in the
    corresponding PauliSum.


    >>> # Simulate circuits with C++.
    >>> my_op = tfq.get_sampled_expectation_op()
    >>> # Prepare some inputs.
    >>> qubit = cirq.GridQubit(0, 0)
    >>> my_symbol = sympy.Symbol('alpha')
    >>> my_circuit_tensor = tfq.convert_to_tensor([
    ...     cirq.Circuit(cirq.H(qubit) ** my_symbol)
    ... ])
    >>> my_values = np.array([[0.123]])
    >>> my_paulis = tfq.convert_to_tensor([[
    ...     3.5 * cirq.X(qubit) - 2.2 * cirq.Y(qubit)
    ... ]])
    >>> my_num_samples = np.array([[100]])
    >>> # This op can now be run with:
    >>> output = my_op(
    ...     my_circuit_tensor, ['alpha'], my_values, my_paulis, my_num_samples)
    >>> output
    tf.Tensor([[0.71530885]], shape=(1, 1), dtype=float32)


    In order to make the op differentiable, a `tfq.differentiator` object is
    needed. see `tfq.differentiators` for more details. Below is a simple
    example of how to make my_op from the above code block differentiable:


    >>> diff = tfq.differentiators.ForwardDifference()
    >>> my_differentiable_op = diff.generate_differentiable_op(
    ...     analytic_op=my_op
    ... )

    Args:
        backend: Optional Python `object` that specifies what backend this op
            should use when evaluating circuits. Can be any `cirq.Sampler`. If
            not provided the default C++ sampled expectation op is returned.
        quantum_concurrent: Optional Python `bool`. True indicates that the
            returned op should not block graph level parallelism on itself when
            executing. False indicates that graph level parallelism on itself
            should be blocked. Defaults to value specified in
            `tfq.get_quantum_concurrent_op_mode` which defaults to True
            (no blocking). This flag is only needed for advanced users when
            using TFQ for very large simulations, or when running on a real
            chip.

    Returns:
        A `callable` with the following signature:

        ```op(programs, symbol_names, symbol_values, pauli_sums, num_samples)```

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
        num_samples: `tf.Tensor` with `num_samples[i][j]` is equal to the
            number of samples to draw in each term of `pauli_sums[i][j]`
            when estimating the expectation. Therefore, `num_samples` must
            have the same shape as `pauli_sums`.

        Returns:
            `tf.Tensor` with shape [batch_size, n_ops] that holds the
                expectation value for each circuit with each op applied to it
                (after resolving the corresponding parameters in).
    """
    # TODO (mbbrough): investigate how the above docstring renders.
    _check_quantum_concurrent(quantum_concurrent)

    op = None
    if backend is None:
        op = TFQStateVectorSimulator.sampled_expectation

    if isinstance(backend, cirq.Sampler):
        op = cirq_ops._get_cirq_sampled_expectation(backend)

    if op is not None:
        if quantum_concurrent is True:
            # Return an op that does not block graph level parallelism.
            return lambda programs, symbol_names, symbol_values, pauli_sums, \
                num_samples: op(programs,
                                symbol_names,
                                symbol_values,
                                pauli_sums,
                                num_samples)

        # Return an op that does block graph level parallelism.
        return lambda programs, symbol_names, symbol_values, pauli_sums, \
            num_samples: quantum_context._GLOBAL_OP_LOCK.execute(
                lambda: op(programs,
                           symbol_names,
                           symbol_values,
                           pauli_sums,
                           num_samples))

    raise TypeError(
        "Backend {} is invalid. Expected a Cirq.Sampler or None.".format(
            backend))
