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
"""Generators for ops that call out to cirq simulators from the tf graph."""
import functools
import numbers

import numpy as np
import tensorflow as tf
import cirq
import cirq_google

from tensorflow_quantum.core.ops import batch_util
from tensorflow_quantum.core.proto import pauli_sum_pb2
from tensorflow_quantum.core.proto import program_pb2
from tensorflow_quantum.core.serialize import serializer


def _upgrade_inputs(op_wrapper):
    """It is helpful to call this on the py_function wrappers you generate,
    as if they are the first element in an eager graph, the inputs
    may or may not already be tensors."""

    @functools.wraps(op_wrapper)
    def wrapper(*args):
        tensorized_args = []
        for arg in args:
            if not tf.is_tensor(arg):
                arg = tf.convert_to_tensor(arg)
            tensorized_args.append(arg)
        return op_wrapper(*tensorized_args)

    return wrapper


def _input_check_helper(programs, symbol_names, symbol_values):
    """Helper function that type checks common inputs.

    Type and size check the `programs`, `symbol_names`, and `symbol_values`
    inputs, which are used by all ops in this module.

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
    """
    if not programs.dtype == tf.dtypes.string:
        raise TypeError('programs tensor must be of type string')
    # if symbol_names is empty it won't be of type string
    if tf.size(symbol_names) > 0 and not symbol_names.dtype == tf.dtypes.string:
        raise TypeError('symbol_names tensor must be of type string')
    if not isinstance(symbol_values.dtype.as_numpy_dtype(), numbers.Real):
        raise TypeError('symbol_values tensor must be a real-valued'
                        ' numeric tensor.')
    if not (int(symbol_values.shape[0]) == int(tf.size(programs))):
        raise ValueError('first dimension of symbol_values tensor'
                         ' must match size of programs tensor.')
    if len(symbol_values.shape) < 2 or not (int(tf.size(symbol_names)) == int(
            symbol_values.shape[1])):
        raise ValueError('size of symbol_names tensor must match second'
                         ' dimension of symbol_values tensor.')


def _batch_deserialize_helper(programs, symbol_names, symbol_values):
    """Helper function that converts tensors to cirq constructs.

     Converts the string representation of the circuits in `programs`
     to `cirq.Circuit` objects and produces a corresponding
     `cirq.ParamResolver` constructed using `symbol_names` and `symbol_values`.

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

    Returns:
        `tuple` containing a `list` of `cirq.Circuit`s constructed from programs
        and a `list` of `cirq.ParamResolver`s.
    """
    de_ser_symbol_names = [x.decode('UTF-8') for x in symbol_names.numpy()]
    de_ser_programs = []
    resolvers = []
    # TODO(zaqqwerty): investigate parallelization of this loop
    for program, values in zip(programs, symbol_values):
        program = program.numpy()
        values = values.numpy().astype(float)

        circuit_proto = program_pb2.Program()
        circuit_proto.ParseFromString(program)

        circuit = serializer.deserialize_circuit(circuit_proto)

        resolver = cirq.study.resolver.ParamResolver(
            dict(zip(de_ser_symbol_names, values)))
        de_ser_programs.append(circuit)
        resolvers.append(resolver)
    return de_ser_programs, resolvers


def _get_cirq_analytical_expectation(simulator=cirq.Simulator()):
    """Get a `callable` that is a TensorFlow op that outputs expectation values.

    Generate a TensorFlow `tf.py_function` op that when called on `tf.Tensor`s
    containing circuits and parameters produces a `tf.Tensor` of expectation
    values.

    Args:
        simulator: `cirq.Simulator` object to use for circuit execution.  Can be
            `cirq.DensityMatrixSimulator` or any
            `cirq.sim.simulator.SimulatesExpectationValues`.

    Returns:
        `callable` that is a TensorFlow op for computing expectation.
    """

    def cirq_analytical_expectation(programs, symbol_names, symbol_values,
                                    pauli_sums):
        """Calculate the expectation value of circuits wrt some operator(s).

        Calculate the expectation value for all the `cirq.PauliSum`s in
        `pauli_sums` on each `cirq.Circuit` in `programs`. Each circuit will
        have the values in `symbol_values` resolved into the symbols in the
        circuit (with the ordering defined by `symbol_names`).

        ```python

        symbol_names = ['a', 'b', 'c']
        programs = tfq.convert_to_tensor(
            [cirq.Circuit(H(q0) ** sympy.Symbol('a'),
                          X(q1) ** sympy.Symbol('b'),
                          Y(q2) ** sympy.Symbol('c'))]
        )

        symbol_values = [[3,2,1]]
        pauli_sums = tfq.convert_to_tensor(
            [1.5 * cirq.Z(q0) * cirq.Z(q1)]
        )

        cirq_analytical_expectation(
            programs, symbol_names, sybmol_values, pauli_sums)
        ```

        Would place the values of 3 into the Symbol labeled 'a', 2 into the
        symbol labeled 'b' and 1 into the symbol labeled 'c'. Then it would
        calculate the ZZ expectation on this circuit.

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

        Returns:
            `tf.Tensor` with shape [batch_size, n_ops] that holds the
                expectation value for each circuit with each op applied to it
                (after resolving the corresponding parameters in).
        """
        _input_check_helper(programs, symbol_names, symbol_values)
        if not (pauli_sums.dtype == tf.dtypes.string):
            raise TypeError('pauli_sums tensor must be of type string.')
        if not (pauli_sums.shape[0] == programs.shape[0]):
            raise TypeError('pauli_sums tensor must have the same batch shape '
                            'as programs tensor.')

        programs, resolvers = _batch_deserialize_helper(programs, symbol_names,
                                                        symbol_values)

        sum_inputs = []
        for sub_list in pauli_sums.numpy():
            to_append = []
            for x in sub_list:
                obj = pauli_sum_pb2.PauliSum()
                obj.ParseFromString(x)
                to_append.append(serializer.deserialize_paulisum(obj))
            sum_inputs.append(to_append)

        expectations = batch_util.batch_calculate_expectation(
            programs, resolvers, sum_inputs, simulator)

        return expectations

    if not isinstance(simulator, (cirq.sim.simulator.SimulatesExpectationValues,
                                  cirq.DensityMatrixSimulator)):
        raise TypeError(
            "simulator must be cirq.DensityMatrixSimulator or inherit "
            "cirq.sim.simulator.SimulatesExpectationValues.")

    @_upgrade_inputs
    def expectation_generator(programs_tf, symbol_names_tf, symbol_values_tf,
                              pauli_sums_tf):
        out = tf.py_function(
            func=cirq_analytical_expectation,
            inp=[
                tf.stop_gradient(programs_tf),
                tf.stop_gradient(symbol_names_tf), symbol_values_tf,
                tf.stop_gradient(pauli_sums_tf)
            ],
            Tout=tf.float32,
        )
        out.set_shape([programs_tf.shape[0], pauli_sums_tf.shape[1]])
        return out

    return expectation_generator


def _get_cirq_sampled_expectation(sampler=cirq.Simulator()):
    """Get a `callable` that is a TensorFlow op that outputs sampled expectation
    values.

    Generate a TensorFlow `tf.py_function` op that when called on `tf.Tensor`s
    containing circuits and parameters produces a `tf.Tensor` of sampled
    expectation values.

    Args:
        sampler: Anything inheriting `cirq.Sampler`.

    Returns:
        `callable` that is a TensorFlow op for computing expectation.
    """

    def cirq_sampled_expectation(programs, symbol_names, symbol_values,
                                 pauli_sums, num_samples):
        """Calculate the sampled expectation value of circuits wrt some
        operator(s).

        Estimates the expectation value for all the `cirq.PauliSum`s in
        `pauli_sums` on each `cirq.Circuit` in `programs`. Each circuit will
        have the values in `symbol_values` resolved into the symbols in the
        circuit (with the ordering defined by `symbol_names`).

        ```python

        symbol_names = ['a', 'b', 'c']
        programs = tfq.convert_to_tensor(
            [cirq.Circuit(H(q0) ** sympy.Symbol('a'),
                          X(q1) ** sympy.Symbol('b'),
                          Y(q2) ** sympy.Symbol('c'))]
        )

        symbol_values = [[3,2,1]]
        pauli_sums = tfq.convert_to_tensor(
            [1.5 * cirq.Z(q0) * cirq.Z(q1)]
        )
        n_samples = [[100]]

        cirq_sampled_expectation(
            programs, symbol_names, sybmol_values, pauli_sums, n_samples)
        ```

        Would place the values of 3 into the Symbol labeled 'a', 2 into the
        symbol labeled 'b' and 1 into the symbol labeled 'c'. Then it would
        estimate the ZZ expectation on this circuit by draw samples from the
        circuit 100 times.

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
            num_samples: `tf.Tensor` with `n_samples[i][j]` is equal to the
                number of samples to draw in each term of `pauli_sums[i][j]`
                when estimating the expectation.

        Returns:
            `tf.Tensor` with shape [batch_size, n_ops] that holds the
                expectation value for each circuit with each op applied to it
                (after resolving the corresponding parameters in).
        """
        _input_check_helper(programs, symbol_names, symbol_values)
        if not (pauli_sums.dtype == tf.dtypes.string):
            raise TypeError('pauli_sums tensor must be of type string.')
        if not (pauli_sums.shape[0] == programs.shape[0]) or \
            len(pauli_sums.shape) != 2:
            raise TypeError('pauli_sums tensor must have the same batch shape '
                            'as programs tensor.')

        if not (num_samples.dtype == tf.dtypes.int32 or
                num_samples.dtype == tf.dtypes.int64):
            raise TypeError('num_samples tensor must be of type int32 of '
                            'int64.')
        if not (num_samples.shape == pauli_sums.shape):
            raise TypeError('num_samples tensor must have the same shape '
                            'as pauli_sums tensor. got: {} expected: {}'.format(
                                num_samples.shape, pauli_sums.shape))
        if tf.less_equal(num_samples, 0).numpy().any():
            raise TypeError('num_samples contains sample value <= 0.')

        programs, resolvers = _batch_deserialize_helper(programs, symbol_names,
                                                        symbol_values)

        num_samples = num_samples.numpy().tolist()

        sum_inputs = []
        for sub_list in pauli_sums.numpy():
            to_append = []
            for x in sub_list:
                obj = pauli_sum_pb2.PauliSum()
                obj.ParseFromString(x)
                to_append.append(serializer.deserialize_paulisum(obj))
            sum_inputs.append(to_append)

        expectations = batch_util.batch_calculate_sampled_expectation(
            programs, resolvers, sum_inputs, num_samples, sampler)

        return expectations

    if not isinstance(sampler, cirq.Sampler):
        raise TypeError("cirq.Sampler is required for sampled expectation.")

    @_upgrade_inputs
    def sampled_expectation_generator(programs_tf, symbol_names_tf,
                                      symbol_values_tf, pauli_sums_tf,
                                      num_samples_tf):
        out = tf.py_function(
            func=cirq_sampled_expectation,
            inp=[
                tf.stop_gradient(programs_tf),
                tf.stop_gradient(symbol_names_tf),
                symbol_values_tf,
                tf.stop_gradient(pauli_sums_tf),
                tf.stop_gradient(num_samples_tf),
            ],
            Tout=tf.float32,
        )
        out.set_shape([programs_tf.shape[0], pauli_sums_tf.shape[1]])
        return out

    return sampled_expectation_generator


# TODO(trevormccrt): should this be removed when differentiators come in ?
def _group_tuples(inputs):
    """Helper that groups a `list` of `tuple`s based on the elements at index 0.

    Given a `list` of `tuple`s, return a `dict` mapping from every unique first
    element in the list to the lists containing the rest of the elements.
    Example:
        [(a,2,3),(b,1,1),(b,1,2),(a,1,1)] -> {a:[(2,3),(1,1)], b:[(1,1),(1,2)]}

    Args:
        input: Python `list` of tuples to group.

    Returns:
        Python `dict` containing groups.
    """
    groups = {}
    for item in inputs:
        current_groups = groups.get(item[0], [])
        current_groups.append(item[1:])
        groups[item[0]] = current_groups
    return groups


def _get_cirq_samples(sampler=cirq.Simulator()):
    """Get a `callable` that is a TensorFlow op that outputs circuit samples.

    Generate a TensorFlow `tf.py_function` op that when called on `tf.Tensor`s
    of circuits and parameters produces a tensor of bitstring samples from all
    the circuits.

    Args:
        sampler: Object inheriting `cirq.Sampler` to use for circuit execution.

    Returns:
        `callable` that is a Tensorflow op for taking samples.
    """
    if not isinstance(sampler, cirq.Sampler):
        raise TypeError("Passed sampler must inherit cirq.Sampler.")

    @tf.custom_gradient
    def cirq_sample(programs, symbol_names, symbol_values, num_samples):
        """Draw samples from circuits.

        Draw samples from `circuits` where each circuit will have the values in
        `symbol_values` resolved into the symbols in the circuit (with the
        ordering defined by `symbol_names`).

        ```python

        symbol_names = ['a', 'b', 'c']
        programs = tfq.convert_to_tensor(
            [cirq.Circuit(H(q0) ** sympy.Symbol('a'),
                          X(q1) ** sympy.Symbol('b'),
                          Y(q2) ** sympy.Symbol('c'))]
        )

        symbol_values = [[3,2,1]]
        n_samples = [100]

        cirq_sample(programs, symbol_names, sybmol_values, n_samples)
        ```

        Would place the values of 3 into the Symbol labeled 'a', 2 into the
        symbol labeled 'b' and 1 into the symbol labeled 'c'. Then it would
        draw 100 samples from the circuit.

        Note: In the case of circuits with varying size, all nonexistant
        samples for a particular circuit are padded with -2.

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
            num_samples: `tf.Tensor` with one element indicating the number of
                samples to draw.

        Returns:
            `tf.Tensor` with shape
                [batch_size, num_samples, <# qubits in largest circuit>] that
                holds samples (as boolean values) for each circuit.
        """

        def _no_grad(grad):
            raise RuntimeError(
                'Differentiation through a sampling operation is not supported.'
            )

        _input_check_helper(programs, symbol_names, symbol_values)

        if not (int(tf.size(num_samples)) == 1):
            raise ValueError("num_samples tensor must have size 1")
        if not isinstance(num_samples.dtype.as_numpy_dtype(), numbers.Integral):
            raise TypeError("num_samples tensor must be of integer type")

        serialized_programs = programs
        programs, resolvers = _batch_deserialize_helper(programs, symbol_names,
                                                        symbol_values)

        num_samples = int(num_samples.numpy())

        if isinstance(sampler, (cirq.Simulator, cirq.DensityMatrixSimulator)):
            # Only local simulators can be handled by batch_sample
            results = batch_util.batch_sample(programs, resolvers, num_samples,
                                              sampler)
            return np.array(results, dtype=np.int8), _no_grad

        # All other samplers need terminal measurement gates.
        programs = [
            p + cirq.Circuit(cirq.measure(*sorted(p.all_qubits()), key='tfq'))
            for p in programs
        ]
        max_n_qubits = max(len(p.all_qubits()) for p in programs)

        if isinstance(sampler, cirq_google.QuantumEngineSampler):
            # group samples from identical circuits to reduce communication
            # overhead. Have to keep track of the order in which things came
            # in to make sure the output is ordered correctly
            to_be_grouped = [
                (ser_prog.numpy(), resolver, index)
                for index, (
                    ser_prog,
                    resolver) in enumerate(zip(serialized_programs, resolvers))
            ]

            grouped = _group_tuples(to_be_grouped)

            # start all the necessary jobs
            results_mapping = {}
            for key, value in grouped.items():
                program = programs[value[0][1]]
                resolvers = [x[0] for x in value]
                orders = [x[1] for x in value]

                # sampler.run_sweep blocks until results are in, so go around it
                result = sampler._engine.run_sweep(
                    program=program,
                    params=resolvers,
                    repetitions=num_samples,
                    processor_ids=sampler._processor_ids,
                    gate_set=sampler._gate_set)
                results_mapping[result] = orders

            # get all results
            cirq_results = [None] * len(programs)
            for key, value in results_mapping.items():
                this_results = key.results()
                for result, index in zip(this_results, value):
                    cirq_results[index] = result

        else:
            # All other cirq.Samplers handled here.
            #TODO(zaqqwerty): replace with run_batch once Cirq #3148 is resolved
            cirq_results = []
            for p, r in zip(programs, resolvers):
                cirq_results.append(sampler.run(p, r, num_samples))

        results = []
        for r in cirq_results:
            results.append(
                tf.keras.preprocessing.sequence.pad_sequences(
                    r.measurements['tfq'],
                    maxlen=max_n_qubits,
                    dtype=np.int8,
                    value=-2,
                    padding='pre'))

        return np.array(results, dtype=np.int8), _no_grad

    @_upgrade_inputs
    def sample_generator(circuit_spec, param_names, param_values, num_samples):
        out = tf.py_function(
            func=cirq_sample,
            inp=[
                tf.stop_gradient(circuit_spec),
                tf.stop_gradient(param_names), param_values,
                tf.stop_gradient(num_samples)
            ],
            Tout=tf.int8,
        )
        out.set_shape([circuit_spec.shape[0], None, None])
        return out

    return sample_generator


def _get_cirq_simulate_state(simulator=cirq.Simulator()):
    """Get a `callable` that is a TensorFlow op that outputs circuit states.

    Generate a TensorFlow `tf.py_function` op that when called on `tf.Tensor`s
    of circuits and parameters produces a `tf.Tensor` containing the final state
    of all the input circuits.

    Args:
        simulator: Simulator object.  Can be any `cirq.SimulatesFinalState`;
            if `simulator` is not a `cirq.DensityMatrixSimulator`, this function
            assumes all final states are dense state vectors.

    Returns:
        `callable` that is a Tensorflow op for calculating states.
    """

    @tf.custom_gradient
    def cirq_simulate_state(programs, symbol_names, symbol_values):
        """Simulate the final state of circuits.

        Calculate the final state of for each `cirq.Circuit` in `programs`
        with the values in `symbol_values` resolved into the symbols in the
        circuit (with the ordering defined by `symbol_names`).

        ```python
        symbol_names = ['a', 'b', 'c']
        programs = tfq.convert_to_tensor(
            [cirq.Circuit(H(q0) ** sympy.Symbol('a'),
                          X(q1) ** sympy.Symbol('b'),
                          Y(q2) ** sympy.Symbol('c'))]
        )

        symbol_values = [[3,2,1]]

        cirq_simulate_state(programs, symbol_names, sybmol_values)
        ```

        Would place the values of 3 into the Symbol labeled 'a', 2 into the
        symbol labeled 'b' and 1 into the symbol labeled 'c'. Then it would
        simulate the final state of the circuit.

        Note: In the case of circuits with varying size, all nonexistent
        amplitudes are padded with -2.

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

        Returns:
            `tf.Tensor` with shape [batch_size, <size of largest circuit state>]
            that contains the state information of the circuit.
        """

        def _no_grad(grad):
            raise RuntimeError(
                'Differentiation through states is not supported.')

        _input_check_helper(programs, symbol_names, symbol_values)

        states = batch_util.batch_calculate_state(
            *_batch_deserialize_helper(programs, symbol_names, symbol_values),
            simulator)

        return states, _no_grad

    if not isinstance(simulator, cirq.SimulatesFinalState):
        raise TypeError("simulator must inherit cirq.SimulatesFinalState.")

    @_upgrade_inputs
    def state_generator(circuit_spec, param_names, param_values):
        out = tf.py_function(
            func=cirq_simulate_state,
            inp=[
                tf.stop_gradient(circuit_spec),
                tf.stop_gradient(param_names),
                param_values,
            ],
            Tout=tf.complex64,
        )
        if isinstance(simulator, cirq.Simulator):
            out.set_shape([circuit_spec.shape[0], None])
        else:
            out.set_shape([circuit_spec.shape[0], None, None])

        return out

    return state_generator
