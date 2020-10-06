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
"""A module to for running Cirq Simulators in parallel."""
import asyncio
import collections
import itertools
import os

import multiprocessing as mp
from multiprocessing.pool import Pool as ProcessPool
import numpy as np
import cirq

from tensorflow_quantum.core.serialize import serializer


# TODO (mbbrough): Remove this workaround class once cirq.PauliSumCollector can
#   be used end to end with engine. This current issue is that
#   cirq.PauliSumCollector does not produce serializable gates for basis
#   conversion.
class TFQPauliSumCollector(cirq.work.collector.Collector):
    """Copy of cirq.PauliSumCollector with some fixes to work with engine."""

    def __init__(self,
                 circuit,
                 observable,
                 *,
                 samples_per_term,
                 max_samples_per_job=1000000):

        observable = cirq.PauliSum.wrap(observable)
        self._circuit = circuit
        self._samples_per_job = max_samples_per_job
        self._pauli_coef_terms = [
            (p / p.coefficient, p.coefficient) for p in observable if p
        ]
        self._identity_offset = 0
        for p in observable:
            if not p:
                self._identity_offset += p.coefficient
        self._zeros = collections.defaultdict(lambda: 0)
        self._ones = collections.defaultdict(lambda: 0)
        self._samples_per_term = samples_per_term
        self._total_samples_requested = 0

    def next_job(self):
        """Get the next job."""
        i = self._total_samples_requested // self._samples_per_term
        if i >= len(self._pauli_coef_terms):
            return None
        pauli, _ = self._pauli_coef_terms[i]
        remaining = self._samples_per_term * (i +
                                              1) - self._total_samples_requested
        amount_to_request = min(remaining, self._samples_per_job)
        self._total_samples_requested += amount_to_request
        return cirq.work.collector.CircuitSampleJob(
            circuit=_fixed_circuit_plus_pauli_string_measurements(
                self._circuit, pauli),
            repetitions=amount_to_request,
            tag=pauli)

    def on_job_result(self, job, result):
        """Post process the `job` and `result` you have."""
        job_id = job.tag
        parities = result.histogram(key='out',
                                    fold_func=lambda bits: np.sum(bits) % 2)
        self._zeros[job_id] += parities[0]
        self._ones[job_id] += parities[1]

    def estimated_energy(self):
        """Sums up the sampled expectations, weighted by their coefficients."""
        energy = 0j
        for pauli_string, coef in self._pauli_coef_terms:
            a = self._zeros[pauli_string]
            b = self._ones[pauli_string]
            if a + b:
                energy += coef * (a - b) / (a + b)
        energy = complex(energy)
        if energy.imag == 0:
            energy = energy.real
        energy += self._identity_offset
        return energy


def _fixed_circuit_plus_pauli_string_measurements(circuit, pauli_string):
    """A circuit measuring the given observable at the end of the given circuit.
    """
    assert pauli_string
    circuit = circuit.copy()
    # Uses cirq.SingleQubitCliffordGates which aren't serializable by engine in
    # cirq 0.6. This is a workaround until fixed.
    # circuit.append(cirq.Moment(pauli_string.to_z_basis_ops()))
    circuit.append(cirq.Moment(cirq.decompose(pauli_string.to_z_basis_ops())))
    circuit.append(
        cirq.Moment([cirq.measure(*sorted(pauli_string.keys()), key='out')]))
    return circuit


def _make_complex_view(shape, init_val):
    """Build a RawArray that will map to the real and imaginary parts of a
    complex number."""
    shape = list(shape)
    shape[-1] *= 2
    data = np.ones(shape, dtype=np.float32) * init_val

    flattened_size = 1
    for dim_size in shape:
        flattened_size *= dim_size
    shared_mem_array = mp.RawArray('f', flattened_size)
    np_view = np.frombuffer(shared_mem_array, dtype=np.float32).reshape(shape)
    np.copyto(np_view, data)
    return shared_mem_array


def _convert_complex_view_to_np(view, shape):
    """Get a numpy view ontop of the rawarray view. Small overhead."""
    shape = list(shape)
    shape[-1] *= 2
    return np.frombuffer(view, dtype=np.float32).reshape(shape)


def _update_complex_np(np_view, i, to_add):
    """Update the shared memory undernath the numpy view.
    to_add is passed by reference since we don't do much with it."""
    np_view[i, ...] = np.pad(to_add,
                             (0, (np_view.shape[-1] // 2 - to_add.shape[-1])),
                             'constant',
                             constant_values=-2).view(np.float32)


def _convert_complex_view_to_result(view, shape):
    """Convert a rawarray view to a numpy array and reindex so that
    the underlying pair of double arrays are squished together to make a
    complex array of half the underlying size."""
    shape = list(shape)
    shape[-1] *= 2
    np_view = np.frombuffer(view, dtype=np.float32).reshape(shape)

    # The below view will cause a re-interpretation of underlying
    # memory so use sparingly.
    return np_view.view(np.complex64)


def _make_simple_view(shape, init_val, dtype, c_code):
    """Make a shared memory view for floating type."""
    data = np.ones(shape, dtype=dtype) * init_val
    flattened_size = 1
    for dim_size in shape:
        flattened_size *= dim_size
    shared_mem_array = mp.RawArray(c_code, flattened_size)
    np_view = np.frombuffer(shared_mem_array, dtype=dtype).reshape(shape)
    np.copyto(np_view, data)
    return shared_mem_array


def _convert_simple_view_to_np(view, dtype, shape):
    """Create a numpy view to a float array, low overhead."""
    return np.frombuffer(view, dtype=dtype).reshape(shape)


def _batch_update_simple_np(np_view, i, to_add):
    """Update the shared memory underneath the numpy view.
    to_add is again passed by reference."""
    np_view[i, ...] = to_add


def _pointwise_update_simple_np(np_view, i, j, to_add):
    """Do a batch and sub-batch index update to numpy view."""
    np_view[i, j, ...] = to_add


def _convert_simple_view_to_result(view, dtype, shape):
    """Convert a RawArray view to final numpy array."""
    return np.frombuffer(view, dtype=dtype).reshape(shape)


def _prep_pool_input_args(indices, *args, slice_args=True):
    """Break down a set of indices, and optional args into a generator
    of length cpu_count."""
    block_size = int(np.ceil(len(indices) / os.cpu_count()))
    for i in range(0, len(indices), block_size):
        if slice_args:
            yield tuple([indices[i:i + block_size]] +
                        [x[i:i + block_size] for x in args])
        else:
            yield tuple([indices[i:i + block_size]] + [x for x in args])


# process are separate from all the other processes,
# so INFO_DICTs will not step on each other.
INFO_DICT = {}


def _setup_dict(array_view, view_shape, simulator, post_process):
    INFO_DICT['arr'] = array_view
    INFO_DICT['shape'] = view_shape
    INFO_DICT['sim'] = simulator
    INFO_DICT['post_process'] = post_process


def _state_worker_func(indices, programs, params):
    """Compute the state vector for each program in indices."""
    x_np = _convert_complex_view_to_np(INFO_DICT['arr'], INFO_DICT['shape'])
    simulator = INFO_DICT['sim']

    for i, index in enumerate(indices):
        result = simulator.simulate(programs[i], params[i])
        final_array = INFO_DICT['post_process'](result).astype(np.complex64)
        _update_complex_np(x_np, index, final_array)


def _analytical_expectation_worker_func(indices, programs, params, ops):
    """Compute the expectation of the op[batch_index], w.r.t
    circuit[batch_index] where batch_index is calculated from indices."""
    x_np = _convert_simple_view_to_np(INFO_DICT['arr'], np.float32,
                                      INFO_DICT['shape'])
    simulator = INFO_DICT['sim']

    # TODO: remove this when picklable.
    for i in range(len(ops)):
        for j in range(len(ops[i])):
            ops[i][j] = serializer.deserialize_paulisum(ops[i][j])

    old_batch_index = -2
    state = -1
    for i, index_tuple in enumerate(indices):
        batch_index = index_tuple[0]
        op_index = index_tuple[1]
        # (#679) Just ignore empty programs.
        if len(programs[batch_index].all_qubits()) == 0:
            continue

        if old_batch_index != batch_index:
            # must compute a new state vector.
            qubit_oder = dict(
                zip(sorted(programs[batch_index].all_qubits()),
                    list(range(len(programs[batch_index].all_qubits())))))
            state = simulator.simulate(programs[batch_index],
                                       params[batch_index])

        result = INFO_DICT['post_process'](ops[batch_index][op_index], state,
                                           qubit_oder)
        _pointwise_update_simple_np(x_np, batch_index, op_index, result)
        old_batch_index = batch_index


def _sample_expectation_worker_func(indices, programs, params, ops, n_samples):
    x_np = _convert_simple_view_to_np(INFO_DICT['arr'], np.float32,
                                      INFO_DICT['shape'])
    simulator = INFO_DICT['sim']

    # TODO: remove this when picklable.
    for i in range(len(ops)):
        for j in range(len(ops[i])):
            ops[i][j] = serializer.deserialize_paulisum(ops[i][j])

    for i, index_tuple in enumerate(indices):
        batch_index = index_tuple[0]
        op_index = index_tuple[1]
        # (#679) Just ignore empty programs.
        if len(programs[batch_index].all_qubits()) == 0:
            continue
        circuit = cirq.resolve_parameters(programs[batch_index],
                                          params[batch_index])

        sampler = TFQPauliSumCollector(
            circuit,
            ops[batch_index][op_index],
            samples_per_term=n_samples[batch_index][op_index])

        asyncio.set_event_loop(asyncio.new_event_loop())
        sampler.collect(simulator, concurrency=1)
        result = sampler.estimated_energy().real

        _pointwise_update_simple_np(x_np, batch_index, op_index, result)


def _sample_worker_func(indices, programs, params, n_samples):
    """Sample n_samples from progams[i] with params[i] placed in it."""
    x_np = _convert_simple_view_to_np(INFO_DICT['arr'], np.int32,
                                      INFO_DICT['shape'])
    simulator = INFO_DICT['sim']

    for i, index in enumerate(indices):
        qubits = sorted(programs[i].all_qubits())
        # (#679) Just ignore empty programs.
        if len(qubits) == 0:
            continue
        state = simulator.simulate(programs[i], params[i])
        samples = INFO_DICT['post_process'](state, len(qubits),
                                            n_samples[i]).astype(np.int32)
        _batch_update_simple_np(
            x_np, index,
            np.pad(samples, ((0, 0), (x_np.shape[2] - len(qubits), 0)),
                   'constant',
                   constant_values=-2))


def _validate_inputs(circuits, param_resolvers, simulator, sim_type):
    """Type check and sanity check inputs."""
    if not isinstance(circuits, (list, tuple, np.ndarray)):
        raise TypeError('circuits must be a list or array.'
                        ' Given: {}'.format(type(circuits)))

    if any(not isinstance(x, cirq.Circuit) for x in circuits):
        raise TypeError('circuits must contain cirq.Circuit objects')

    if not isinstance(param_resolvers, (list, tuple, np.ndarray)):
        raise TypeError('param_resolvers must be a list or array.'
                        ' Given: {}'.format(type(param_resolvers)))

    if any(not isinstance(x, cirq.ParamResolver) for x in param_resolvers):
        raise TypeError('param_resolvers must contain cirq.ParamResolvers.')

    if not (len(circuits) == len(param_resolvers)):
        raise ValueError('Circuit batch size does not match resolve batch '
                         'size.')

    if sim_type == 'analytic':
        if not isinstance(simulator, cirq.SimulatesFinalState):
            raise TypeError('For analytic operations only'
                            ' cirq.SimulatesFinalState'
                            ' is required. Given: {}'.format(type(simulator)))
    elif sim_type == 'sample':
        if not isinstance(simulator, cirq.Sampler):
            raise TypeError('For sample based operations a cirq.Sampler is '
                            'required. Given: {}'.format(type(simulator)))
    else:
        raise ValueError('Invalid simulator type specified.')


def batch_calculate_state(circuits, param_resolvers, simulator):
    """Compute states using a given simulator using parallel processing.

    Returns a NumPy array containing the final circuit state for each
    `cirq.Circuit` in `circuits`, given that the corresponding
    `cirq.ParamResolver` in `param_resolvers` was used to resolve any symbols
    in it. If simulator is a `cirq.DensityMatrixSimulator` this final state will
    be a density matrix, if simulator is a `cirq.Simulator` this final state
    will be a state vector. More specifically for a given `i`
    `batch_calculate_state` will use `param_resolvers[i]` to resolve the symbols
    in `circuits[i]` and then place the final state in the return list at index
    `i`.

    Args:
        circuits: Python `list` of `cirq.Circuit`s.
        param_resolvers: Python `list` of `cirq.ParamResolver`s, where
            `param_resolvers[i]` is the resolver to be used with `circuits[i]`.
        simulator: Simulator object. Currently
            supported are `cirq.DensityMatrixSimulator` and `cirq.Simulator`.

    Returns:
        `np.ndarray` containing the resulting state information. The array will
        have dimensions: [len(circuits), <size of biggest state>] in the
        case of `cirq.Simulator`. In the case of `cirq.DensityMatrixSimulator`
        the shape is
         [len(circuits), <size of biggest state>, <size of biggest state>]
    """
    _validate_inputs(circuits, param_resolvers, simulator, 'analytic')

    biggest_circuit = max(len(circuit.all_qubits()) for circuit in circuits)
    if isinstance(simulator, cirq.DensityMatrixSimulator):
        return_mem_shape = (len(circuits), 1 << biggest_circuit,
                            1 << biggest_circuit)
        post_process = lambda x: x.final_density_matrix
    elif isinstance(simulator, cirq.Simulator):
        return_mem_shape = (len(circuits), 1 << biggest_circuit)
        post_process = lambda x: x.final_state_vector
    else:
        raise TypeError('Simulator {} is not supported by '
                        'batch_calculate_state.'.format(type(simulator)))

    shared_array = _make_complex_view(return_mem_shape, -2)
    input_args = _prep_pool_input_args(range(len(circuits)), circuits,
                                       param_resolvers)
    with ProcessPool(processes=None,
                     initializer=_setup_dict,
                     initargs=(shared_array, return_mem_shape, simulator,
                               post_process)) as pool:

        pool.starmap(_state_worker_func, list(input_args))

    return _convert_complex_view_to_result(shared_array, return_mem_shape)


def batch_calculate_expectation(circuits, param_resolvers, ops, simulator):
    """Compute expectations from circuits using parallel processing.

    Returns a `np.ndarray` containing the expectation values of `ops`
    applied to a specific circuit in `circuits`, given that the
    corresponding `cirq.ParamResolver` in `param_resolvers` was used to resolve
    any symbols in the circuit. Specifically the returned array at index `i,j`
    will be equal to the expectation value of `ops[i][j]` on `circuits[i]` with
    `param_resolvers[i]` used to resolve any symbols in `circuits[i]`.
    Expectation calculations will be carried out using the simulator object
    (`cirq.DensityMatrixSimulator` and `cirq.Simulator` are currently supported)

    Args:
        circuits: Python `list` of `cirq.Circuit`s.
        param_resolvers: Python `list` of `cirq.ParamResolver`s, where
            `param_resolvers[i]` is the resolver to be used with `circuits[i]`.
        ops: 2d Python `list` of `cirq.PauliSum` objects where `ops[i][j]` will
            be used to calculate the expectation on `circuits[i]` for all `j`,
            after `param_resolver[i]` is used to resolve any parameters
            in the circuit.
        simulator: Simulator object. Currently supported are
            `cirq.DensityMatrixSimulator` and `cirq.Simulator`.

    Returns:
        `np.ndarray` containing the expectation values. Shape is:
            [len(circuits), len(ops[0])]
    """
    _validate_inputs(circuits, param_resolvers, simulator, 'analytic')
    if not isinstance(ops, (list, tuple, np.ndarray)):
        raise TypeError('ops must be a list or array.'
                        ' Given: {}'.format(type(ops)))

    if len(ops) != len(circuits):
        raise ValueError('Shape of ops and circuits do not match.')

    for sub_list in ops:
        if not isinstance(sub_list, (list, tuple, np.ndarray)):
            raise TypeError('elements of ops must be type list.')
        for x in sub_list:
            if not isinstance(x, cirq.PauliSum):
                raise TypeError('ops must contain only cirq.PauliSum objects.'
                                ' Given: {}'.format(type(x)))

    return_mem_shape = (len(circuits), len(ops[0]))
    if isinstance(simulator, cirq.DensityMatrixSimulator):
        post_process = lambda op, state, order: sum(
            x._expectation_from_density_matrix_no_validation(
                state.final_density_matrix, order) for x in op).real
    elif isinstance(simulator, cirq.Simulator):
        post_process = \
            lambda op, state, order: op.expectation_from_state_vector(
                state.final_state_vector, order).real
    else:
        raise TypeError('Simulator {} is not supported by '
                        'batch_calculate_expectation.'.format(type(simulator)))

    shared_array = _make_simple_view(return_mem_shape, -2, np.float32, 'f')

    # avoid mutating ops array
    ops = np.copy(ops)
    # TODO (mbbrough): make cirq PauliSUms pickable at some point ?
    for i in range(len(ops)):
        for j in range(len(ops[i])):
            ops[i][j] = serializer.serialize_paulisum(ops[i][j])

    input_args = list(
        _prep_pool_input_args(list(
            itertools.product(range(len(circuits)), range(len(ops[0])))),
                              circuits,
                              param_resolvers,
                              ops,
                              slice_args=False))

    with ProcessPool(processes=None,
                     initializer=_setup_dict,
                     initargs=(shared_array, return_mem_shape, simulator,
                               post_process)) as pool:

        pool.starmap(_analytical_expectation_worker_func, input_args)

    return _convert_simple_view_to_result(shared_array, np.float32,
                                          return_mem_shape)


def batch_calculate_sampled_expectation(circuits, param_resolvers, ops,
                                        n_samples, simulator):
    """Compute expectations from sampling circuits using parallel processing.

    Returns a `np.ndarray` containing the expectation values of `ops`
    applied to a specific circuit in `circuits`, given that the
    corresponding `cirq.ParamResolver` in `param_resolvers` was used to resolve
    any symbols in the circuit. Specifically the returned array at index `i,j`
    will be equal to the expectation value of `ops[i][j]` on `circuits[i]` with
    `param_resolvers[i]` used to resolve any symbols in `circuits[i]`.
    Expectation estimations will be carried out using the simulator object
    (`cirq.DensityMatrixSimulator` and `cirq.Simulator` are currently supported)
    . Expectations for ops[i][j] are estimated by drawing n_samples[i][j]
    samples.

    Args:
        circuits: Python `list` of `cirq.Circuit`s.
        param_resolvers: Python `list` of `cirq.ParamResolver`s, where
            `param_resolvers[i]` is the resolver to be used with `circuits[i]`.
        ops: 2d Python `list` of `cirq.PauliSum` objects where `ops[i][j]` will
            be used to calculate the expectation on `circuits[i]` for all `j`,
            after `param_resolver[i]` is used to resolve any parameters
            in the circuit.
        n_samples: 2d Python `list` of `int`s where `n_samples[i][j]` is
            equal to the number of samples to draw in each term of `ops[i][j]`
            when estimating the expectation.
        simulator: Simulator object. Currently supported are
            `cirq.DensityMatrixSimulator` and `cirq.Simulator`.

    Returns:
        `np.ndarray` containing the expectation values. Shape is:
            [len(circuits), len(ops[0])]
    """
    _validate_inputs(circuits, param_resolvers, simulator, 'sample')
    if not isinstance(ops, (list, tuple, np.ndarray)):
        raise TypeError('ops must be a list or array.'
                        ' Given: {}'.format(type(ops)))

    if len(ops) != len(circuits):
        raise ValueError('Shape of ops and circuits do not match.')

    if len(n_samples) != len(circuits):
        raise ValueError('Shape of n_samples does not match circuits.')

    for sub_list in n_samples:
        if not isinstance(sub_list, (list, tuple, np.ndarray)):
            raise TypeError('Elements of n_elements must be lists of ints.')
        for x in sub_list:
            if not isinstance(x, int):
                raise TypeError('Non-integer value found in n_samples.')
            if x <= 0:
                raise ValueError('n_samples contains sample value <= 0.')

    for sub_list in ops:
        if not isinstance(sub_list, (list, tuple, np.ndarray)):
            raise TypeError('elements of ops must be type list.')
        for x in sub_list:
            if not isinstance(x, cirq.PauliSum):
                raise TypeError('ops must contain only cirq.PauliSum objects.'
                                ' Given: {}'.format(type(x)))

    return_mem_shape = (len(circuits), len(ops[0]))
    shared_array = _make_simple_view(return_mem_shape, -2, np.float32, 'f')

    # avoid mutating ops array
    ops = np.copy(ops)
    # TODO (mbbrough): make cirq PauliSums pickable at some point ?
    for i in range(len(ops)):
        for j in range(len(ops[i])):
            ops[i][j] = serializer.serialize_paulisum(ops[i][j])

    input_args = list(
        _prep_pool_input_args(list(
            itertools.product(range(len(circuits)), range(len(ops[0])))),
                              circuits,
                              param_resolvers,
                              ops,
                              n_samples,
                              slice_args=False))

    with ProcessPool(processes=None,
                     initializer=_setup_dict,
                     initargs=(shared_array, return_mem_shape, simulator,
                               None)) as pool:

        pool.starmap(_sample_expectation_worker_func, input_args)

    return _convert_simple_view_to_result(shared_array, np.float32,
                                          return_mem_shape)


def batch_sample(circuits, param_resolvers, n_samples, simulator):
    """Sample from circuits using parallel processing.

    Returns a `np.ndarray` containing n_samples samples from all the circuits in
    circuits given that the corresponding `cirq.ParamResolver` in
    `param_resolvers` was used to resolve any symbols. Specifically the
    returned array at index `i,j` will correspond to a `np.ndarray` of
    booleans representing bitstring `j` that was sampled from `circuits[i]`.
    Samples are drawn using the provided simulator object (Currently supported
    are `cirq.DensityMatrixSimulator` and `cirq.Simulator`).

    Note: In order to keep numpy shape consistent, smaller circuits will
        have sample bitstrings padded with -2 on "qubits that don't exist
        in the circuit".

    Args:
        circuits: Python `list` of `cirq.Circuit`s.
        param_resolvers: Python `list` of `cirq.ParamResolver`s, where
            `param_resolvers[i]` is the resolver to be used with `circuits[i]`.
        n_samples: `int` describing number of samples to draw from each
            circuit.
        simulator: Simulator object. Currently
            supported are `cirq.DensityMatrixSimulator` and `cirq.Simulator`.

    Returns:
        `np.ndarray` containing the samples with invalid qubits blanked out.
        It's shape is
        [len(circuits), n_samples, <# qubits in largest circuit>].
        circuits that are smaller than #qubits in largest circuit have null
        qubits in bitstrings mapped to -2.
    """
    _validate_inputs(circuits, param_resolvers, simulator, 'sample')
    if not isinstance(n_samples, int):
        raise TypeError('n_samples must be an int.'
                        'Given: {}'.format(type(n_samples)))

    if n_samples <= 0:
        raise ValueError('n_samples must be > 0.')

    biggest_circuit = max(len(circuit.all_qubits()) for circuit in circuits)
    return_mem_shape = (len(circuits), n_samples, biggest_circuit)
    shared_array = _make_simple_view(return_mem_shape, -2, np.int32, 'i')

    if isinstance(simulator, cirq.DensityMatrixSimulator):
        post_process = lambda state, size, n_samples: \
            cirq.sample_density_matrix(
                state.final_density_matrix, [i for i in range(size)],
                repetitions=n_samples)
    elif isinstance(simulator, cirq.Simulator):
        post_process = lambda state, size, n_samples: cirq.sample_state_vector(
            state.final_state_vector, list(range(size)), repetitions=n_samples)
    else:
        raise TypeError('Simulator {} is not supported by batch_sample.'.format(
            type(simulator)))

    input_args = list(
        _prep_pool_input_args(range(len(circuits)), circuits, param_resolvers,
                              [n_samples] * len(circuits)))

    with ProcessPool(processes=None,
                     initializer=_setup_dict,
                     initargs=(shared_array, return_mem_shape, simulator,
                               post_process)) as pool:

        pool.starmap(_sample_worker_func, input_args)

    return _convert_simple_view_to_result(shared_array, np.int32,
                                          return_mem_shape)
