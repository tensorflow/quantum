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
"""Utility functions for stochastic generator differentiator."""
import tensorflow as tf

from tensorflow_quantum.python import util


def _get_pdist_shifts(weights, uniform_sampling):
    """Helper function to calculate probabilistic distributions of sampling
    shifts.
    correction_factor: `tf.Tensor` of real numbers for correction factors
    ${1\over{prob(k)}}={{\sum_k |\gamma_k|}\over{|\gamma_k|}}$ where
    $prob(k)$ is sampling probability of k-th parameter-shift sampling
    term among `n_param_gates` possible samples.
    If uniform_sampling = True, it has integers equal to or less than
    `n_param_gates` because only non-zero terms are considered.
    It has the shape of [sub_total_programs, n_total_samples].
    Args:
        weights: `tf.Tensor` of real numbers for parameter-shift weights.
            [sub_total_programs, n_param_gates]
        uniform_sampling: `tf.Tensor` of a boolean value to decide
            probabilistic distribution of samplers inside.
    Returns:
        corrected_weights: `tf.Tensor` of real numbers for new weights used in
            sampling process. It is multiplied by correction factors as above.
            [sub_total_programs, 1].
        pdist: `tf.Tensor` of probabilistic distribution of given terms
            $prob(k)$
            [sub_total_programs, 1].
    """
    if uniform_sampling:
        non_zeros = tf.cast(tf.not_equal(weights, 0.0), dtype=tf.float32)
        correction_factor = tf.reduce_sum(non_zeros, axis=1, keepdims=True)
        pdist = tf.math.divide_no_nan(non_zeros, correction_factor)
    else:
        weights_abs = tf.abs(weights)
        pdist = tf.math.divide_no_nan(
            weights_abs, tf.reduce_sum(weights_abs, axis=1, keepdims=True))
        correction_factor = tf.math.divide_no_nan(
            1.0, tf.cast(pdist, dtype=tf.float32))
    corrected_weights = correction_factor * weights
    return corrected_weights, pdist


def _sampling_helper_from_pdist_shifts(pdist_shifts, sub_total_programs,
                                       n_param_gates):
    pdist_shifts = tf.reshape(pdist_shifts, [-1, 2 * n_param_gates])
    sampled_idx = tf.random.categorical(
        tf.math.log(pdist_shifts[:, :n_param_gates]), 1)
    sampled_idx = tf.reshape(tf.tile(sampled_idx, [1, 2]),
                             [sub_total_programs, 1])
    return sampled_idx


# TODO(jaeyoo) : this will be c++ op
def stochastic_generator_preprocessor(new_programs, weights, shifts, n_programs,
                                      n_symbols, n_param_gates, n_shifts,
                                      uniform_sampling):
    """Helper function to sample parameter shift rule terms.
    It can be two of the followings:
    - uniform distribution prob(k) = 1/n_param_gates
    - parameter-shift weight-based probability distributions
    Args:
        new_programs: `tf.Tensor' of deserialized parameter-shifted
            program strings with the shape of
            [n_symbols, n_programs, n_param_gates, n_shifts].
        weights: `tf.Tensor` of real numbers for parameter-shift weights.
            [n_symbols, n_programs, n_param_gates, n_shifts]
        shifts: `tf.Tensor` of real numbers for shift values.
            [n_symbols, n_programs, n_param_gates, n_shifts]
        n_programs: `tf.Tensor` of the number of programs.
        n_symbols: `tf.Tensor` of the number of symbols.
        n_param_gates: `tf.Tensor` of the number of maximum parameter gates
            given all programs.
        n_shifts: `tf.Tensor` of the number of shift terms.
        uniform_sampling: `tf.Tensor` of a boolean value to decide
            probabilistic distribution of samplers inside.
    Returns:
        newly sampled new_programs, weights, shifts, whose are `tf.Tensor` with
            the shape of [n_symbols, n_programs, n_param_gates=1, n_shifts].
        n_param_gates: this is used at the post-processing.
    """
    sub_total_programs = n_symbols * n_programs * n_shifts

    # Transpose to [n_symbols, n_programs, n_shifts, n_param_gates]
    new_programs = tf.transpose(new_programs, [0, 1, 3, 2])
    weights = tf.transpose(weights, [0, 1, 3, 2])
    shifts = tf.transpose(shifts, [0, 1, 3, 2])

    new_programs = tf.reshape(new_programs, [sub_total_programs, n_param_gates])
    weights = tf.reshape(weights, [sub_total_programs, n_param_gates])
    shifts = tf.reshape(shifts, [sub_total_programs, n_param_gates])

    corrected_weights, pdist_shifts = _get_pdist_shifts(weights,
                                                        uniform_sampling)

    sampled_idx = _sampling_helper_from_pdist_shifts(pdist_shifts,
                                                     sub_total_programs,
                                                     n_param_gates)
    # TODO(jaeyoo) : make sure all symbols appear in circuit.
    #  not appearing symbols make probability distribution with 0.0 logits.
    #  tf.random.categorical outputs out-of-index value if all logits are 0.0
    #  this makes tf.gather_nd fail due to ouf-of-index error.
    #  for now, it was fixed by adding a dummy additional column.
    #  BUT, find the way to mask no-show symbols, and reduce n_symbols.
    new_programs = tf.concat([new_programs, new_programs[:, :1]], axis=-1)
    corrected_weights = tf.concat(
        [corrected_weights,
         tf.zeros((sub_total_programs, 1))], axis=-1)
    shifts = tf.concat([shifts, tf.zeros([sub_total_programs, 1])], axis=-1)

    new_programs = tf.gather_nd(new_programs, sampled_idx, batch_dims=1)
    weights = tf.gather_nd(corrected_weights, sampled_idx, batch_dims=1)
    shifts = tf.gather_nd(shifts, sampled_idx, batch_dims=1)

    n_param_gates = 1
    new_programs = tf.reshape(new_programs,
                              [n_symbols, n_programs, n_shifts, n_param_gates])
    weights = tf.reshape(weights,
                         [n_symbols, n_programs, n_shifts, n_param_gates])
    shifts = tf.reshape(shifts,
                        [n_symbols, n_programs, n_shifts, n_param_gates])

    # Return back to [n_symbols, n_programs, n_param_gates=1, n_shifts].
    new_programs = tf.transpose(new_programs, [0, 1, 3, 2])
    weights = tf.transpose(weights, [0, 1, 3, 2])
    shifts = tf.transpose(shifts, [0, 1, 3, 2])

    return new_programs, weights, shifts, n_param_gates


def _get_pdist_symbols(weights, uniform_sampling):
    """Helper function to calculate probabilistic distributions of sampling
    symbols.
    correction_factor: `tf.Tensor` of real numbers for correction factors
    ${1\over{prob(k)}}={{\sum_k |\gamma_k|}\over{|\gamma_k|}}$ where
    $prob(k)$ is sampling probability of k-th parameter-shift sampling
    term among `n_symbols` possible samples.
    If uniform_sampling = True, it has integers equal to or less than
    `n_symbols` because only non-zero terms are considered.
    It has the shape of [sub_total_programs, n_symbols].
    Args:
        weights: `tf.Tensor` of real numbers for parameter-shift weights.
            [sub_total_programs, n_total_samples]
        uniform_sampling: `tf.Tensor` of a boolean value to decide
            probabilistic distribution of samplers inside.
    Returns:
        corrected_weights: `tf.Tensor` of real numbers for new weights used in
            sampling process. It is multiplied by correction factors as above.
            [1, n_symbols].
        pdist: `tf.Tensor` of probabilistic distribution of given terms
            $prob(k)$
            [1, n_symbols].
    """
    if uniform_sampling:
        non_zeros = tf.cast(tf.not_equal(weights, 0.0), dtype=tf.float32)
        pdist = tf.reduce_sum(non_zeros, axis=0,
                              keepdims=True) / tf.reduce_sum(non_zeros)
    else:
        weights_abs = tf.abs(weights)
        pdist = tf.reduce_sum(weights_abs, axis=0,
                              keepdims=True) / tf.reduce_sum(weights_abs)

    correction_factor = tf.math.divide_no_nan(
        tf.ones_like(pdist, dtype=tf.float32), tf.cast(pdist, dtype=tf.float32))
    corrected_weights = correction_factor * weights
    return corrected_weights, pdist


# TODO(jaeyoo) : this will be c++ op
def stochastic_coordinate_preprocessor(new_programs,
                                       symbol_values,
                                       pauli_sums,
                                       weights,
                                       shifts,
                                       n_programs,
                                       n_symbols,
                                       n_param_gates,
                                       n_shifts,
                                       n_ops,
                                       uniform_sampling,
                                       num_samples=None):
    """Helper function to sample symbols.
    It can be two of the followings:
    - uniform distribution prob(k) = 1/n_symbols
    - parameter-shift weight-based probability distributions
    Args:
        new_programs: `tf.Tensor' of deserialized parameter-shifted
            program strings with the shape of
            [n_symbols, n_programs, n_param_gates, n_shifts].
        symbol_values: `tf.Tensor` of real numbers with shape
            [n_programs, n_symbols] specifying parameter values to resolve
            into the circuits specified by programs.
        pauli_sums : `tf.Tensor` of strings with shape [n_programs, n_ops]
            representing output observables for each program.
        weights: `tf.Tensor` of real numbers for parameter-shift weights.
            [n_symbols, n_programs, n_param_gates, n_shifts]
        shifts: `tf.Tensor` of real numbers for shift values.
            [n_symbols, n_programs, n_param_gates, n_shifts]
        n_programs: `tf.Tensor` of the number of programs.
        n_symbols: `tf.Tensor` of the number of symbols.
        n_param_gates: `tf.Tensor` of the number of maximum parameter gates
            given all programs.
        n_shifts: `tf.Tensor` of the number of shift terms.
        n_ops: `tf.Tensor` of the number of pauli sums.
        uniform_sampling: `tf.Tensor` of a boolean value to decide
            probabilistic distribution of samplers inside.
        num_samples : Optional `tf.Tensor` of the numbers of samples.
            Defaults to None.
    Returns:
        flat_programs: `tf.Tensor' of the programs of the newly sampled symbols.
            [n_programs * n_param_gates * n_shifts].
        flat_perturbations: `tf.Tensor' of real numbers of perturbations of the
            newly sampled symbols.
            [n_programs * n_param_gates * n_shifts, (n_symbols + 1)].
        flat_ops : `tf.Tensor` of strings of the newly sampled output
            observables.
            [n_programs * n_param_gates * n_shifts, n_ops].
        flat_num_samples : `tf.Tensor` of int32 of the numbers of samples.
            [n_programs * n_param_gates * n_shifts, n_ops].
        weights: `tf.Tensor` of real numbers of re-sampled weights.
            this is used at the post-processing.
            [n_symbols, n_param_gates, n_shifts, n_programs]
        coordinate_relocator: `tf.Tensor` of one-hot matrix with real numbers.
            This is used to restore squeezed symbol dimension at the
            post-processing.
            [n_programs * n_param_gates * n_shifts, n_symbols]
    """
    sub_total_programs = n_programs * n_shifts * n_param_gates
    # [n_param_gates, n_shifts, n_programs, n_symbols]
    new_programs = tf.transpose(new_programs, [1, 2, 3, 0])
    weights = tf.transpose(weights, [1, 2, 3, 0])
    shifts = tf.transpose(shifts, [1, 2, 3, 0])

    new_programs = tf.reshape(new_programs, [sub_total_programs, n_symbols])
    weights = tf.reshape(weights, [sub_total_programs, n_symbols])
    shifts = tf.reshape(shifts, [sub_total_programs, n_symbols])

    corrected_weights, pdist_symbols = _get_pdist_symbols(
        weights, uniform_sampling)

    sampled_idx = tf.transpose(
        tf.random.categorical(tf.math.log(pdist_symbols), sub_total_programs),
        [1, 0])

    flat_programs = tf.gather_nd(new_programs, sampled_idx, batch_dims=1)
    flat_shifts = tf.gather_nd(shifts, sampled_idx, batch_dims=1)
    # It doesn't change n_symbols because it loses locations info of
    # symbol_names. Rather we use one_hot matrix to restore the
    # locations.
    weights = tf.gather_nd(corrected_weights, sampled_idx, batch_dims=1)
    coordinate_relocator = tf.reshape(tf.one_hot(sampled_idx, depth=n_symbols),
                                      [-1, n_symbols])

    # Return back to [n_param_gates, n_shifts, n_programs, n_symbols])
    weights = tf.reshape(tf.einsum('ij,i->ij', coordinate_relocator, weights),
                         [n_param_gates, n_shifts, n_programs, n_symbols])

    # Transpose to the original shape
    # [n_symbols, n_param_gates, n_shifts, n_programs]
    weights = tf.transpose(weights, [3, 0, 1, 2])

    n_sub_tile = n_shifts * n_param_gates
    flat_perturbations = tf.concat([
        tf.reshape(
            tf.tile(tf.expand_dims(symbol_values, 0),
                    tf.stack([n_sub_tile, 1, 1])),
            [sub_total_programs, n_symbols]),
        tf.expand_dims(flat_shifts, axis=1)
    ],
                                   axis=1)
    flat_ops = tf.reshape(
        tf.tile(tf.expand_dims(pauli_sums, 0), tf.stack([n_sub_tile, 1, 1])),
        [sub_total_programs, n_ops])

    flat_num_samples = None
    if num_samples is not None:
        flat_num_samples = tf.reshape(
            tf.tile(tf.expand_dims(num_samples, 0),
                    tf.stack([n_sub_tile, 1, 1])), [sub_total_programs, n_ops])

    return flat_programs, flat_perturbations, flat_ops, flat_num_samples, \
           weights, coordinate_relocator


def _get_parse_pauli_sums():
    """Helper function to obtain the generator of the sampled list of the pauli
    sum coefficients after parsing pauli sums."""

    # TODO(jaeyoo) : this will be c++ op
    def _parse_pauli_sums(pauli_sums, n_programs, n_ops):
        """Helper function to parse given pauli sums to collect observable
        coefficients.
        Currently `cirq.PauliSum` is not subscriptable, which means it is not
        possible to construct a uniform-shape tensor whose elements are
        consistently matched to `cirq.PauliString` inside of given `PauliSum`
        because the order of `PauliString`'s can be different whenever accessed.
        So, the current version of _parse_pauli_sums only consider a `PauliSum`
        to be sampled, not a `PauliString`. The observable coefficients are then
        sum of the absolute value of coefficients of `PauliString`'s in the
        `PauliSum`.
        Args:
            pauli_sums : `tf.Tensor` of strings with shape [n_programs, n_ops]
                representing output observables for each program.
            n_programs: `tf.Tensor` of the number of programs.
            n_ops: `tf.Tensor` of the number of pauli sums.
        Returns:
            observable_coeff_: `tf.Tensor` of real numbers. This involves the
                coefficients of Pauli sum terms of the first PauliString.
                It is directly used to calculate probabilities.
                [n_programs, n_ops]
        """
        pauli_sums = util.from_tensor(pauli_sums)

        def get_pauli_sum_coeff(i):

            def get_i_pauli_sum_coeff(j):

                # Because PauliSum object is not subscriptable, use for-loop.
                # pauli_sums[i][j] : j-th `PauliSum` of i-th program.
                return tf.reduce_sum(
                    tf.abs([
                        pstring.coefficient.real for pstring in pauli_sums[i][j]
                    ]))

            return tf.map_fn(get_i_pauli_sum_coeff,
                             tf.range(n_ops),
                             dtype=tf.float32)

        observable_coeff = tf.map_fn(get_pauli_sum_coeff,
                                     tf.range(n_programs),
                                     dtype=tf.float32)

        return observable_coeff

    def parse_pauli_sums_generator(pauli_sums, n_programs, n_ops):
        """tf.py_function wrapper generator of _parse_programs()."""
        # observable_coeff has the shape of [n_programs, n_ops]
        observable_coeff = tf.py_function(func=_parse_pauli_sums,
                                          inp=[
                                              tf.stop_gradient(pauli_sums),
                                              tf.stop_gradient(n_programs),
                                              tf.stop_gradient(n_ops),
                                          ],
                                          Tout=tf.float32)
        return observable_coeff

    return parse_pauli_sums_generator


def _get_pdist_cost(op_coeff, uniform_sampling):
    """Helper function to calculate probabilistic distributions of sampling
    `PauliSum`'s.
    correction_factor: `tf.Tensor` of real numbers for correction factors
    ${1\over{prob(k)}}={{\sum_k |\gamma_k|}\over{|\gamma_k|}}$ where
    $prob(k)$ is sampling probability of k-th observable PauliSum among `n_ops`
    possible samples.
    If uniform_sampling = True, it has integers equal to or less than
    `n_ops` because only non-zero terms are considered.
    It has the shape of [n_programs, n_ops].
    Args:
        op_coeff: `tf.Tensor` of real numbers for cost Hamiltonian coefficients.
            [n_programs, n_total_samples]
        uniform_sampling: `tf.Tensor` of a boolean value to decide
            probabilistic distribution of samplers inside.
    Returns:
        correction_factor_ops: `tf.Tensor` of real numbers for new observables
            used in sampling process.
            [1, n_ops].
        pdist: `tf.Tensor` of probabilistic distribution of given terms
            $prob(k)$
            [1, n_ops].
    """
    if uniform_sampling:
        ones = tf.ones_like(op_coeff)
        pdist = tf.reduce_sum(ones, axis=0, keepdims=True) / tf.reduce_sum(ones)
    else:
        pdist = tf.reduce_sum(op_coeff, axis=0,
                              keepdims=True) / tf.reduce_sum(op_coeff)

    correction_factor_ops = tf.math.divide_no_nan(
        tf.ones_like(pdist, dtype=tf.float32), tf.cast(pdist, dtype=tf.float32))
    return correction_factor_ops, pdist


# TODO(jaeyoo) : this will be c++ op
def stochastic_cost_preprocessor(pauli_sums, n_programs, n_ops,
                                 uniform_sampling):
    """Helper function to sample pauli_sums.
    It can be two of the followings:
    - uniform distribution prob(k) = 1/n_ops
    - PauliSum coefficient-based probability distributions
    Args:
        pauli_sums : `tf.Tensor` of strings with shape [n_programs, n_ops]
            representing output observables for each program.
        n_programs: `tf.Tensor` of the number of programs.
        n_ops: `tf.Tensor` of the number of pauli sums.
        uniform_sampling: `tf.Tensor` of a boolean value to decide
            probabilistic distribution of samplers inside.
    Returns:
        new_pauli_sums : `tf.Tensor` of strings of the newly sampled output
            observables.
            [n_programs, n_ops=1].
        cost_relocator: `tf.Tensor` of one-hot matrix with real numbers.
            This is used to restore squeezed pauli_sums dimension at the
            post-processing.
            [n_programs, n_ops]
        n_ops: `tf.Tensor` of the new number of pauli sums = 1.
    """
    parser = _get_parse_pauli_sums()
    op_coeff = parser(pauli_sums, n_programs, n_ops)
    correction_factor_ops, pdist_ops = _get_pdist_cost(op_coeff,
                                                       uniform_sampling)

    sampled_idx = tf.transpose(
        tf.random.categorical(tf.math.log(pdist_ops), n_programs), [1, 0])

    # Construct one_hot matrix to restore the locations at the post-processing.
    new_pauli_sums = tf.reshape(
        tf.gather_nd(pauli_sums, sampled_idx, batch_dims=1), [n_programs, 1])
    cost_relocator = tf.reshape(tf.one_hot(sampled_idx, depth=n_ops),
                                [n_programs, n_ops])

    # Set the output tensor shapes
    correction_factor_ops = tf.reshape(correction_factor_ops, [1, n_ops])
    cost_relocator = cost_relocator * correction_factor_ops
    n_ops = 1

    return new_pauli_sums, cost_relocator, n_ops
