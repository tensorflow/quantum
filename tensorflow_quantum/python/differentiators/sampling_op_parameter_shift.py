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
"""Differentiation of Sample post-processing with the parameter shift rule."""
import tensorflow as tf

from tensorflow_quantum.core.ops import circuit_execution_ops
from tensorflow_quantum.python.differentiators import parameter_shift_util


def get_sample_op_postprocessor(backend=None, post_process_func=None):
    """Make bitstring post-processing differentiable.

    Create an op that will post-process output bitstring samples taken from either a
    simulated quantum state or a real quantum computer

    Args:
        backend: Optional Backend to use. Defaults to the native TensorFlow
            Quantum simulator (None), however users may also specify a
            preconfigured cirq execution object to use instead, which must
            inherit `cirq.Sampler`.
        post_process_func: Differentiable function from tensor of
            type `tf.int8` of shape [repetitions, num_qubits] to scalar of
            type `tf.float32`. Given function is applied to all circuits.
    """
    sample_op = circuit_execution_ops.get_sampling_op(backend)

    @tf.function
    def sample_post_process(programs, symbol_names, symbol_values, num_samples):
        ragged_samples = sample_op(programs, symbol_names, symbol_values,
                                   num_samples)
        scalar_list = tf.TensorArray(tf.dtypes.float32, tf.shape(programs)[0])
        for i in tf.range(tf.shape(programs)[0]):
            scalar_list = scalar_list.write(
                i, post_process_func(ragged_samples[i].to_tensor()))
        scalar_list = scalar_list.stack()
        return tf.expand_dims(scalar_list, axis=-1)

    @tf.custom_gradient
    def sample_post_process_wrapper(programs, symbol_names, symbol_values,
                                    num_samples):
        forward_pass_vals = sample_post_process(
            programs, symbol_names, symbol_values, num_samples)
        
        def gradient(grad):
            # these get used a lot
            n_symbols = tf.gather(tf.shape(symbol_names), 0)
            n_programs = tf.gather(tf.shape(programs), 0)

            # Assume cirq.decompose() generates gates with at most two distinct
            # eigenvalues, which results in two parameter shifts.
            n_shifts = 2

            # STEP 1: Generate required inputs for executor
            # Deserialize programs and parse the whole parameterized gates
            # new_programs has [n_symbols, n_param_gates, n_shifts, n_programs].
            # These new_programs has programs that parameter-shift rule is applied,
            # so those programs has
            (new_programs, weights, shifts,
             n_param_gates) = parameter_shift_util.parse_programs(
                 programs, symbol_names, symbol_values, n_symbols)
            
            # Reshape & transpose new_programs, weights and shifts to fit into
            # the input format of tensorflow_quantum simulator.
            # [n_symbols, n_param_gates, n_shifts, n_programs]
            new_programs = tf.transpose(new_programs, [0, 2, 3, 1])
            weights = tf.transpose(weights, [0, 2, 3, 1])
            shifts = tf.transpose(shifts, [0, 2, 3, 1])
            
            # reshape everything to fit into expectation op correctly
            total_programs = n_programs * n_shifts * n_param_gates * n_symbols
            # tile up and then reshape to order programs correctly
            flat_programs = tf.reshape(new_programs, [total_programs])
            flat_shifts = tf.reshape(shifts, [total_programs])

            # tile up and then reshape to order ops correctly
            n_tile = n_shifts * n_param_gates * n_symbols
            flat_perturbations = tf.concat([
                tf.reshape(
                    tf.tile(tf.expand_dims(symbol_values, 0),
                            tf.stack([n_tile, 1, 1])), [total_programs, n_symbols]),
                tf.expand_dims(flat_shifts, axis=1)
            ],
                                           axis=1)
            # Append impurity symbol into symbol name
            new_symbol_names = tf.concat([
                symbol_names,
                tf.expand_dims(tf.constant(
                    parameter_shift_util._PARAMETER_IMPURITY_NAME),
                               axis=0)
            ],
                                         axis=0)
            
            # STEP 2: calculate the required expectation values
            expectations = sample_post_process(flat_programs, new_symbol_names,
                                               flat_perturbations, num_samples)
            
            # STEP 3: generate gradients according to the results
            
            # we know the rows are grouped according to which parameter
            # was perturbed, so reshape to reflect that
            grouped_expectations = tf.reshape(
                expectations,
                [n_symbols, n_shifts * n_programs * n_param_gates, -1])
            
            # now we can calculate the partial of the circuit output with
            # respect to each perturbed parameter
            def rearrange_expectations(grouped):
                
                def split_vertically(i):
                    return tf.slice(grouped, [i * n_programs, 0],
                                    [n_programs, 1])
                
                return tf.map_fn(split_vertically,
                                 tf.range(n_param_gates * n_shifts),
                                 dtype=tf.float32)
            
            # reshape so that expectations calculated on different programs are
            # separated by a dimension
            rearranged_expectations = tf.map_fn(rearrange_expectations,
                                                grouped_expectations)
            
            # now we will calculate all of the partial derivatives
            partials = tf.einsum(
                'spco,spc->sco', rearranged_expectations,
                tf.cast(
                    tf.reshape(weights,
                               [n_symbols, n_param_gates * n_shifts, n_programs]),
                    rearranged_expectations.dtype))
            
            # now apply the chain rule
            this_grad_vec = tf.einsum('sco,co -> cs', partials, grad)
            return None, None, this_grad_vec, None

        return forward_pass_vals, gradient

    return sample_post_process_wrapper
