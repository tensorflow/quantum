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


def get_differentiable_sample_op_parameter_shift(backend=None,
                                                 post_process_func=None):
    """Make bitstring post-processing differentiable.

    Create a layer that will output bitstring samples taken from either a
    simulated quantum state or a real quantum computer

    Args:
        backend: Optional Backend to use. Defaults to the native TensorFlow
            Quantum simulator (None), however users may also specify a
            preconfigured cirq execution object to use instead, which must
            `cirq.Sampler`.
        post_process_func: Differentiable function from tensor of
            type `tf.int8` of shape [repetitions, num_qubits] to a scalar.
            Assumes same function is to be applied to all sampled outputs.
    """
    sample_op = circuit_execution_ops.get_sampling_op(backend)
    
    @tf.custom_gradient
    def sample_post_process_wrapper(programs, symbol_names, symbol_values,
                                    num_samples):        
        ragged_samples = sample_op(programs, symbol_names, symbol_values,
                                   num_samples)
        if post
        if ragged_samples.shape[0] 
        for i in tf.range(cpp_ragged.shape[0]):
            
            print(cpp_ragged[i].to_tensor())
        
        forward_pass_vals = 
        def gradient(grad):
            self._differentiate_sam(programs, symbol_names,
                                           symbol_values, pauli_sums,
                                           num_samples, forward_pass_vals,
                                           grad)
            return None, None, this_grad_vec, None

        return forward_pass_vals, gradient

    return sample_post_process_wrapper
