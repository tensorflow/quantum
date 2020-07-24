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
"""Expose bindings for tfq utility ops."""
import tensorflow as tf
from tensorflow_quantum.core.ops.load_module import load_module

UTILITY_OP_MODULE = load_module("_tfq_utility_ops.so")

# pylint: disable=invalid-name
tfq_append_circuit = UTILITY_OP_MODULE.tfq_append_circuit
tfq_resolve_parameters = UTILITY_OP_MODULE.tfq_resolve_parameters


@tf.function
def padded_to_ragged(masked_state):
    """Utility `tf.function` that converts a padded tensor to ragged.

    Convert a state `tf.Tensor` padded with the value -2 to a `tf.RaggedTensor`
    using efficient boolean masking.

    Args:
        masked_state: `tf.State` tensor with -2 padding.
    Returns:
        state_ragged: State tensor without padding as a `tf.RaggedTensor`.
    """
    # All valid values will be < 1, anything > 1 is a padding entry.
    abs_state = tf.abs(tf.cast(masked_state, tf.float32))
    mask = tf.math.less(abs_state, tf.constant(1.1, dtype=abs_state.dtype))
    state_ragged = tf.ragged.boolean_mask(masked_state, mask)
    return state_ragged


@tf.function
def padded_to_ragged2d(masked_state):
    """Utility `tf.function` that converts a 2d padded tensor to ragged.

    Convert a [batch, dim, dim] `tf.Tensor` padded with -2 to a
    `tf.RaggedTensor` using 2d boolean masking.

    Args:
        masked_state: `tf.Tensor` of rank 3 with -2 padding.
    Returns:
        state_ragged: `tf.RaggedTensor` of rank 3 with no -2 padding where the
            outer most dimensions are now ragged instead of padded.
    """
    # All valid values will be < 1, anything > 1 is a padding entry.
    col_mask = tf.abs(tf.cast(masked_state[:, 0], tf.float32)) < 1.1
    masked = tf.ragged.boolean_mask(masked_state, col_mask)
    return padded_to_ragged(masked)
