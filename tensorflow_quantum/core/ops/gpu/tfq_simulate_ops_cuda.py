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
"""Module to register cuda simulation python op."""
import os
import tensorflow as tf
from tensorflow_quantum.core.ops.load_module import load_module

SIM_OP_MODULE = load_module(os.path.join("gpu", "_tfq_simulate_ops_gpu.so"))


def tfq_simulate_expectation(programs, symbol_names, symbol_values, pauli_sums,
                             num_threads_in_sim=32, block_count=2,
                             thread_per_block=32, cpu_cycle=200):
    """Calculates the expectation value of circuits wrt some operator(s).

    Args:
        programs: `tf.Tensor` of strings with shape [batch_size] containing
            the string representations of the circuits to be executed.
        symbol_names: `tf.Tensor` of strings with shape [n_params], which
            is used to specify the order in which the values in
            `symbol_values` should be placed inside of the circuits in
            `programs`.
        symbol_values: `tf.Tensor` of real numbers with shape
            [batch_size, n_params] specifying parameter values to resolve
            into the circuits specificed by programs, following the ordering
            dictated by `symbol_names`.
        pauli_sums: `tf.Tensor` of strings with shape [batch_size, n_ops]
            containing the string representation of the operators that will
            be used on all of the circuits in the expectation calculations.
        num_threads_in_sim: Python integer to specify the number of threads in
            QSim SimulatorCUDA. One of [32, 64, 128, 256]
        block_count: Python integer for the number of blocks in QSim
            StateSpaceCUDA. One of [2, 16]
        thread_per_block: Python integer to specify the number of threads per
            block in StateSpaceCUDA. One of [32, 64, 128, 256, 512, 1024]
        cpu_cycle: Python integer to specify the number of CPU cycles.

    Returns:
        `tf.Tensor` with shape [batch_size, n_ops] that holds the
            expectation value for each circuit with each op applied to it
            (after resolving the corresponding parameters in).
    """
    return SIM_OP_MODULE.tfq_simulate_expectation_gpu_cpu(
        programs, symbol_names, tf.cast(symbol_values, tf.float32), pauli_sums,
        num_threads_in_sim=num_threads_in_sim,
        block_count=block_count,
        thread_per_block=thread_per_block,
        cpu_cycle=cpu_cycle)

