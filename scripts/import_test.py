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
"""Tests to check if importing `tfq` APIs is successful or not."""
import tensorflow_quantum as tfq


def test_imports():
    """Test that pip package was built with proper structure."""

    # Top level modules.
    _ = tfq.layers
    _ = tfq.differentiators

    # Ops and Op getters.
    _ = tfq.get_expectation_op
    _ = tfq.get_sampled_expectation_op
    _ = tfq.get_sampling_op
    _ = tfq.get_state_op
    _ = tfq.get_unitary_op
    _ = tfq.append_circuit
    _ = tfq.padded_to_ragged
    _ = tfq.padded_to_ragged2d
    _ = tfq.resolve_parameters

    # Util functions.
    _ = tfq.convert_to_tensor
    _ = tfq.get_quantum_concurrent_op_mode
    _ = tfq.from_tensor
    _ = tfq.set_quantum_concurrent_op_mode
    _ = tfq.util.get_supported_gates
    _ = tfq.util.exponential

    # Keras layers.
    _ = tfq.layers.AddCircuit
    _ = tfq.layers.Expectation
    _ = tfq.layers.Sample
    _ = tfq.layers.State
    _ = tfq.layers.SampledExpectation
    _ = tfq.layers.ControlledPQC
    _ = tfq.layers.PQC

    # Differentiators.
    _ = tfq.differentiators.Adjoint
    _ = tfq.differentiators.ForwardDifference
    _ = tfq.differentiators.CentralDifference
    _ = tfq.differentiators.LinearCombination
    _ = tfq.differentiators.ParameterShift
    _ = tfq.differentiators.Differentiator

    # Datasets.
    _ = tfq.datasets.excited_cluster_states
    _ = tfq.datasets.tfi_chain
    _ = tfq.datasets.xxz_chain


if __name__ == "__main__":
    test_imports()
