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
"""Module for tfq.core.ops.*"""

# Import getters for constructing ops.
from tensorflow_quantum.core.ops.circuit_execution_ops import (
    get_expectation_op, get_sampled_expectation_op, get_sampling_op,
    get_state_op)

from tensorflow_quantum.core.ops.tfq_unitary_op import calculate_unitary
from tensorflow_quantum.core.ops.tfq_utility_ops import (padded_to_ragged,
                                                         padded_to_ragged2d,
                                                         resolve_parameters,
                                                         tfq_append_circuit)
