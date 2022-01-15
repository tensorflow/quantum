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
"""Module for tfq.core.ops.math_ops.*"""

from tensorflow_quantum.core.ops.math_ops.fidelity_op import fidelity
from tensorflow_quantum.core.ops.math_ops.inner_product_op import inner_product
from tensorflow_quantum.core.ops.math_ops.simulate_mps import (
    mps_1d_expectation, mps_1d_sample, mps_1d_sampled_expectation)
