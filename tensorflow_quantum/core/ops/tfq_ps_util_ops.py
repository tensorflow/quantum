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
"""Expose bindings for ParameterShift C++ ops."""
from tensorflow_quantum.core.ops.load_module import load_module

PS_UTIL_MODULE = load_module("_tfq_ps_utils.so")

# pylint: disable=invalid-name
tfq_ps_decompose = PS_UTIL_MODULE.tfq_ps_decompose
tfq_ps_symbol_replace = PS_UTIL_MODULE.tfq_ps_symbol_replace
tfq_ps_weights_from_symbols = PS_UTIL_MODULE.tfq_ps_weights_from_symbols
