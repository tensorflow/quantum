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
# =============================================================================
"""Utility functions for benchmark tools."""
import tensorflow as tf
import test_log_pb2


def read_benchmark_entry(f):
    s = tf.io.gfile.GFile(f, "rb").read()
    entries = test_log_pb2.BenchmarkEntries.FromString(s)
    return entries.entry[0]
