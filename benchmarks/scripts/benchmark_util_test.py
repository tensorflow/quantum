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
"""Tests for utilities related to reading/running benchmarks."""
import os
import tempfile

import tensorflow as tf

import test_log_pb2
import benchmark_util


def _make_dummy_benchmark_report():
    """Make a serialized benchmark report."""
    entries = test_log_pb2.BenchmarkEntries()
    entry = entries.entry.add()
    entry.name = "dummy_report"
    entry.iters = 1234
    entry.wall_time = 5678
    return entries.SerializeToString()


class ReadBenchmarkEntryTest(tf.test.TestCase):
    """Test reading serialized benchmark results."""

    def test_read_benchmark_entry(self):
        """Test reading test_log protobuf contents."""

        # Do temp file setup and queue teardown.
        with tempfile.NamedTemporaryFile(prefix='ReadBenchmarkEntryTest',
                                         dir=self.get_temp_dir(),
                                         delete=False) as temp:
            temp.write(_make_dummy_benchmark_report())
        self.addCleanup(lambda: os.remove(temp.name))

        res = benchmark_util.read_benchmark_entry(temp.name)
        self.assertEqual(res.name, "dummy_report")
        self.assertEqual(res.iters, 1234)
        self.assertEqual(res.wall_time, 5678)


if __name__ == '__main__':
    tf.test.main()
