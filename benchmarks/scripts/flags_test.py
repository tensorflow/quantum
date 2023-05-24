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
"""Tests for benchmark command line flags."""

import tensorflow as tf
from benchmarks.scripts import flags


class FlagsTest(tf.test.TestCase):
    """Test the flag and test-flag interface."""

    def test_test_flags_defaults(self):
        """Test default values in TEST_FLAGS conform to flag defaults."""
        params = flags.TEST_FLAGS()
        assert params.n_runs == 1
        assert params.n_burn == 0
        assert params.n_iters == 1

    def test_test_flags(self):
        """Test that kwargs convert to attributes."""
        params = flags.TEST_FLAGS(garbage="garbage value", other_garbage=123)
        assert params.garbage == "garbage value"
        assert params.other_garbate == 123


if __name__ == "__main__":
    tf.test.main()
