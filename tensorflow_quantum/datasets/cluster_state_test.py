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
"""Test the cluster state dataset."""
# Remove PYTHONPATH collisions for protobuf.
# pylint: disable=wrong-import-position
import sys

NEW_PATH = [x for x in sys.path if 'com_google_protobuf' not in x]
sys.path = NEW_PATH
# pylint: enable=wrong-import-position

import tensorflow as tf
import cirq

from tensorflow_quantum.datasets import cluster_state


class ClusterStateDataTest(tf.test.TestCase):
    """Small test to make sure dataset for ClusterState works."""

    def test_errors(self):
        """Test that it errors on invalid qubits."""
        with self.assertRaisesRegex(TypeError, expected_regex='must be a list'):
            cluster_state.excited_cluster_states('junk')

        with self.assertRaisesRegex(ValueError,
                                    expected_regex='cirq.GridQubit'):
            cluster_state.excited_cluster_states([cirq.NamedQubit('bob')])

        with self.assertRaisesRegex(ValueError,
                                    expected_regex='more than two qubits.'):
            cluster_state.excited_cluster_states(cirq.GridQubit.rect(1, 2))

    def test_creation(self):
        """Test that it returns the correct number of circuits."""
        qubits = cirq.GridQubit.rect(1, 5)
        circuits, labels = cluster_state.excited_cluster_states(qubits)

        self.assertEqual(len(circuits), 6)
        self.assertEqual(len(labels), 6)


if __name__ == '__main__':
    tf.test.main()
