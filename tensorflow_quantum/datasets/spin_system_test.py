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
"""Test the spin system dataset"""
import tensorflow as tf
import numpy as np
import cirq
from tensorflow_quantum.datasets import spin_system


class SpinSystemDataTest(tf.test.TestCase):
    """Small test to make sure dataset for ClusterState works."""

    def test_errors(self):
        """Test that it errors on invalid arguments."""
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='Supported number of'):
            qbs = cirq.GridQubit.rect(3, 1)
            spin_system.tfi_chain(qbs)
        with self.assertRaisesRegex(
                ValueError, expected_regex='Supported boundary conditions'):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain(qbs, 'open')
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='expected str, bytes '):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain(qbs, data_dir=123)
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='must be a list of'):
            spin_system.tfi_chain(['bob'])

        with self.assertRaisesRegex(TypeError,
                                    expected_regex='must be a one-dimensional'):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain([qbs])

    def test_fidelity(self):
        """Test that it returns the correct number of circuits."""
        supported_nspins = [4, 8, 12, 16]
        for nspins in supported_nspins:
            qbs = cirq.GridQubit.rect(nspins, 1)
            circuits, _, _, addinfo, _ = spin_system.tfi_chain(
                qbs,
                'closed',
            )
            fidelities = []
            for n in range(len(addinfo)):
                phi = cirq.Simulator().simulate(circuits[n]).final_state
                gs = addinfo[n].gs
                fidelities.append(np.abs(np.conj(np.dot(gs, phi))))
            assert all([np.isclose(fid, 1.0, rtol=1e-3) for fid in fidelities])

    def test_paulisum(self):
        """Test that hamiltonian returns a PauliSum"""
        qbs = cirq.GridQubit.rect(4, 1)
        pauli_sums = spin_system.tfi_chain(qbs, 'closed')[2]
        assert all(isinstance(ps, cirq.PauliSum) for ps in pauli_sums)


if __name__ == '__main__':
    tf.test.main()
