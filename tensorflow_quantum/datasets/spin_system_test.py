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
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='must be an integer'):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain(qbs, 'junk')
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='Supported number of'):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain(qbs, 3)
        with self.assertRaisesRegex(
                ValueError, expected_regex='Supported boundary conditions'):
            qbs = cirq.GridQubit.rect(4, 1)
            spin_system.tfi_chain(qbs, 4, 'open')
        with self.assertRaisesRegex(TypeError,
                                    expected_regex='must be a list of'):
            spin_system.tfi_chain(['bob'], 4)
        with self.assertRaisesRegex(
                ValueError, expected_regex='cirq.Gridqubit objects with shape'):
            qbs = cirq.GridQubit.rect(2, 2)
            spin_system.tfi_chain(qbs, 4)
        with self.assertRaisesRegex(ValueError,
                                    expected_regex='Expected 4 cirq.Gridqubit'):
            qbs = cirq.GridQubit.rect(3, 1)
            spin_system.tfi_chain(qbs, 4)

    def test_fidelity(self):
        """Test that it returns the correct number of circuits."""
        nspins = 4
        qbs = cirq.GridQubit.rect(nspins, 1)
        circuit, resolved_parameters, system = spin_system.tfi_chain(
            qbs, nspins, 'closed')
        fidelities = []
        for n in range(80):
            phi = cirq.Simulator().simulate(circuit,
                                            resolved_parameters[n]).final_state
            gs = system[n].ground_state()
            fidelities.append(
                np.abs(
                    np.conj(gs[:, 0].reshape((1, -1))) @ phi.reshape(
                        (-1, 1)))[0])
        assert all([np.isclose(fid, 1.0, rtol=1e-3) for fid in fidelities])


if __name__ == '__main__':
    tf.test.main()
